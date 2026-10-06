"""
Qdrant vector database service with Gemini embeddings.

Handles storing, querying, and managing crawled page data in Qdrant.
Uses Google Gemini API for generating text embeddings.
"""

import hashlib
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Callable, TypeVar

import grpc
import httpx
from google import genai
from google.genai import errors as genai_errors
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

from src.services.crawler import CrawledPage
from src.config import get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Transport-level failures worth retrying. Covers connect/read/write/pool
# timeouts and dropped connections, which is what a long-haul or tunnelled
# link produces under load.
_RETRYABLE_EXCEPTIONS = (
    httpx.TransportError,
    ResponseHandlingException,
    grpc.RpcError,
)


def _chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
    """Split text into overlapping chunks for embedding."""
    settings = get_settings()
    chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
    overlap = overlap if overlap is not None else settings.chunk_overlap

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap

    return chunks


def _content_hash(text: str) -> str:
    """Return a stable identity for the exact text sent to the embedder."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _point_id(domain: str, url: str, content_hash: str) -> str:
    """Return the same Qdrant point ID for the same source chunk."""
    identity = f"{domain}\0{url}\0{content_hash}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, identity))


@dataclass
class AdditiveCrawlSession:
    """Progress and identity shared by one streaming domain crawl."""

    domain: str
    crawl_id: str
    added: int = 0
    skipped: int = 0
    pages_processed: int = 0


class VectorStore:
    """Manages Qdrant vector storage with Gemini embeddings."""

    def __init__(self):
        self.settings = get_settings()
        self.client = QdrantClient(
            host=self.settings.qdrant_host,
            port=self.settings.qdrant_port,
            grpc_port=self.settings.qdrant_grpc_port,
            prefer_grpc=self.settings.qdrant_prefer_grpc,
            https=self.settings.qdrant_https or None,
            api_key=self.settings.qdrant_api_key or None,
            timeout=self.settings.qdrant_timeout,
        )
        logger.info(
            "Qdrant client: %s:%s (grpc=%s, timeout=%ss, writes=per-chunk)",
            self.settings.qdrant_host,
            self.settings.qdrant_grpc_port if self.settings.qdrant_prefer_grpc else self.settings.qdrant_port,
            self.settings.qdrant_prefer_grpc,
            self.settings.qdrant_timeout,
        )
        self.gemini_client = genai.Client(api_key=self.settings.gemini_api_key)
        self._ensure_collection()

    # --- Retry helper ---

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        """Transport errors are retryable; so are 429 and 5xx responses."""
        if isinstance(exc, UnexpectedResponse):
            return exc.status_code == 429 or exc.status_code >= 500
        if isinstance(exc, genai_errors.APIError):
            return exc.code == 429 or exc.code >= 500
        return isinstance(exc, _RETRYABLE_EXCEPTIONS)

    def _with_retry(self, op_name: str, fn: Callable[[], T]) -> T:
        """Run fn, retrying transient failures with exponential backoff."""
        delay = self.settings.qdrant_retry_base_delay
        attempts = max(1, self.settings.qdrant_max_retries)

        for attempt in range(1, attempts + 1):
            try:
                return fn()
            except Exception as exc:
                if not self._is_retryable(exc) or attempt == attempts:
                    raise
                logger.warning(
                    "%s failed (attempt %d/%d): %s: %s - retrying in %.1fs",
                    op_name, attempt, attempts, type(exc).__name__, exc, delay,
                )
                time.sleep(delay)
                delay = min(delay * 2, self.settings.qdrant_retry_max_delay)

        raise AssertionError("unreachable")

    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        existing = self._with_retry("get_collections", lambda: self.client.get_collections())
        collections = [c.name for c in existing.collections]
        if self.settings.qdrant_collection not in collections:
            self._with_retry(
                "create_collection",
                lambda: self.client.create_collection(
                    collection_name=self.settings.qdrant_collection,
                    vectors_config=VectorParams(
                        size=self.settings.embedding_dimension,
                        distance=Distance.COSINE,
                    ),
                ),
            )
            logger.info(f"Created collection: {self.settings.qdrant_collection}")

    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts using Gemini."""
        embeddings = []
        batch_size = self.settings.embedding_batch_size
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            result = self._with_retry(
                "Gemini embedding",
                lambda b=batch: self.gemini_client.models.embed_content(
                    model=self.settings.embedding_model,
                    contents=b,
                ),
            )
            embeddings.extend([e.values for e in result.embeddings])
        return embeddings

    def _get_existing_chunk_keys(
        self, domain: str, url: str | None = None
    ) -> set[tuple[str, str]]:
        """Return stored ``(url, content_hash)`` keys, optionally for one URL.

        Older points did not have ``content_hash`` in their payload, so derive
        it from their stored text during the transition instead of duplicating
        those vectors on the first additive recrawl.
        """
        domain_clean = self._clean_domain(domain)
        must = [FieldCondition(key="domain", match=MatchValue(value=domain_clean))]
        if url:
            must.append(FieldCondition(key="url", match=MatchValue(value=url)))
        existing_filter = Filter(must=must)
        existing: set[tuple[str, str]] = set()
        offset = None

        while True:
            points, next_offset = self._with_retry(
                "scroll existing chunks",
                lambda o=offset: self.client.scroll(
                    collection_name=self.settings.qdrant_collection,
                    scroll_filter=existing_filter,
                    limit=self.settings.qdrant_scroll_batch_size,
                    offset=o,
                    with_payload=["url", "text", "content_hash"],
                    with_vectors=False,
                ),
            )
            for point in points:
                payload = point.payload or {}
                url = payload.get("url")
                text = payload.get("text")
                content_hash = payload.get("content_hash")
                if url and (content_hash or isinstance(text, str)):
                    existing.add((url, content_hash or _content_hash(text)))

            if next_offset is None:
                break
            offset = next_offset

        return existing

    def start_additive_crawl(
        self, domain: str, crawl_id: str | None = None
    ) -> AdditiveCrawlSession:
        """Create progress state for a streaming, additive domain crawl."""
        domain_clean = self._clean_domain(domain)
        return AdditiveCrawlSession(
            domain=domain_clean,
            crawl_id=crawl_id or str(uuid.uuid4()),
        )

    def store_page(
        self,
        page: CrawledPage,
        session: AdditiveCrawlSession,
        on_checkpoint: Callable[[int], None] | None = None,
    ) -> int:
        """Check and immediately store every unseen chunk from one URL.

        Existing keys are fetched from Qdrant only for this page's URL, keeping
        memory bounded for large sites. Each successful write updates the local
        key set before the next chunk, providing a durable checkpoint if a
        later operation fails.
        """
        page_domain = self._clean_domain(page.domain)
        if page_domain != session.domain:
            raise ValueError(
                f"Page domain {page_domain!r} does not match crawl session "
                f"domain {session.domain!r}"
            )

        page_text = f"Title: {page.title}\n"
        if page.meta_description:
            page_text += f"Description: {page.meta_description}\n"
        page_text += f"\n{page.content}"

        chunks = _chunk_text(page_text)
        existing_keys = self._get_existing_chunk_keys(session.domain, page.url)
        added_for_page = 0
        for idx, chunk in enumerate(chunks):
            content_hash = _content_hash(chunk)
            key = (page.url, content_hash)
            if key in existing_keys:
                session.skipped += 1
                continue

            logger.info(
                "Embedding and storing new chunk %d/%d from %s",
                idx + 1,
                len(chunks),
                page.url,
            )
            embeddings = self._get_embeddings([chunk])
            if len(embeddings) != 1:
                raise RuntimeError("Gemini did not return an embedding for the chunk")

            point = PointStruct(
                id=_point_id(session.domain, page.url, content_hash),
                vector=embeddings[0],
                payload={
                    "text": chunk,
                    "url": page.url,
                    "domain": session.domain,
                    "title": page.title,
                    "meta_description": page.meta_description,
                    "headings": page.headings,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "content_hash": content_hash,
                    "crawl_id": session.crawl_id,
                },
            )
            self._with_retry(
                f"upsert chunk {idx + 1}/{len(chunks)} from {page.url}",
                lambda p=point: self.client.upsert(
                    collection_name=self.settings.qdrant_collection,
                    points=[p],
                ),
            )

            # Mark it as existing only after Qdrant confirms the write.
            existing_keys.add(key)
            session.added += 1
            added_for_page += 1
            if on_checkpoint:
                on_checkpoint(session.added)
            logger.info(
                "Stored checkpoint %d: %s chunk %d",
                session.added,
                page.url,
                idx + 1,
            )

        session.pages_processed += 1
        return added_for_page

    def store_pages(self, pages: list[CrawledPage], crawl_id: str | None = None) -> int:
        """Add only previously unseen page chunks to Qdrant.

        A chunk is considered existing when its source URL and exact embedded
        text match a stored point in the same domain. Existing points are never
        updated or removed here; delete the domain first when a complete fresh
        crawl is required. Every unseen chunk is embedded and durably upserted
        before the next chunk is processed, so a later retry can resume after
        the last successful checkpoint. Returns the number of new points stored.
        """
        if not pages:
            logger.warning("No pages to store.")
            return 0

        sessions: dict[str, AdditiveCrawlSession] = {}
        for page in pages:
            domain = self._clean_domain(page.domain)
            if domain not in sessions:
                sessions[domain] = self.start_additive_crawl(domain, crawl_id)
            self.store_page(page, sessions[domain])

        added = sum(session.added for session in sessions.values())
        skipped = sum(session.skipped for session in sessions.values())

        logger.info(
            "Added %d new vectors from %d crawled pages; skipped %d existing chunks.",
            added,
            len(pages),
            skipped,
        )
        return added

    @staticmethod
    def _clean_domain(domain: str) -> str:
        """Normalize a domain string to the bare host used in payloads."""
        return (
            domain.lower()
            .removeprefix("http://")
            .removeprefix("https://")
            .removeprefix("www.")
            .split("/")[0]
        )

    def delete_by_domain(self, domain: str) -> int:
        """Delete all points for a domain. Returns count of deleted points."""
        domain_clean = self._clean_domain(domain)

        conditions = Filter(
            must=[FieldCondition(key="domain", match=MatchValue(value=domain_clean))],
        )

        count_result = self._with_retry(
            "count",
            lambda: self.client.count(
                collection_name=self.settings.qdrant_collection,
                count_filter=conditions,
                exact=True,
            ),
        )
        count = count_result.count

        if count > 0:
            self._with_retry(
                "delete",
                lambda: self.client.delete(
                    collection_name=self.settings.qdrant_collection,
                    points_selector=conditions,
                ),
            )
            logger.info(f"Deleted {count} vectors for domain: {domain_clean}")

        return count

    def get_domain_stats(self) -> dict[str, int]:
        """Get count of stored vectors per domain."""
        # Scroll through all points to aggregate domain stats
        stats: dict[str, int] = {}
        offset = None
        while True:
            points, next_offset = self._with_retry(
                "scroll",
                lambda o=offset: self.client.scroll(
                    collection_name=self.settings.qdrant_collection,
                    limit=self.settings.qdrant_scroll_batch_size,
                    offset=o,
                    with_payload=["domain"],
                    with_vectors=False,
                ),
            )
            for point in points:
                domain = point.payload.get("domain", "unknown")
                stats[domain] = stats.get(domain, 0) + 1

            if next_offset is None:
                break
            offset = next_offset

        return stats

    def get_collection_info(self) -> dict:
        """Get collection statistics."""
        info = self._with_retry(
            "get_collection",
            lambda: self.client.get_collection(self.settings.qdrant_collection),
        )
        return {
            "name": self.settings.qdrant_collection,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status.value,
        }
