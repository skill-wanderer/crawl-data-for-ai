"""
Qdrant vector database service with Gemini embeddings.

Handles storing, querying, and managing crawled page data in Qdrant.
Uses Google Gemini API for generating text embeddings.
"""

import logging
import time
import uuid
from typing import Callable, TypeVar

import grpc
import httpx
from google import genai
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
            "Qdrant client: %s:%s (grpc=%s, timeout=%ss, upsert_batch=%s)",
            self.settings.qdrant_host,
            self.settings.qdrant_grpc_port if self.settings.qdrant_prefer_grpc else self.settings.qdrant_port,
            self.settings.qdrant_prefer_grpc,
            self.settings.qdrant_timeout,
            self.settings.qdrant_upsert_batch_size,
        )
        self.gemini_client = genai.Client(api_key=self.settings.gemini_api_key)
        self._ensure_collection()

    # --- Retry helper ---

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        """Transport errors are retryable; so are 429 and 5xx responses."""
        if isinstance(exc, UnexpectedResponse):
            return exc.status_code == 429 or exc.status_code >= 500
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
            result = self.gemini_client.models.embed_content(
                model=self.settings.embedding_model,
                contents=batch,
            )
            embeddings.extend([e.values for e in result.embeddings])
        return embeddings

    def store_pages(self, pages: list[CrawledPage], crawl_id: str | None = None) -> int:
        """Store crawled pages in Qdrant. Returns number of points stored.

        Every point is tagged with `crawl_id` so a caller can delete the
        previous generation only after this one has landed successfully
        (see `delete_by_domain(exclude_crawl_id=...)`).
        """
        crawl_id = crawl_id or str(uuid.uuid4())
        all_chunks: list[dict] = []

        for page in pages:
            # Build a rich text representation for each page
            page_text = f"Title: {page.title}\n"
            if page.meta_description:
                page_text += f"Description: {page.meta_description}\n"
            page_text += f"\n{page.content}"

            chunks = _chunk_text(page_text)
            for idx, chunk in enumerate(chunks):
                all_chunks.append({
                    "text": chunk,
                    "url": page.url,
                    "domain": page.domain,
                    "title": page.title,
                    "meta_description": page.meta_description,
                    "headings": page.headings,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                })

        if not all_chunks:
            logger.warning("No chunks to store.")
            return 0

        logger.info(f"Generating embeddings for {len(all_chunks)} chunks...")
        texts = [c["text"] for c in all_chunks]
        embeddings = self._get_embeddings(texts)

        points = []
        for chunk_data, embedding in zip(all_chunks, embeddings):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": chunk_data["text"],
                    "url": chunk_data["url"],
                    "domain": chunk_data["domain"],
                    "title": chunk_data["title"],
                    "meta_description": chunk_data["meta_description"],
                    "headings": chunk_data["headings"],
                    "chunk_index": chunk_data["chunk_index"],
                    "total_chunks": chunk_data["total_chunks"],
                    "crawl_id": crawl_id,
                },
            )
            points.append(point)

        # Upsert in batches. Point IDs are fixed before the loop, so a retried
        # batch overwrites itself rather than creating duplicates.
        batch_size = self.settings.qdrant_upsert_batch_size
        total_batches = (len(points) + batch_size - 1) // batch_size
        for batch_num, i in enumerate(range(0, len(points), batch_size), start=1):
            batch = points[i : i + batch_size]
            self._with_retry(
                f"upsert batch {batch_num}/{total_batches}",
                lambda b=batch: self.client.upsert(
                    collection_name=self.settings.qdrant_collection,
                    points=b,
                ),
            )
            logger.info(
                "Upserted batch %d/%d (%d points)", batch_num, total_batches, len(batch)
            )

        logger.info(f"Stored {len(points)} vectors for {len(pages)} pages.")
        return len(points)

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

    def delete_by_domain(self, domain: str, exclude_crawl_id: str | None = None) -> int:
        """Delete points for a domain. Returns count of deleted points.

        If `exclude_crawl_id` is given, points from that crawl are kept - use it
        to retire the previous generation only once the new one is stored.
        """
        domain_clean = self._clean_domain(domain)

        conditions = Filter(
            must=[FieldCondition(key="domain", match=MatchValue(value=domain_clean))],
            must_not=(
                [FieldCondition(key="crawl_id", match=MatchValue(value=exclude_crawl_id))]
                if exclude_crawl_id
                else None
            ),
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
