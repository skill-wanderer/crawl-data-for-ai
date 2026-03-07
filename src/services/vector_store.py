"""
Qdrant vector database service with Gemini embeddings.

Handles storing, querying, and managing crawled page data in Qdrant.
Uses Google Gemini API for generating text embeddings.
"""

import logging
import uuid
import textwrap

from google import genai
from qdrant_client import QdrantClient
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

# Maximum characters per chunk to stay within embedding model limits
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks for embedding."""
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
        )
        self.gemini_client = genai.Client(api_key=self.settings.gemini_api_key)
        self._ensure_collection()

    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.settings.qdrant_collection not in collections:
            self.client.create_collection(
                collection_name=self.settings.qdrant_collection,
                vectors_config=VectorParams(
                    size=self.settings.embedding_dimension,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created collection: {self.settings.qdrant_collection}")

    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts using Gemini."""
        embeddings = []
        # Process in batches of 100 (Gemini batch limit)
        for i in range(0, len(texts), 100):
            batch = texts[i : i + 100]
            result = self.gemini_client.models.embed_content(
                model=self.settings.embedding_model,
                contents=batch,
            )
            embeddings.extend([e.values for e in result.embeddings])
        return embeddings

    def store_pages(self, pages: list[CrawledPage]) -> int:
        """Store crawled pages in Qdrant. Returns number of points stored."""
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
                },
            )
            points.append(point)

        # Upsert in batches of 100
        for i in range(0, len(points), 100):
            batch = points[i : i + 100]
            self.client.upsert(
                collection_name=self.settings.qdrant_collection,
                points=batch,
            )

        logger.info(f"Stored {len(points)} vectors for {len(pages)} pages.")
        return len(points)

    def delete_by_domain(self, domain: str) -> int:
        """Delete all points belonging to a domain. Returns count of deleted points."""
        # First count the points that will be deleted
        domain_clean = domain.lower().removeprefix("www.").removeprefix("http://").removeprefix("https://").split("/")[0]

        count_result = self.client.count(
            collection_name=self.settings.qdrant_collection,
            count_filter=Filter(
                must=[FieldCondition(key="domain", match=MatchValue(value=domain_clean))]
            ),
            exact=True,
        )
        count = count_result.count

        if count > 0:
            self.client.delete(
                collection_name=self.settings.qdrant_collection,
                points_selector=Filter(
                    must=[FieldCondition(key="domain", match=MatchValue(value=domain_clean))]
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
            results = self.client.scroll(
                collection_name=self.settings.qdrant_collection,
                limit=100,
                offset=offset,
                with_payload=["domain"],
                with_vectors=False,
            )
            points, next_offset = results
            for point in points:
                domain = point.payload.get("domain", "unknown")
                stats[domain] = stats.get(domain, 0) + 1

            if next_offset is None:
                break
            offset = next_offset

        return stats

    def get_collection_info(self) -> dict:
        """Get collection statistics."""
        info = self.client.get_collection(self.settings.qdrant_collection)
        return {
            "name": self.settings.qdrant_collection,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status.value,
        }
