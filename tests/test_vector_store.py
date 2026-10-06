from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src.services.crawler import CrawledPage
from src.services.vector_store import VectorStore, _content_hash, _point_id


class FakeQdrantClient:
    def __init__(self, existing_payloads=None):
        self.existing_payloads = existing_payloads or []
        self.upserted = []

    def scroll(self, **kwargs):
        records = [SimpleNamespace(payload=payload) for payload in self.existing_payloads]
        return records, None

    def upsert(self, **kwargs):
        self.upserted.extend(kwargs["points"])


def make_store(existing_payloads=None):
    store = VectorStore.__new__(VectorStore)
    store.settings = SimpleNamespace(
        qdrant_collection="test_collection",
        qdrant_scroll_batch_size=100,
        qdrant_upsert_batch_size=10,
        qdrant_max_retries=1,
        qdrant_retry_base_delay=0,
        qdrant_retry_max_delay=0,
    )
    store.client = FakeQdrantClient(existing_payloads)
    store._get_embeddings = Mock(side_effect=lambda texts: [[0.1] for _ in texts])
    return store


class AdditiveStoreTests(unittest.TestCase):
    def setUp(self):
        self.page = CrawledPage(
            url="https://example.com/about",
            domain="example.com",
            title="About",
            content="This is enough page content for an additive crawl test.",
            meta_description="Example description",
            headings=["About"],
        )

    def test_legacy_point_without_hash_is_not_added_again(self):
        stored_text = (
            "Title: About\n"
            "Description: Example description\n\n"
            "This is enough page content for an additive crawl test."
        )
        store = make_store([{"url": self.page.url, "text": stored_text}])

        added = store.store_pages([self.page], crawl_id="crawl-test")

        self.assertEqual(added, 0)
        store._get_embeddings.assert_not_called()
        self.assertEqual(store.client.upserted, [])

    def test_only_unseen_chunks_are_embedded_and_upserted(self):
        existing_text = "already stored"
        store = make_store([
            {
                "url": self.page.url,
                "text": existing_text,
                "content_hash": _content_hash(existing_text),
            }
        ])

        with patch(
            "src.services.vector_store._chunk_text",
            return_value=[existing_text, "new content", "new content"],
        ):
            added = store.store_pages([self.page], crawl_id="crawl-test")

        self.assertEqual(added, 1)
        store._get_embeddings.assert_called_once_with(["new content"])
        self.assertEqual(len(store.client.upserted), 1)

        point = store.client.upserted[0]
        expected_hash = _content_hash("new content")
        self.assertEqual(
            str(point.id),
            _point_id(self.page.domain, self.page.url, expected_hash),
        )
        self.assertEqual(point.payload["content_hash"], expected_hash)
        self.assertEqual(point.payload["chunk_index"], 1)
        self.assertEqual(point.payload["total_chunks"], 3)
        self.assertEqual(point.payload["crawl_id"], "crawl-test")

    def test_same_text_at_another_url_is_kept_as_a_separate_source(self):
        text = "shared content"
        store = make_store([
            {
                "url": "https://example.com/first",
                "text": text,
                "content_hash": _content_hash(text),
            }
        ])
        second_page = CrawledPage(
            url="https://example.com/second",
            domain="example.com",
            title="Second",
            content="Second page content long enough to crawl.",
        )

        with patch("src.services.vector_store._chunk_text", return_value=[text]):
            added = store.store_pages([second_page], crawl_id="crawl-test")

        self.assertEqual(added, 1)
        self.assertEqual(store.client.upserted[0].payload["url"], second_page.url)


if __name__ == "__main__":
    unittest.main()
