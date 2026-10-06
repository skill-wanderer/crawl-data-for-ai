from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src import app as app_module
from src.services.crawler import CrawledPage


class FakeStreamingCrawler:
    events = []
    pages_to_crawl = [
        CrawledPage(
            url="https://example.com/one",
            domain="example.com",
            title="One",
            content="First page content long enough to crawl.",
        ),
        CrawledPage(
            url="https://example.com/two",
            domain="example.com",
            title="Two",
            content="Second page content long enough to crawl.",
        ),
    ]

    def __init__(self, url, include_subdomains=False):
        self.url = url
        self.pages_crawled = 0

    async def crawl(self, on_page=None, collect_pages=True):
        if on_page is None or collect_pages:
            raise AssertionError("The application must use streaming page handling")
        for page in self.pages_to_crawl:
            self.events.append(("crawl", page.url))
            self.pages_crawled += 1
            on_page(page)
        return []

    def stop(self):
        pass


class FakeStreamingStore:
    def __init__(self, events):
        self.events = events

    def start_additive_crawl(self, domain):
        self.events.append(("session", domain))
        return SimpleNamespace(added=0)

    def store_page(self, page, session, on_checkpoint=None):
        self.events.append(("store", page.url))
        session.added += 1
        if on_checkpoint:
            on_checkpoint(session.added)
        return 1


class StreamingWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        app_module.crawl_jobs.clear()
        app_module.crawl_tasks.clear()
        FakeStreamingCrawler.events = []

    async def test_each_url_is_stored_before_the_next_url_is_crawled(self):
        store = FakeStreamingStore(FakeStreamingCrawler.events)

        with patch.object(app_module, "WebCrawler", FakeStreamingCrawler):
            await app_module.crawl_and_store("https://example.com", store)

        self.assertEqual(
            FakeStreamingCrawler.events,
            [
                ("session", "example.com"),
                ("crawl", "https://example.com/one"),
                ("store", "https://example.com/one"),
                ("crawl", "https://example.com/two"),
                ("store", "https://example.com/two"),
            ],
        )
        self.assertEqual(
            app_module.crawl_jobs["example.com"],
            {
                "status": "completed",
                "pages_crawled": 2,
                "vectors_stored": 2,
                "error": None,
            },
        )


if __name__ == "__main__":
    unittest.main()
