import unittest
from unittest.mock import patch

from src.services.crawler import WebCrawler


class FakeResponse:
    status = 200
    headers = {"content-type": "text/html; charset=utf-8"}


class FakePage:
    def __init__(self, html, events):
        self.html = html
        self.events = events

    async def goto(self, url, **kwargs):
        self.events.append(("crawl", url))
        return FakeResponse()

    async def wait_for_timeout(self, milliseconds):
        pass

    async def content(self):
        return self.html

    async def close(self):
        pass


class FakeBrowserContext:
    def __init__(self, html_pages, events):
        self.html_pages = iter(html_pages)
        self.events = events

    async def new_page(self):
        return FakePage(next(self.html_pages), self.events)


class FakeBrowser:
    def __init__(self, html_pages, events):
        self.context = FakeBrowserContext(html_pages, events)

    async def new_context(self, **kwargs):
        return self.context

    async def close(self):
        pass


class FakeChromium:
    def __init__(self, html_pages, events):
        self.html_pages = html_pages
        self.events = events

    async def launch(self, **kwargs):
        return FakeBrowser(self.html_pages, self.events)


class FakePlaywrightContextManager:
    def __init__(self, html_pages, events):
        self.playwright = type(
            "FakePlaywright",
            (),
            {"chromium": FakeChromium(html_pages, events)},
        )()

    async def __aenter__(self):
        return self.playwright

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class CrawlerStreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_page_handler_finishes_before_next_url_is_crawled(self):
        events = []
        html_pages = [
            "<html><head><title>One</title></head><body><main>"
            + ("First page content. " * 5)
            + '<a href="/two">Next</a></main></body></html>',
            "<html><head><title>Two</title></head><body><main>"
            + ("Second page content. " * 5)
            + "</main></body></html>",
        ]
        crawler = WebCrawler("https://example.com")

        def store_page(page):
            events.append(("store", page.url))

        with patch(
            "src.services.crawler.async_playwright",
            return_value=FakePlaywrightContextManager(html_pages, events),
        ):
            pages = await crawler._crawl_impl(
                on_page=store_page,
                collect_pages=False,
            )

        self.assertEqual(
            events,
            [
                ("crawl", "https://example.com"),
                ("store", "https://example.com"),
                ("crawl", "https://example.com/two"),
                ("store", "https://example.com/two"),
            ],
        )
        self.assertEqual(pages, [])
        self.assertEqual(crawler.pages_crawled, 2)


if __name__ == "__main__":
    unittest.main()
