"""
Playwright-based web crawler service.

Crawls all pages within a given domain (same-domain only, no subdomains by default).
Extracts text content, metadata, and stores the original domain for future subdomain support.
"""

import asyncio
import logging
import sys
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass, field

from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class CrawledPage:
    url: str
    domain: str  # original root domain (e.g. "skill-wanderer.com")
    title: str
    content: str  # cleaned text content
    meta_description: str = ""
    headings: list[str] = field(default_factory=list)


class WebCrawler:
    """Crawls a website using Playwright, extracting text content from each page."""

    def __init__(self, base_url: str, include_subdomains: bool = False):
        parsed = urlparse(base_url)
        self.base_url = f"{parsed.scheme}://{parsed.netloc}"
        self.domain = parsed.netloc.lower()
        # Strip www. prefix for domain matching
        self.root_domain = self.domain.removeprefix("www.")
        self.include_subdomains = include_subdomains
        self.visited: set[str] = set()
        self.pages: list[CrawledPage] = []

    def _is_same_domain(self, url: str) -> bool:
        """Check if a URL belongs to the same domain (or subdomain if enabled)."""
        parsed = urlparse(url)
        host = parsed.netloc.lower().removeprefix("www.")

        if self.include_subdomains:
            return host == self.root_domain or host.endswith(f".{self.root_domain}")
        return host == self.root_domain

    def _normalize_url(self, url: str) -> str:
        """Normalize URL by removing fragments and trailing slashes."""
        parsed = urlparse(url)
        # Remove fragment, keep scheme/netloc/path/query
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if parsed.query:
            normalized += f"?{parsed.query}"
        return normalized.rstrip("/")

    def _extract_content(self, html: str, url: str) -> CrawledPage | None:
        """Extract meaningful text content from HTML."""
        soup = BeautifulSoup(html, "lxml")

        # Remove script, style, nav, footer elements
        for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else ""

        meta_desc = ""
        meta_tag = soup.find("meta", attrs={"name": "description"})
        if meta_tag and meta_tag.get("content"):
            meta_desc = meta_tag["content"].strip()

        headings = []
        for h in soup.find_all(["h1", "h2", "h3"]):
            text = h.get_text(strip=True)
            if text:
                headings.append(text)

        # Get main content - prefer <main> or <article>, fall back to <body>
        main = soup.find("main") or soup.find("article") or soup.find("body")
        if not main:
            return None

        content = main.get_text(separator="\n", strip=True)
        # Collapse multiple blank lines
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        content = "\n".join(lines)

        if not content or len(content) < 50:
            return None

        return CrawledPage(
            url=url,
            domain=self.root_domain,
            title=title,
            content=content,
            meta_description=meta_desc,
            headings=headings,
        )

    def _extract_links(self, html: str, current_url: str) -> list[str]:
        """Extract all valid same-domain links from HTML."""
        soup = BeautifulSoup(html, "lxml")
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            # Skip mailto, tel, javascript links
            if href.startswith(("mailto:", "tel:", "javascript:", "#")):
                continue
            absolute = urljoin(current_url, href)
            # Skip non-http(s) and file links
            parsed = urlparse(absolute)
            if parsed.scheme not in ("http", "https"):
                continue
            # Skip common non-page extensions
            path_lower = parsed.path.lower()
            skip_extensions = (
                ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp",
                ".zip", ".tar", ".gz", ".mp4", ".mp3", ".wav",
                ".css", ".js", ".xml", ".json",
            )
            if any(path_lower.endswith(ext) for ext in skip_extensions):
                continue
            if self._is_same_domain(absolute):
                links.append(self._normalize_url(absolute))
        return links

    async def crawl(self) -> list[CrawledPage]:
        """Crawl the entire website and return extracted pages.

        On Windows, runs in a separate thread with a ProactorEventLoop
        because Playwright needs subprocess support that SelectorEventLoop lacks.
        """
        if sys.platform == "win32":
            return await asyncio.to_thread(self._crawl_in_proactor_loop)
        return await self._crawl_impl()

    def _crawl_in_proactor_loop(self) -> list[CrawledPage]:
        """Run the crawl in a new ProactorEventLoop (Windows only)."""
        loop = asyncio.ProactorEventLoop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._crawl_impl())
        finally:
            loop.close()

    async def _crawl_impl(self) -> list[CrawledPage]:
        """Core crawl logic using Playwright."""
        logger.info(f"Starting crawl of {self.base_url} (domain: {self.root_domain})")

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/131.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1280, "height": 720},
            )

            queue = [self._normalize_url(self.base_url)]
            self.visited.clear()
            self.pages.clear()

            while queue:
                url = queue.pop(0)
                if url in self.visited:
                    continue
                self.visited.add(url)

                try:
                    page = await context.new_page()
                    logger.info(f"Crawling: {url}")
                    response = await page.goto(url, wait_until="networkidle", timeout=30000)

                    if not response or response.status >= 400:
                        logger.warning(f"Skipping {url} - status {response.status if response else 'no response'}")
                        await page.close()
                        continue

                    content_type = response.headers.get("content-type", "")
                    if "text/html" not in content_type:
                        await page.close()
                        continue

                    # Wait a bit for JS rendering
                    await page.wait_for_timeout(1000)
                    html = await page.content()
                    await page.close()

                    # Extract content
                    crawled = self._extract_content(html, url)
                    if crawled:
                        self.pages.append(crawled)
                        logger.info(f"Extracted: {crawled.title} ({len(crawled.content)} chars)")

                    # Extract and queue new links
                    new_links = self._extract_links(html, url)
                    for link in new_links:
                        if link not in self.visited:
                            queue.append(link)

                except Exception as e:
                    logger.error(f"Error crawling {url}: {e}")
                    continue

            await browser.close()

        logger.info(f"Crawl complete. {len(self.pages)} pages extracted from {self.root_domain}")
        return self.pages
