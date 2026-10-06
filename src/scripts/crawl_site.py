"""
Standalone script to crawl a website and store data in Qdrant.
Usage: python -m src.scripts.crawl_site https://skill-wanderer.com
"""

import asyncio
import logging
import sys

from src.services.crawler import WebCrawler
from src.services.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main(url: str):
    logger.info(f"Starting crawl for: {url}")

    # Check and store each URL before crawling the next one. Full page bodies
    # and domain-wide Qdrant keys are not retained in memory.
    crawler = WebCrawler(url, include_subdomains=False)
    store = VectorStore()
    session = await asyncio.to_thread(
        store.start_additive_crawl, crawler.root_domain
    )

    def store_crawled_page(page):
        store.store_page(page, session)

    await crawler.crawl(on_page=store_crawled_page, collect_pages=False)
    logger.info(
        f"Crawled {crawler.pages_crawled} pages and added "
        f"{session.added} new vectors to Qdrant."
    )

    # Print summary
    print(f"\n{'='*50}")
    print(f"Crawl Summary for {crawler.root_domain}")
    print(f"{'='*50}")
    print(f"Pages crawled: {crawler.pages_crawled}")
    print(f"New vectors added: {session.added}")
    print(f"{'='*50}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.scripts.crawl_site <url>")
        print("Example: python -m src.scripts.crawl_site https://skill-wanderer.com")
        sys.exit(1)

    asyncio.run(main(sys.argv[1]))
