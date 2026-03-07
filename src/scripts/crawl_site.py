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

    # Crawl
    crawler = WebCrawler(url, include_subdomains=False)
    pages = await crawler.crawl()
    logger.info(f"Crawled {len(pages)} pages.")

    if not pages:
        logger.warning("No pages found. Exiting.")
        return

    # Store in Qdrant
    store = VectorStore()
    # Delete existing data for this domain first
    deleted = store.delete_by_domain(crawler.root_domain)
    if deleted > 0:
        logger.info(f"Deleted {deleted} existing vectors for {crawler.root_domain}")

    vectors = store.store_pages(pages)
    logger.info(f"Stored {vectors} vectors in Qdrant.")

    # Print summary
    print(f"\n{'='*50}")
    print(f"Crawl Summary for {crawler.root_domain}")
    print(f"{'='*50}")
    print(f"Pages crawled: {len(pages)}")
    print(f"Vectors stored: {vectors}")
    print(f"{'='*50}")
    for page in pages:
        print(f"  - {page.title}: {page.url}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.scripts.crawl_site <url>")
        print("Example: python -m src.scripts.crawl_site https://skill-wanderer.com")
        sys.exit(1)

    asyncio.run(main(sys.argv[1]))
