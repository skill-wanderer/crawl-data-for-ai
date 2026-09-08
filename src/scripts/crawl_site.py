"""
Standalone script to crawl a website and store data in Qdrant.
Usage: python -m src.scripts.crawl_site https://skill-wanderer.com
"""

import asyncio
import logging
import sys
import uuid

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

    # Store in Qdrant. The previous generation is only retired once the new
    # one has landed, so a failed upload leaves the existing data intact.
    store = VectorStore()
    crawl_id = str(uuid.uuid4())
    vectors = store.store_pages(pages, crawl_id=crawl_id)
    logger.info(f"Stored {vectors} vectors in Qdrant.")

    deleted = store.delete_by_domain(crawler.root_domain, exclude_crawl_id=crawl_id)
    if deleted > 0:
        logger.info(f"Retired {deleted} vectors from the previous crawl of {crawler.root_domain}")

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
