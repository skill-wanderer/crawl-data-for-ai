"""
FastAPI application with domain management endpoints and UI.
"""

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, field_validator

from src.services.crawler import WebCrawler
from src.services.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Track crawl jobs and their asyncio tasks
crawl_jobs: dict[str, dict] = {}
crawl_tasks: dict[str, asyncio.Task] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize vector store
    app.state.vector_store = VectorStore()
    logger.info("Vector store initialized.")
    yield
    # Shutdown: cancel all running crawl tasks
    for domain, task in list(crawl_tasks.items()):
        if not task.done():
            logger.info(f"Cancelling crawl task for {domain}")
            task.cancel()
    # Wait for all tasks to finish cancellation
    if crawl_tasks:
        await asyncio.gather(*crawl_tasks.values(), return_exceptions=True)
        crawl_tasks.clear()
    logger.info("All crawl tasks cleaned up.")


app = FastAPI(title="Web Crawler for AI", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="public"), name="static")
templates = Jinja2Templates(directory="public")


# --- Pydantic models ---

class DomainRequest(BaseModel):
    url: str

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        v = v.strip()
        if not v.startswith(("http://", "https://")):
            v = f"https://{v}"
        parsed = urlparse(v)
        if not parsed.netloc:
            raise ValueError("Invalid URL")
        return v


class DomainResponse(BaseModel):
    domain: str
    status: str
    message: str


class JobStatus(BaseModel):
    domain: str
    status: str  # "crawling", "embedding", "completed", "failed"
    pages_crawled: int = 0
    vectors_stored: int = 0
    error: str | None = None


# --- Background crawl task ---

async def crawl_and_store(url: str, vector_store: VectorStore):
    """Background task: crawl a website and store results in Qdrant."""
    parsed = urlparse(url)
    domain = parsed.netloc.lower().removeprefix("www.")

    crawl_jobs[domain] = {
        "status": "crawling",
        "pages_crawled": 0,
        "vectors_stored": 0,
        "error": None,
    }

    crawler = WebCrawler(url, include_subdomains=False)
    try:
        # Step 1: Crawl
        pages = await crawler.crawl()
        crawl_jobs[domain]["pages_crawled"] = len(pages)
        crawl_jobs[domain]["status"] = "embedding"

        # Step 2: Store in Qdrant under a fresh crawl_id
        crawl_id = str(uuid.uuid4())
        vectors_count = await asyncio.to_thread(
            vector_store.store_pages, pages, crawl_id=crawl_id
        )
        crawl_jobs[domain]["vectors_stored"] = vectors_count

        # Step 3: Only now retire the previous generation for this domain.
        # If step 2 failed, the old vectors are still serving queries.
        deleted = await asyncio.to_thread(
            vector_store.delete_by_domain, domain, crawl_id
        )
        if deleted > 0:
            logger.info(f"Retired {deleted} vectors from the previous crawl of {domain}")

        crawl_jobs[domain]["status"] = "completed"
        logger.info(f"Crawl+store complete for {domain}: {len(pages)} pages, {vectors_count} vectors")

    except asyncio.CancelledError:
        crawler.stop()
        logger.info(f"Crawl cancelled for {domain} (server shutting down)")
        crawl_jobs[domain]["status"] = "failed"
        crawl_jobs[domain]["error"] = "Cancelled due to server shutdown"
    except Exception as e:
        logger.error(f"Crawl failed for {domain}: {e}")
        crawl_jobs[domain]["status"] = "failed"
        crawl_jobs[domain]["error"] = str(e)
    finally:
        crawl_tasks.pop(domain, None)


# --- API routes ---

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/crawl", response_model=DomainResponse)
async def start_crawl(req: DomainRequest):
    """Start crawling a domain.

    Existing vectors for the domain stay queryable and are replaced only after
    the new crawl has been stored successfully.
    """
    parsed = urlparse(req.url)
    domain = parsed.netloc.lower().removeprefix("www.")

    # Check if already crawling
    if domain in crawl_jobs and crawl_jobs[domain]["status"] in ("crawling", "embedding"):
        raise HTTPException(status_code=409, detail=f"Crawl already in progress for {domain}")

    # Existing data is left in place and only retired once the new crawl has
    # been stored successfully - see crawl_and_store().
    vector_store: VectorStore = app.state.vector_store

    # Start background crawl as a tracked asyncio task
    task = asyncio.create_task(crawl_and_store(req.url, vector_store))
    crawl_tasks[domain] = task

    return DomainResponse(
        domain=domain,
        status="started",
        message=f"Crawl started for {domain}. Existing vectors are kept until the new crawl completes.",
    )


@app.delete("/api/domain/{domain}", response_model=DomainResponse)
async def delete_domain(domain: str):
    """Delete all data for a domain from Qdrant."""
    vector_store: VectorStore = app.state.vector_store
    deleted = await asyncio.to_thread(vector_store.delete_by_domain, domain)

    # Clear job status
    clean_domain = domain.lower().removeprefix("www.").removeprefix("http://").removeprefix("https://").split("/")[0]
    crawl_jobs.pop(clean_domain, None)

    return DomainResponse(
        domain=clean_domain,
        status="deleted",
        message=f"Deleted {deleted} vectors for {clean_domain}.",
    )


@app.get("/api/status/{domain}", response_model=JobStatus)
async def get_crawl_status(domain: str):
    """Get the status of a crawl job."""
    clean_domain = domain.lower().removeprefix("www.")
    if clean_domain not in crawl_jobs:
        raise HTTPException(status_code=404, detail=f"No crawl job found for {clean_domain}")

    job = crawl_jobs[clean_domain]
    return JobStatus(
        domain=clean_domain,
        status=job["status"],
        pages_crawled=job["pages_crawled"],
        vectors_stored=job["vectors_stored"],
        error=job.get("error"),
    )


@app.get("/api/domains")
async def list_domains():
    """List all domains with their vector counts and crawl status."""
    vector_store: VectorStore = app.state.vector_store
    stats = await asyncio.to_thread(vector_store.get_domain_stats)

    domains = []
    for domain, count in stats.items():
        job = crawl_jobs.get(domain, {})
        domains.append({
            "domain": domain,
            "vectors_count": count,
            "crawl_status": job.get("status", "unknown"),
            "pages_crawled": job.get("pages_crawled", 0),
        })

    return {"domains": domains}


@app.get("/api/stats")
async def get_stats():
    """Get overall collection statistics."""
    vector_store: VectorStore = app.state.vector_store
    info = await asyncio.to_thread(vector_store.get_collection_info)
    return info
