# crawl-data-for-ai

Web crawler that uses **Playwright** to scrape websites and stores the content in a **Qdrant** vector database using **Google Gemini** embeddings. Built for powering AI chatbots with website knowledge.

## Architecture

```
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│   Web UI (HTML)  │─────▶│  FastAPI Backend  │─────▶│   Qdrant Vector  │
│  Domain Manager  │      │  /api/crawl       │      │   Database       │
└──────────────────┘      │  /api/domains     │      └──────────────────┘
                          │  /api/status      │              ▲
                          └────────┬──────────┘              │
                                   │                         │
                          ┌────────▼──────────┐     ┌────────┴─────────┐
                          │ Playwright Crawler │     │ Gemini Embeddings│
                          │ (headless browser) │────▶│ (text → vectors) │
                          └───────────────────┘     └──────────────────┘
```

## Features

- **Playwright-based crawling** — renders JavaScript-heavy sites via headless Chromium
- **Domain-scoped storage** — each crawled page stores its root domain, ready for future subdomain support
- **Qdrant vector database** — stores chunked page content as embeddings for semantic search
- **Gemini embeddings** — uses Google's latest embedding model
- **Additive recrawls** — recrawls embed and store only content chunks that are not already present
- **Web UI** — add domains, monitor crawl progress, add new content, or delete domain data
- **Background processing** — crawls run asynchronously without blocking the API

## Prerequisites

- Python 3.11+
- Docker (for Qdrant)
- Google Gemini API key

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/skill-wanderer/crawl-data-for-ai.git
cd crawl-data-for-ai

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
playwright install chromium
```

### 2. Start Qdrant

```bash
docker compose up -d
```

Qdrant dashboard will be available at http://localhost:6333/dashboard

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and add your Gemini API key
```

### 4. Run the application

**Web UI + API server:**
```bash
python -m src
```
Open http://localhost:8000 in your browser.

**CLI crawl (standalone):**
```bash
python -m src.scripts.crawl_site https://skill-wanderer.com
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web UI |
| `POST` | `/api/crawl` | Start crawling a domain (body: `{"url": "..."}`) |
| `GET` | `/api/domains` | List all crawled domains with stats |
| `GET` | `/api/status/{domain}` | Get crawl job status |
| `DELETE` | `/api/domain/{domain}` | Delete all vectors for a domain |
| `GET` | `/api/stats` | Get Qdrant collection statistics |

## Project Structure

```
src/
├── __init__.py
├── __main__.py          # Entry point (uvicorn server)
├── app.py               # FastAPI application + routes
├── config.py            # Settings from .env
├── services/
│   ├── crawler.py       # Playwright web crawler
│   └── vector_store.py  # Qdrant + Gemini embeddings
└── scripts/
    └── crawl_site.py    # CLI crawl script
public/
    └── index.html       # Web UI
```

## How It Works

1. **Crawl** — Playwright visits every same-domain page starting from the given URL, extracting text content, titles, headings, and metadata.
2. **Chunk** — Long pages are split into overlapping chunks (2000 chars, 200 overlap) to stay within embedding model limits.
3. **Embed** — Only chunks not already stored for that source URL are sent to the Gemini embedding API, one chunk at a time.
4. **Store immediately** — Each new vector is upserted to Qdrant as soon as its embedding is returned, before processing the next chunk. Completed chunks remain stored if a later chunk fails.
5. **Recrawl** — The crawler compares each chunk by source URL and exact content hash, then immediately adds only chunks that are not already stored. Existing vectors are never changed or removed.
6. **Full recrawl** — Delete the domain first, then crawl it again. This is the only workflow that replaces all previously stored data.

Data is tagged with the **root domain** (e.g., `skill-wanderer.com`) so the chatbot microservice can query by domain, and future subdomain crawling will share the same domain tag.

An additive recrawl intentionally keeps historical chunks. If an existing page changes, its new chunks are added and its older chunks remain available. Delete the domain before crawling when the database should contain only the website's current version.

## Future: Chatbot Integration

The data stored in Qdrant is designed to be consumed by a separate chatbot microservice. That service will:
- Accept user questions
- Generate an embedding for the question using Gemini
- Search Qdrant for the most relevant chunks
- Use an LLM to generate an answer based on the retrieved context
