# Qdrant Vector Schema — Chatbot Consumer Guide

This document describes the data stored in Qdrant by the **crawl-data-for-ai** pipeline and how a chatbot microservice should query and consume it.

---

## Overview

The pipeline crawls websites, splits page content into overlapping text chunks, generates embeddings via **Google Gemini**, and stores them in a Qdrant collection. A chatbot can perform semantic similarity search against this collection to retrieve relevant context for answering user questions.

## Qdrant Connection

| Parameter       | Default            | Env Variable       |
| --------------- | ------------------ | ------------------ |
| Host            | `localhost`        | `QDRANT_HOST`      |
| REST API Port   | `6333`             | `QDRANT_PORT`      |
| gRPC Port       | `6334`             | —                  |
| Collection Name | `website_pages`    | `QDRANT_COLLECTION`|

The collection is also available via Docker Compose (service `qdrant`). Inside a Docker network, use `qdrant` as the hostname instead of `localhost`.

---

## Collection Configuration

| Setting    | Value                |
| ---------- | -------------------- |
| Vector Size | **3072**            |
| Distance    | **Cosine**          |
| Embedding Model | `gemini-embedding-001` (Google Gemini) |

---

## Point Structure

Each point in the collection represents a **single text chunk** from a crawled page.

### ID

- Type: `UUID v4` (string)
- Randomly generated per chunk.

### Vector

- **3072-dimensional** float array produced by `gemini-embedding-001`.
- The input text sent to the embedding model has the format:

```
Title: <page title>
Description: <meta description>   ← included only when available

<page body text>
```

This means the vector already encodes title and description context.

### Payload Fields

| Field              | Type         | Description |
| ------------------ | ------------ | ----------- |
| `text`             | `string`     | The raw text chunk that was embedded. This is the content you should display or feed to the LLM as context. |
| `url`              | `string`     | Full URL of the source page (e.g. `https://example.com/about`). |
| `domain`           | `string`     | Root domain without `www.` prefix (e.g. `example.com`). Useful for filtering results to a specific website. |
| `title`            | `string`     | HTML `<title>` of the source page. |
| `meta_description` | `string`     | Content of the `<meta name="description">` tag. Empty string if absent. |
| `headings`         | `string[]`   | List of `h1`, `h2`, `h3` heading texts extracted from the page. |
| `chunk_index`      | `int`        | Zero-based index of this chunk within the page (e.g. `0`, `1`, `2`…). |
| `total_chunks`     | `int`        | Total number of chunks the page was split into. |

### Example Point (JSON)

```json
{
  "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "vector": [0.0123, -0.0456, 0.0789, "... (3072 floats)"],
  "payload": {
    "text": "Title: About Us\nDescription: Learn more about our company.\n\nWe are a technology company founded in...",
    "url": "https://example.com/about",
    "domain": "example.com",
    "title": "About Us",
    "meta_description": "Learn more about our company.",
    "headings": ["About Us", "Our Mission", "Our Team"],
    "chunk_index": 0,
    "total_chunks": 3
  }
}
```

---

## Chunking Strategy

| Parameter     | Value  |
| ------------- | ------ |
| Chunk size    | **2000 characters** |
| Overlap       | **200 characters**  |

- Pages shorter than 2000 characters produce a single chunk.
- Longer pages are split into overlapping windows so context is not lost at chunk boundaries.

---

## How to Query from a Chatbot Microservice

### 1. Generate the Query Embedding

Use the same embedding model (`gemini-embedding-001`) to embed the user's question:

```python
from google import genai

client = genai.Client(api_key="YOUR_GEMINI_API_KEY")
result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=["user question here"],
)
query_vector = result.embeddings[0].values  # list[float], length 3072
```

### 2. Search Qdrant

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

qdrant = QdrantClient(host="localhost", port=6333)

# Basic search — top 5 most relevant chunks
results = qdrant.query_points(
    collection_name="website_pages",
    query=query_vector,
    limit=5,
    with_payload=True,
)
```

### 3. Filter by Domain (optional)

If your chatbot serves multiple domains, restrict results to one:

```python
results = qdrant.query_points(
    collection_name="website_pages",
    query=query_vector,
    query_filter=Filter(
        must=[
            FieldCondition(key="domain", match=MatchValue(value="example.com"))
        ]
    ),
    limit=5,
    with_payload=True,
)
```

### 4. Build LLM Context

Assemble the retrieved chunks into a prompt for your LLM:

```python
context_parts = []
for point in results.points:
    p = point.payload
    context_parts.append(
        f"[Source: {p['title']} — {p['url']}]\n{p['text']}"
    )

context = "\n\n---\n\n".join(context_parts)

prompt = f"""Answer the user's question based on the following context.
If the context does not contain enough information, say so.

Context:
{context}

Question: {user_question}
"""
```

### 5. Score Threshold (recommended)

Filter out low-relevance results by setting a `score_threshold`:

```python
results = qdrant.query_points(
    collection_name="website_pages",
    query=query_vector,
    limit=5,
    score_threshold=0.5,  # adjust based on testing
    with_payload=True,
)
```

---

## Data Lifecycle

| Operation       | Behavior |
| --------------- | -------- |
| Re-crawl domain | All existing vectors for that domain are **deleted first**, then new vectors are inserted. Data is always fresh. |
| Delete domain   | Vectors are filtered by the `domain` payload field and removed. |
| Multiple domains | The same collection holds vectors for all domains. Use the `domain` filter to scope queries. |

---

## Quick Reference — Payload Fields for Chatbot Use

| Chatbot Need                     | Payload Field(s)              |
| -------------------------------- | ----------------------------- |
| Text to feed to LLM             | `text`                        |
| Citation / source link           | `url`, `title`               |
| Scope to a specific website      | `domain` (use as filter)     |
| Show page summary                | `title`, `meta_description`  |
| Reassemble full page             | `chunk_index`, `total_chunks`|
| Display section headings         | `headings`                   |
