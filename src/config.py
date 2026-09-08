from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    gemini_api_key: str = ""

    # --- Qdrant connection ---
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_grpc_port: int = 6334
    # gRPC sends vectors as packed float32 instead of JSON decimal text
    # (~12KB vs ~65KB per 3072-dim point). Strongly recommended for remote servers.
    qdrant_prefer_grpc: bool = False
    qdrant_https: bool = False
    qdrant_api_key: str = ""
    # Seconds. The client library defaults to httpx's 5s, which is far too low
    # for multi-megabyte upserts over a long-haul link.
    qdrant_timeout: int = 120
    qdrant_collection: str = "website_pages"

    # --- Qdrant request sizing ---
    # Points per upsert request. Smaller batches keep each request well inside
    # network buffers and make a failure cheap to retry.
    qdrant_upsert_batch_size: int = 32
    # Points per scroll request when aggregating stats (payload-only, so this
    # can be much larger than the upsert batch).
    qdrant_scroll_batch_size: int = 1000

    # --- Qdrant retry policy (exponential backoff) ---
    qdrant_max_retries: int = 5
    qdrant_retry_base_delay: float = 1.0
    qdrant_retry_max_delay: float = 30.0

    # --- Embeddings ---
    embedding_model: str = "gemini-embedding-001"
    embedding_dimension: int = 3072
    # Texts per Gemini embed_content call. Independent of the Qdrant batch size;
    # billing is per input token, so this affects request count, not cost.
    embedding_batch_size: int = 100

    # --- Chunking ---
    # Changing these changes how stored documents are split; existing vectors
    # are not re-chunked retroactively.
    chunk_size: int = 2000
    chunk_overlap: int = 200

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
