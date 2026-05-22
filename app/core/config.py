"""
app/core/config.py
------------------
Centralized application configuration using pydantic-settings.

All environment variables are loaded from the .env file and validated here.
This prevents scattered os.getenv() calls throughout the codebase and ensures
type safety at startup rather than at runtime.
"""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables / .env file.

    Architecture note: Using BaseSettings over plain dataclasses ensures that
    all config is validated at startup (fail-fast) rather than failing silently
    when a misconfigured key is first accessed deep in a request cycle.
    """

    # --- Database ---
    DB_URL: str = "sqlite:///./greenpack.db"

    # --- Ollama LLM ---
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    MODEL_NAME: str = "llama3.2:1b"

    # --- Groq LLM ---
    GROQ_API_KEY: str
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # --- FAISS Vector Store ---
    FAISS_INDEX_PATH: str = "./faiss_index"

    # --- Document Storage ---
    DOCS_PATH: str = "./documents/epr_docs"

    # --- ERP Data ---
    ERP_CSV_PATH: str = "./mock_data/erp_feed.csv"

    # --- Logging ---
    LOG_LEVEL: str = "INFO"

    # --- RAG Chunking ---
    CHUNK_SIZE: int = 700
    CHUNK_OVERLAP: int = 130

    # --- RAG Retrieval ---
    RETRIEVAL_TOP_K: int = 4
    RETRIEVAL_THRESHOLD: float = 0.45  # Minimum cosine similarity for context inclusion

    # --- HTTP Client ---
    OLLAMA_TIMEOUT: float = 60.0
    OLLAMA_RETRIES: int = 2

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

    @property
    def faiss_index_dir(self) -> Path:
        """Resolved path to the FAISS index directory."""
        return Path(self.FAISS_INDEX_PATH)

    @property
    def docs_path_dir(self) -> Path:
        """Resolved path to the EPR documents directory."""
        return Path(self.DOCS_PATH)


# Singleton settings instance — import this everywhere
settings = Settings()
