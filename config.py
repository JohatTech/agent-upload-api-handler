"""
Single source of truth for every setting the blob watcher and report generator needs.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent / ".env")
load_dotenv(Path(__file__).parent.parent / ".env")

# Detect Vercel / serverless environment (including Azure Functions)
IS_VERCEL: bool = (
    os.getenv("VERCEL") == "1" 
    or os.getenv("AWS_LAMBDA_FUNCTION_NAME") is not None 
    or os.getenv("FUNCTIONS_WORKER_RUNTIME") is not None
)

# Helper
def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {name}")
    return value

# ── Watched Folder ────────────────────────────────────────────────────────────
WATCH_FOLDER_PATH: str | None = os.getenv("WATCH_FOLDER_PATH")

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

# ── Embedding Provider & Model ────────────────────────────────────────────────
EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-similarity-davinci-001")

AZURE_OPENAI_API_KEY: str | None = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT: str | None = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str | None = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

# ── LLM Model Registry ──────────────────────────────────────────────────────
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for a single LLM model."""
    name: str
    endpoint: str
    api_key: str
    deployment: str
    api_version: str
    provider: str

class ModelRegistry:
    def __init__(self):
        self._models: dict[str, ModelConfig] = {}
        self._discover()
    
    def _discover(self):
        prefix = "MODEL_"
        suffix = "_ENDPOINT"
        for key, value in os.environ.items():
            if key.startswith(prefix) and key.endswith(suffix):
                clean_value = value.strip('"').strip("'")
                name = key[len(prefix):-len(suffix)]
                self._models[name] = ModelConfig(
                    name=name,
                    endpoint=clean_value,
                    api_key=os.getenv(f"MODEL_{name}_API_KEY", "").strip('"').strip("'"),
                    deployment=os.getenv(f"MODEL_{name}_DEPLOYMENT", "").strip('"').strip("'"),
                    api_version=os.getenv(f"MODEL_{name}_API_VERSION", "2024-12-01-preview").strip('"').strip("'"),
                    provider=os.getenv(f"MODEL_{name}_PROVIDER", "azure_openai").lower().strip('"').strip("'"),
                )
    
    def get(self, name: str) -> ModelConfig:
        key = name.upper()
        if key not in self._models:
            available = ", ".join(self._models.keys()) or "(none registered)"
            raise KeyError(f"Model '{name}' not found. Available: {available}")
        return self._models[key]

MODEL_REGISTRY = ModelRegistry()
DEFAULT_CHAT_MODEL: str = os.getenv("DEFAULT_CHAT_MODEL", "CLAUDE")

# Backward Compatibility mapping
try:
    _def_cfg = MODEL_REGISTRY.get(DEFAULT_CHAT_MODEL)
    AZURE_OPENAI_CHAT_DEPLOYMENT = _def_cfg.deployment
    AZURE_OPENAI_API_KEY_CHAT = _def_cfg.api_key
    AZURE_OPENAI_ENDPOINT_CHAT = _def_cfg.endpoint
except Exception:
    AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4")
    AZURE_OPENAI_API_KEY_CHAT = os.getenv("AZURE_OPENAI_API_KEY_CHAT")
    AZURE_OPENAI_ENDPOINT_CHAT = os.getenv("AZURE_OPENAI_ENDPOINT_CHAT")

# ── Qdrant ────────────────────────────────────────────────────────────────────
QDRANT_MODE: str = os.getenv("QDRANT_MODE", "local").lower()
QDRANT_LOCAL_PATH: str = os.getenv("QDRANT_LOCAL_PATH", "./qdrant_data")
QDRANT_URL: str | None = os.getenv("QDRANT_URL")
QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY")

# ── Supabase ──────────────────────────────────────────────────────────────────
SUPABASE_URL: str | None = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY: str | None = os.getenv("SUPABASE_SERVICE_KEY")

# ── Vectorstore Targets ──────────────────────────────────────────────────────
VECTORSTORE_TARGETS: list[str] = [
    t.strip().lower()
    for t in os.getenv("VECTORSTORE_TARGETS", "supabase").split(",")
    if t.strip()
]

# ── Processing ────────────────────────────────────────────────────────────────
MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "4"))
STABILITY_WAIT_SECONDS: int = int(os.getenv("STABILITY_WAIT_SECONDS", "10"))
USE_POLLING_WATCHER: bool = os.getenv("USE_POLLING_WATCHER", "False").lower() in ("true", "1", "yes")
POLLING_INTERVAL_SECONDS: float = float(os.getenv("POLLING_INTERVAL_SECONDS", "10.0"))

# ── Azure Blob Storage Watcher ───────────────────────────────────────────────
USE_BLOB_WATCHER: bool = os.getenv("USE_BLOB_WATCHER", "False").lower() in ("true", "1", "yes")
AZURE_STORAGE_CONNECTION_STRING: str | None = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_CONTAINER_NAME: str | None = os.getenv("AZURE_STORAGE_CONTAINER_NAME")
if IS_VERCEL:
    BLOB_EVENT_HISTORY_LOG_PATH: str = os.getenv("BLOB_EVENT_HISTORY_LOG_PATH", "/tmp/blob_events.log")
else:
    BLOB_EVENT_HISTORY_LOG_PATH: str = os.getenv("BLOB_EVENT_HISTORY_LOG_PATH", "./blob_data/blob_events.log")

# ── Webhook Callback URLs ────────────────────────────────────────────────────
POWER_AUTOMATE_WEBHOOK_URL: str | None = os.getenv("POWER_AUTOMATE_WEBHOOK_URL")
FRONTEND_API_URL: str | None = os.getenv("FRONTEND_API_URL")

# URL of the separated chat-responder service (for project registration)
CHAT_RESPONDER_URL: str = os.getenv("CHAT_RESPONDER_URL", "http://localhost:8000")

# ── Supported File Extensions ────────────────────────────────────────────────
SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".xlsx": "xlsx",
    ".txt": "txt",
}

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s  │  %(name)-22s  │  %(levelname)-7s  │  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

DATABASE_URL: str | None = os.getenv("DATABASE_URL")
