"""Central configuration: paths, database URL and the LLM provider (the one place to change it)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")  # local development only; in Azure these are app settings

# --- Data and artefacts ----------------------------------------------------------------------
# SQLite for now; the SQL tool and loaders only need this URL (the tool's read-only
# connection is SQLite-specific, see src/tools/sql_guard.py).
DB_URL = os.getenv("DATABASE_URL", f"sqlite:///{ROOT / 'data' / 'sales.db'}")
DOCS_DIR = ROOT / "docs"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
CHROMA_DIR = ROOT / "chroma_db"
TRUE_DEMAND_PATH = ROOT / "data" / "true_demand.csv.gz"  # simulator truth, for the oracle only
# MLflow: local SQLite backend (no server needed). The plain file store is deprecated in MLflow 3.
MLFLOW_TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"


# --- LLM provider ------------------------------------------------------------------------------
@dataclass(frozen=True)
class LLMSettings:
    """Which LLM and embedding backend to use, resolved from environment variables."""

    provider: str  # "azure", "openai" or "none" (no key configured)
    api_key: str | None
    azure_endpoint: str | None
    azure_api_version: str
    chat_model: str  # OpenAI model name, or the Azure *deployment* name
    embedding_provider: str  # "azure", "openai" or "local" (offline Chroma model)
    embedding_model: str  # OpenAI model name, or Azure deployment name; unused for "local"


def load_llm_settings() -> LLMSettings:
    """Azure OpenAI if its env vars are set, otherwise OpenAI, otherwise 'none'.

    Set LLM_PROVIDER=openai|azure to force one, and EMBEDDING_PROVIDER=local to force the
    offline embedding model. Read at call time so tests can change the environment.
    """
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    openai_key = os.getenv("OPENAI_API_KEY")

    provider = os.getenv("LLM_PROVIDER", "").lower()
    if not provider:
        if azure_key and azure_endpoint:
            provider = "azure"
        elif openai_key:
            provider = "openai"
        else:
            provider = "none"

    azure_embedding = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "").lower()
    if not embedding_provider:
        if provider == "azure" and azure_embedding:
            embedding_provider = "azure"
        elif provider == "openai":
            embedding_provider = "openai"
        else:
            embedding_provider = "local"

    if provider == "azure":
        chat_model = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini")
        embedding_model = azure_embedding or ""
    else:
        chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    return LLMSettings(
        provider=provider,
        api_key=azure_key if provider == "azure" else openai_key,
        azure_endpoint=azure_endpoint,
        azure_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        chat_model=chat_model,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
    )
