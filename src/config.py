"""Central configuration: paths, database URL and the LLM provider (the one place to change it)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

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
    azure_base_url: str | None  # https://<resource>/openai/v1/ (Azure only)
    azure_api_version: str | None  # None = use the Azure "v1" API, which needs no version
    chat_model: str  # OpenAI model name, or the Azure *deployment* name
    embedding_provider: str  # "azure", "openai" or "local" (offline Chroma model)
    embedding_model: str  # OpenAI model name, or Azure deployment name; unused for "local"


# A real Azure api-version looks like 2024-10-21 or 2025-04-01-preview. Anything else
# (an empty value, a template placeholder) is ignored and the v1 endpoint is used.
_API_VERSION = re.compile(r"^\d{4}-\d{2}-\d{2}(-preview)?$")


def azure_base_url(endpoint: str) -> str:
    """Reduce any Azure endpoint the portal shows to the OpenAI "v1" base URL.

    The portal may give https://x.openai.azure.com/, https://x.services.ai.azure.com/api/projects/y
    or a full request URL ending in /openai/v1/responses. All of them mean the same resource, so
    we keep only scheme + host and append /openai/v1/.
    """
    endpoint = endpoint.strip()
    if "://" not in endpoint:
        endpoint = "https://" + endpoint
    parsed = urlparse(endpoint)
    return f"{parsed.scheme}://{parsed.netloc}/openai/v1/"


def load_llm_settings() -> LLMSettings:
    """Azure OpenAI is the default; OpenAI is the fallback; otherwise provider is 'none'.

    Azure is chosen when AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT are set. We call it
    through Azure's "v1" endpoint, so no API version is needed. Setting a real
    AZURE_OPENAI_API_VERSION switches to the older dated-version API instead.
    LLM_PROVIDER=openai|azure forces one provider, and EMBEDDING_PROVIDER=local forces the
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
        # Azure calls the model you deployed a "deployment"; its name is what we pass as `model`.
        chat_model = (
            os.getenv("AZURE_OPENAI_DEPLOYMENT")
            or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
            or "gpt-4o-mini"
        )
        embedding_model = azure_embedding or ""
    else:
        chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "").strip()
    return LLMSettings(
        provider=provider,
        api_key=azure_key if provider == "azure" else openai_key,
        azure_base_url=azure_base_url(azure_endpoint) if azure_endpoint else None,
        azure_api_version=api_version if _API_VERSION.match(api_version) else None,
        chat_model=chat_model,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
    )
