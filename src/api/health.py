"""Readiness checks behind GET /health."""

from __future__ import annotations

import sqlalchemy as sa

from src.config import MODELS_DIR, load_llm_settings
from src.forecast.predict import get_engine


def database_ok() -> bool:
    try:
        with get_engine().connect() as conn:
            return conn.execute(sa.text("SELECT COUNT(*) FROM outlets")).scalar() > 0
    except Exception:  # noqa: BLE001 - any failure simply means "not ready"
        return False


def model_ok() -> bool:
    return (MODELS_DIR / "lgbm_forecast.joblib").exists() and (
        MODELS_DIR / "model_meta.json"
    ).exists()


def readiness(version: str, policy_index: bool | None) -> dict:
    """Summarise what works. 'unavailable' = core data/model missing; 'degraded' = no LLM key.

    Without an LLM, /forecast and /recommend still work but /ask cannot.
    """
    database, model = database_ok(), model_ok()
    provider = load_llm_settings().provider
    if not (database and model):
        status = "unavailable"
    elif provider == "none":
        status = "degraded"
    else:
        status = "ok"
    return {
        "status": status,
        "version": version,
        "database": database,
        "model": model,
        "policy_index": policy_index,
        "llm_provider": provider,
    }
