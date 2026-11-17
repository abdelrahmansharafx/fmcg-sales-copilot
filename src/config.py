"""Central configuration: paths and database URL. (LLM provider settings are added in Phase 4.)"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# SQLite for now; set DATABASE_URL to a Postgres URL to swap it, no code changes needed.
DB_URL = os.getenv("DATABASE_URL", f"sqlite:///{ROOT / 'data' / 'sales.db'}")

MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
# MLflow: local SQLite backend (no server needed). The plain file store is deprecated in MLflow 3.
MLFLOW_TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
