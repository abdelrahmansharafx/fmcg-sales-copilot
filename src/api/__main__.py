"""Run the API: `uv run python -m src.api` (host/port from API_HOST / PORT)."""

import os

import uvicorn

from src.api.logging_setup import configure_logging

if __name__ == "__main__":
    configure_logging()
    uvicorn.run(
        "src.api.main:app",
        host=os.getenv("API_HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        access_log=False,  # our middleware writes one structured line per request instead
        log_config=None,  # keep uvicorn's own messages in the same JSON format
    )
