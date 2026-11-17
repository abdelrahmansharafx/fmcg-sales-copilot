"""Structured (JSON) logging: one line per event, easy to search in Azure Log Analytics."""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import UTC, datetime
from typing import TextIO


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON. Extra fields go in `extra={"fields": {...}}`."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            **getattr(record, "fields", {}),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(stream: TextIO | None = None, level: int = logging.INFO) -> None:
    """Send all logs to `stream` as JSON. Idempotent. MCP uses stderr: stdout is its protocol."""
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)


_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def clean_request_id(value: str | None) -> str | None:
    """Accept a caller-supplied X-Request-ID only if it is short and harmless (no log injection)."""
    return value if value and _REQUEST_ID.match(value) else None
