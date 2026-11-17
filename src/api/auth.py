"""API-key authentication: callers send the key in the `X-API-Key` header.

The valid key(s) come from the API_KEY environment variable (comma-separated, so a new key can
be added before the old one is removed). The check fails CLOSED: with no key configured every
protected endpoint answers 503 rather than letting anyone in.
"""

from __future__ import annotations

import hashlib
import os
import secrets

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

API_KEY_HEADER = "X-API-Key"
_header = APIKeyHeader(
    name=API_KEY_HEADER, auto_error=False, description="Shared secret set in the API_KEY env var."
)


def configured_keys() -> list[str]:
    """Valid keys from the environment, read on every request so rotation needs no code change."""
    return [key.strip() for key in os.getenv("API_KEY", "").split(",") if key.strip()]


def key_id(key: str) -> str:
    """A short, non-reversible label for a key, so logs can say WHICH key was used safely."""
    return hashlib.sha256(key.encode()).hexdigest()[:8]


def key_is_valid(provided: str, valid_keys: list[str]) -> bool:
    """Constant-time comparison against every key (no early exit that would leak timing)."""
    matched = False
    for valid in valid_keys:
        matched |= secrets.compare_digest(provided.encode(), valid.encode())
    return matched


def require_api_key(request: Request, provided: str | None = Security(_header)) -> str:
    """FastAPI dependency: reject the request unless it carries a valid API key."""
    keys = configured_keys()
    if not keys:
        raise HTTPException(503, "API authentication is not configured on the server.")
    if not provided or not key_is_valid(provided, keys):
        raise HTTPException(
            401, "Invalid or missing API key.", headers={"WWW-Authenticate": "ApiKey"}
        )
    request.state.log_fields["api_key_id"] = key_id(provided)
    return key_id(provided)
