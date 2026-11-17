"""The chat page served at "/" by the same FastAPI app (no extra service).

The page is one HTML file with inline CSS and JS and no external resources. It is public (it
contains no data); every question goes to POST /ask with the user's API key in the X-API-Key header.

Security, in layers:
  * Content-Security-Policy: scripts and styles must match the SHA-256 hashes of the inline blocks
    (computed here from the file, so no 'unsafe-inline'), and `connect-src 'self'` means the page
    can only talk to this server. The API key therefore cannot be sent anywhere else, even if a
    bug or injected markup tried to.
  * The page renders answers with textContent only, never innerHTML (a test enforces it).
  * nosniff, no referrer, and no framing.
"""

from __future__ import annotations

import base64
import hashlib
import re
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

PAGE = Path(__file__).parent / "static" / "index.html"
router = APIRouter()


def _hash(text: str) -> str:
    """CSP source for an inline block: 'sha256-<base64 of the SHA-256 of its exact text>'."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return f"'sha256-{base64.b64encode(digest).decode()}'"


def build_csp(html: str) -> str:
    """The policy for this exact page: one inline script, one inline style, nothing external."""
    scripts = re.findall(r"<script>(.*?)</script>", html, flags=re.DOTALL)
    styles = re.findall(r"<style>(.*?)</style>", html, flags=re.DOTALL)
    if len(scripts) != 1 or len(styles) != 1:
        raise ValueError("index.html must contain exactly one <script> and one <style> block")
    return "; ".join(
        [
            "default-src 'none'",
            f"script-src {_hash(scripts[0])}",
            f"style-src {_hash(styles[0])}",
            "connect-src 'self'",
            "img-src data:",  # the inline SVG favicon
            "base-uri 'none'",
            "form-action 'none'",
            "frame-ancestors 'none'",
        ]
    )


@lru_cache
def load_page() -> tuple[str, dict[str, str]]:
    """The page text and the response headers, built once per process."""
    html = PAGE.read_text(encoding="utf-8")
    headers = {
        "Content-Security-Policy": build_csp(html),
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "no-referrer",
        "X-Frame-Options": "DENY",
        "Cache-Control": "no-cache",
    }
    return html, headers


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> HTMLResponse:
    html, headers = load_page()
    return HTMLResponse(html, headers=headers)
