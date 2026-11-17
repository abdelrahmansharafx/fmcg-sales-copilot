"""Retry helper for rate limits (HTTP 429). A busy API must not crash or fail an eval run."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

import openai

T = TypeVar("T")


def is_rate_limit(error: BaseException) -> bool:
    """True for an openai RateLimitError, directly or as the cause of a wrapped error."""
    return isinstance(error, openai.RateLimitError) or isinstance(
        error.__cause__, openai.RateLimitError
    )


def with_retries(
    call: Callable[[], T],
    *,
    attempts: int = 4,
    delay: float = 20.0,
    sleep: Callable[[float], None] | None = None,
    on_retry: Callable[[int], None] | None = None,
) -> T:
    """Run `call`; on a rate limit wait delay, 2*delay, ... seconds and try again."""
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as error:
            if not is_rate_limit(error) or attempt == attempts:
                raise
            if on_retry:
                on_retry(attempt)
            (sleep or time.sleep)(delay * attempt)  # looked up now, so tests can patch it
    raise AssertionError("unreachable")
