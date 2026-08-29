"""Retry with exponential backoff for transient provider failures.

A single rate-limit or 5xx response should not turn into an empty tag list for
that image. Only transient failures are retried: a malformed request or a bad
API key fails the same way every time, so retrying it just wastes money and
delays the error the caller needs to see.

The Anthropic and OpenAI SDKs retry some of this internally; this layer sits
above them so the behaviour is explicit, logged, and identical for every
provider.
"""
from __future__ import annotations

import random
import time

from .logconf import get_logger

log = get_logger(__name__)

# Status codes worth trying again: rate limits, timeouts, conflicts, 5xx.
_TRANSIENT_STATUS = {408, 409, 429}

# Matched against the exception class name, so no provider SDK is imported here.
_TRANSIENT_NAMES = (
    "RateLimitError", "APIConnectionError", "APITimeoutError",
    "InternalServerError", "APIStatusError", "ConnectionError",
    "Timeout", "TimeoutError", "ServiceUnavailable",
)


def is_transient(exc: BaseException) -> bool:
    """True when retrying `exc` could plausibly succeed."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in _TRANSIENT_STATUS or status >= 500
    return type(exc).__name__ in _TRANSIENT_NAMES


def call_with_retry(fn, attempts: int = 3, base_delay: float = 1.0,
                    max_delay: float = 30.0, sleep=time.sleep):
    """Call `fn()`, retrying transient failures with exponential backoff.

    Raises the last exception once attempts are exhausted, or immediately for
    anything that is not transient.
    """
    last: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last = exc
            if not is_transient(exc) or attempt == attempts:
                raise
            delay = min(base_delay * 2 ** (attempt - 1), max_delay)
            delay += random.uniform(0, delay * 0.1)   # jitter
            log.warning("retrying_after_transient_error", extra={
                "attempt": attempt, "of": attempts,
                "delay_s": round(delay, 2),
                "error_type": type(exc).__name__, "error": str(exc)[:200]})
            sleep(delay)
    raise last  # unreachable, kept for type-checkers
