"""Retry helpers for transient I/O failures.

Updates:
  v0.1.0 - 2025-12-12 - Add async/sync exponential backoff retry helpers.
"""

from __future__ import annotations

import asyncio
import random
import time
import urllib.error
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

_RETRYABLE_HTTP_STATUS_CODES = {408, 429}


def is_retryable_http_status(status_code: int) -> bool:
    """Return ``True`` when *status_code* suggests a transient failure."""
    return status_code in _RETRYABLE_HTTP_STATUS_CODES or 500 <= status_code < 600


def is_retryable_httpx_error(exc: Exception) -> bool:
    """Return ``True`` when *exc* represents a transient httpx error."""
    if isinstance(exc, httpx.HTTPStatusError):
        return is_retryable_http_status(exc.response.status_code)
    return isinstance(exc, (httpx.TimeoutException, httpx.TransportError))


def is_retryable_url_error(exc: Exception) -> bool:
    """Return ``True`` when *exc* represents a transient urllib error."""
    if isinstance(exc, urllib.error.HTTPError):
        return is_retryable_http_status(exc.code)
    return isinstance(exc, urllib.error.URLError)


def _compute_delay_seconds(
    attempt: int,
    *,
    base: float,
    maximum: float,
    jitter: float,
) -> float:
    delay = min(maximum, base * (2 ** (attempt - 1)))
    if jitter <= 0:
        return delay
    return delay + (delay * jitter * random.random())


async def async_retry[T](
    operation: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 3,
    base_delay_seconds: float = 0.5,
    max_delay_seconds: float = 4.0,
    jitter_fraction: float = 0.1,
    should_retry: Callable[[Exception], bool],
) -> T:
    """Execute *operation* with exponential-backoff retries.

    Args:
      operation: Zero-argument coroutine factory to execute.
      max_attempts: Total attempts including the first call.
      base_delay_seconds: Base backoff delay for the second attempt.
      max_delay_seconds: Cap for exponential backoff.
      jitter_fraction: Add random jitter as a fraction of the computed delay.
      should_retry: Predicate that decides whether an exception is retryable.

    Returns:
      The value returned by *operation* on success.

    Raises:
      Exception: Re-raises the last exception when retries are exhausted or non-retryable.
    """
    attempts = max(1, int(max_attempts))
    for attempt in range(1, attempts + 1):
        try:
            return await operation()
        except Exception as exc:
            if attempt >= attempts or not should_retry(exc):
                raise
            if base_delay_seconds <= 0:
                continue
            delay = _compute_delay_seconds(
                attempt,
                base=base_delay_seconds,
                maximum=max_delay_seconds,
                jitter=jitter_fraction,
            )
            await asyncio.sleep(delay)
    raise RuntimeError("async_retry exhausted retries")  # pragma: no cover


def retry[T](
    operation: Callable[[], T],
    *,
    max_attempts: int = 3,
    base_delay_seconds: float = 0.5,
    max_delay_seconds: float = 4.0,
    jitter_fraction: float = 0.1,
    should_retry: Callable[[Exception], bool],
) -> T:
    """Execute *operation* with exponential-backoff retries.

    Args:
      operation: Zero-argument callable to execute.
      max_attempts: Total attempts including the first call.
      base_delay_seconds: Base backoff delay for the second attempt.
      max_delay_seconds: Cap for exponential backoff.
      jitter_fraction: Add random jitter as a fraction of the computed delay.
      should_retry: Predicate that decides whether an exception is retryable.

    Returns:
      The value returned by *operation* on success.

    Raises:
      Exception: Re-raises the last exception when retries are exhausted or non-retryable.
    """
    attempts = max(1, int(max_attempts))
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as exc:
            if attempt >= attempts or not should_retry(exc):
                raise
            if base_delay_seconds <= 0:
                continue
            delay = _compute_delay_seconds(
                attempt,
                base=base_delay_seconds,
                maximum=max_delay_seconds,
                jitter=jitter_fraction,
            )
            time.sleep(delay)
    raise RuntimeError("retry exhausted retries")  # pragma: no cover
