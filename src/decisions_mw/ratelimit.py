import time

from .errors import RateLimitedError

MAX_TRACKED_KEYS = 10_000


class RateLimiter:
    """In-process token bucket per API key: `per_minute` requests, refilled continuously."""

    def __init__(self, per_minute: int, max_keys: int = MAX_TRACKED_KEYS):
        self.capacity = float(per_minute)
        self.rate = per_minute / 60.0
        self.max_keys = max_keys
        self._buckets: dict[str, tuple[float, float]] = {}

    def _prune(self, now: float) -> None:
        # Forwarded keys aren't validated before this check, so callers control how many keys
        # we see. A bucket that has refilled completely holds no state and can be dropped.
        self._buckets = {
            k: (tokens, last)
            for k, (tokens, last) in self._buckets.items()
            if tokens + (now - last) * self.rate < self.capacity
        }

    def check(self, key: str) -> None:
        if not self.capacity:
            return
        now = time.monotonic()
        if key not in self._buckets and len(self._buckets) >= self.max_keys:
            self._prune(now)
        tokens, last = self._buckets.get(key, (self.capacity, now))
        tokens = min(self.capacity, tokens + (now - last) * self.rate)
        if tokens < 1:
            self._buckets[key] = (tokens, now)
            raise RateLimitedError("rate limit exceeded", retry_after=(1 - tokens) / self.rate)
        self._buckets[key] = (tokens - 1, now)
