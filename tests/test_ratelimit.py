import pytest

from decisions_mw.errors import RateLimitedError
from decisions_mw.ratelimit import RateLimiter


def test_bucket_per_key():
    limiter = RateLimiter(per_minute=2)
    limiter.check("a")
    limiter.check("a")
    with pytest.raises(RateLimitedError):
        limiter.check("a")
    limiter.check("b")  # other keys are unaffected


def test_disabled_when_zero():
    limiter = RateLimiter(per_minute=0)
    for _ in range(100):
        limiter.check("a")


def test_tracked_keys_stay_bounded():
    limiter = RateLimiter(per_minute=60, max_keys=10)
    limiter._buckets = {f"idle{i}": (60.0, 0.0) for i in range(10)}  # full buckets: droppable
    limiter.check("new")
    assert set(limiter._buckets) == {"new"}
