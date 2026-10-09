"""Errors mapped onto Jev's status codes (401, 422, 429, 529) plus 502 for bad upstream output."""


class DecisionError(Exception):
    status_code = 500
    error_type = "api_error"

    def __init__(self, message: str, *, retry_after: float | None = None):
        super().__init__(message)
        self.message = message
        self.retry_after = retry_after


class AuthenticationError(DecisionError):
    status_code = 401
    error_type = "authentication_error"


class InvalidRequestError(DecisionError):
    status_code = 422
    error_type = "invalid_request_error"


class RateLimitedError(DecisionError):
    status_code = 429
    error_type = "rate_limit_error"


class OverloadedError(DecisionError):
    """Upstream 5xx, timeout or connection failure after retries."""

    status_code = 529
    error_type = "overloaded_error"


class UpstreamRequestError(DecisionError):
    """Upstream rejected the request for a reason the caller can't fix (auth, credits, params)."""

    status_code = 502
    error_type = "upstream_error"


class BadUpstreamOutputError(DecisionError):
    """Upstream answered, but not with output matching the decision schema."""

    status_code = 502
    error_type = "upstream_output_error"
