"""HTTP surface: the same decision engine in two wire formats.

    POST /v1/systemone   Jev (TypeSafe) format
    POST /v1/decisions   OpenAI Decisions API format (works with the OpenAI SDK)

Run with: uvicorn --factory decisions_mw.main:create_app

Request headers (optional extensions; the body stays pure Jev):
    X-Samples: k          average k upstream samples (1..DMW_MAX_SAMPLES)
    X-Calibration: off    return raw, uncalibrated probabilities (for fitting calibration)
"""

import hmac
import json
import logging
import math
import time
from contextlib import asynccontextmanager
from typing import Annotated, Any

import httpx
from fastapi import Depends, FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from . import __version__
from .calibration import CalibrationStore
from .config import Settings, get_settings
from .errors import AuthenticationError, DecisionError, InvalidRequestError
from .openai_format import DecisionsRequest, DecisionsResponse, from_jev, to_jev
from .ratelimit import RateLimiter
from .schemas import DecisionRequest, DecisionResponse
from .service import DecisionMeta, DecisionService, new_request_id
from .upstream import UpstreamClient

logger = logging.getLogger("decisions_mw")

DECISIONS_PATH = "/v1/decisions"


def _configure_logging(level: str) -> None:
    """One JSON line per request on stderr, independent of the server's logging config."""
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level.upper())


def _elapsed_ms(request: Request) -> int:
    return round((time.perf_counter() - request.state.start) * 1000)


def _error_response(
    request: Request,
    status: int,
    error_type: str,
    message: str,
    *,
    exception: BaseException | None = None,
    **extra: Any,
) -> JSONResponse:
    if status == 422 and request.url.path == DECISIONS_PATH:
        status = 400  # OpenAI's API (and its SDK) treat invalid requests as 400 Bad Request
    event = {
        "event": "decision_error",
        "request_id": request.state.request_id,
        "status": status,
        "type": error_type,
        "message": message,
        "latency_ms": _elapsed_ms(request),
    }
    if exception is not None:
        # The class name only: an exception's message may quote request data.
        event["exception"] = type(exception).__name__
    logger.log(logging.ERROR if status == 500 else logging.WARNING, json.dumps(event))
    return JSONResponse(
        status_code=status, content={"error": {"type": error_type, "message": message, **extra}}
    )


def _report(
    response: Response, result: DecisionResponse, meta: DecisionMeta, endpoint: str
) -> None:
    """Response headers and the success log line, shared by both endpoints."""
    response.headers["X-Upstream-Model"] = meta.upstream_model
    if meta.upstream_providers:
        response.headers["X-Upstream-Provider"] = ",".join(meta.upstream_providers)
    response.headers["X-Upstream-Calls"] = str(meta.calls)
    response.headers["X-Samples"] = str(meta.samples)
    if meta.samples_failed:
        response.headers["X-Samples-Failed"] = str(meta.samples_failed)
    response.headers["X-Calibration"] = "on" if meta.calibrated else "off"
    response.headers["X-Latency-Ms"] = f"{meta.latency_ms:.0f}"
    if meta.cost_usd is not None:
        response.headers["X-Cost-USD"] = f"{meta.cost_usd:.8f}"
    logger.info(
        json.dumps(
            {
                "event": "decision",
                "endpoint": endpoint,
                "request_id": result.request_id,
                "model": meta.upstream_model,
                "providers": meta.upstream_providers,
                "questions": len(result.answers),
                "samples": meta.samples,
                "samples_failed": meta.samples_failed,
                "calls": meta.calls,
                "input_tokens": result.usage.input_tokens,
                "output_tokens": result.usage.output_tokens,
                "cost_usd": meta.cost_usd,
                "latency_ms": round(meta.latency_ms),
            }
        )
    )


def _bearer(request: Request) -> str | None:
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    token = token.strip()
    return token if scheme.lower() == "bearer" and token else None


def _key_matches(token: str, keys: frozenset[str]) -> bool:
    # Compare bytes: compare_digest rejects non-ASCII str, and headers may carry any latin-1.
    raw = token.encode()
    return any(hmac.compare_digest(raw, k.encode()) for k in keys)


def create_app(
    settings: Settings | None = None, http_client: httpx.AsyncClient | None = None
) -> FastAPI:
    settings = settings or get_settings()
    _configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        http = http_client or httpx.AsyncClient()
        client = UpstreamClient(settings, http)
        calibration = CalibrationStore.load(settings.calibration_path)
        app.state.service = DecisionService(settings, client, calibration)
        app.state.limiter = RateLimiter(settings.rate_limit_per_minute)
        try:
            yield
        finally:
            if http_client is None:
                await http.aclose()

    app = FastAPI(title="decisions-middleware", version=__version__, lifespan=lifespan)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request.state.request_id = new_request_id()
        request.state.start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-Id"] = request.state.request_id
        return response

    @app.exception_handler(DecisionError)
    async def decision_error(request: Request, exc: DecisionError) -> JSONResponse:
        response = _error_response(request, exc.status_code, exc.error_type, exc.message)
        if exc.retry_after:
            response.headers["Retry-After"] = str(math.ceil(exc.retry_after))
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            {"loc": list(e.get("loc", ())), "msg": e.get("msg"), "type": e.get("type")}
            for e in exc.errors()
        ]
        return _error_response(
            request, 422, "invalid_request_error", "request failed validation", details=details
        )

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        # Starlette runs this outside request_context, so the request id header is set here.
        # It re-raises afterwards, so the server still logs the traceback.
        response = _error_response(
            request, 500, "api_error", "internal server error", exception=exc
        )
        response.headers["X-Request-Id"] = request.state.request_id
        return response

    def authenticate(request: Request) -> str | None:
        """Check the caller; returns the key to forward upstream (forward-auth mode only)."""
        token = _bearer(request)
        if settings.forward_auth:
            if token is None:
                raise AuthenticationError("missing API key")
        elif settings.api_key_set:
            if token is None or not _key_matches(token, settings.api_key_set):
                raise AuthenticationError("missing or invalid API key")
        else:
            token = None  # anonymous mode
        request.app.state.limiter.check(token or "anonymous")
        return token if settings.forward_auth else None

    def options(request: Request) -> tuple[int | None, bool]:
        raw = request.headers.get("x-samples")
        samples = None
        if raw is not None:
            try:
                samples = int(raw)
            except ValueError:
                raise InvalidRequestError("X-Samples must be an integer") from None
        calibrate = request.headers.get("x-calibration", "on").strip().lower() != "off"
        return samples, calibrate

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok", "model": settings.luna_model}

    @app.post("/v1/systemone", response_model=DecisionResponse)
    async def systemone(
        body: DecisionRequest,
        request: Request,
        response: Response,
        forward_key: Annotated[str | None, Depends(authenticate)],
        opts: Annotated[tuple[int | None, bool], Depends(options)],
    ) -> DecisionResponse:
        samples, calibrate = opts
        result, meta = await request.app.state.service.decide(
            body,
            request_id=request.state.request_id,
            samples=samples,
            calibrate=calibrate,
            api_key=forward_key,
        )
        _report(response, result, meta, "systemone")
        return result

    @app.post(DECISIONS_PATH, response_model=DecisionsResponse, response_model_exclude_none=True)
    async def decisions(
        body: DecisionsRequest,
        request: Request,
        response: Response,
        forward_key: Annotated[str | None, Depends(authenticate)],
        opts: Annotated[tuple[int | None, bool], Depends(options)],
    ) -> DecisionsResponse:
        samples, calibrate = opts
        result, meta = await request.app.state.service.decide(
            to_jev(body),
            request_id=request.state.request_id,
            samples=samples,
            calibrate=calibrate,
            api_key=forward_key,
        )
        _report(response, result, meta, "decisions")
        return from_jev(body, result, meta)

    return app
