"""Async chat-completions client for the model upstream, with retries and Jev-style error mapping.

Two request flavors:
- openrouter: OpenRouter's extensions (`reasoning` object, provider routing, usage accounting).
- openai: plain OpenAI Chat Completions, for OpenAI, Azure or a LiteLLM proxy. OpenAI rejects
  unknown fields, so none of OpenRouter's extensions are sent.
"""

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx

from .config import Settings
from .errors import (
    AuthenticationError,
    DecisionError,
    OverloadedError,
    RateLimitedError,
    UpstreamRequestError,
)

MAX_RETRY_DELAY_S = 10.0
LITELLM_COST_HEADER = "x-litellm-response-cost"


@dataclass(frozen=True)
class Completion:
    content: str | None
    finish_reason: str | None
    model: str
    provider: str | None
    input_tokens: int
    output_tokens: int
    cost: float | None
    cached_tokens: int = 0
    reasoning_tokens: int = 0


def _details(usage: dict[str, Any], key: str, field: str) -> int:
    details = usage.get(key)
    return int(details.get(field) or 0) if isinstance(details, dict) else 0


def _header_float(resp: httpx.Response, name: str) -> float | None:
    try:
        return float(resp.headers[name])
    except (KeyError, ValueError):
        return None


def _error_message(payload: Any) -> str:
    if isinstance(payload, dict):
        err = payload.get("error", payload)
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
    return "no details"


def error_for_status(
    status: int, message: str, retry_after: float | None = None, *, caller_key: bool = False
) -> DecisionError:
    if status == 429:
        return RateLimitedError(f"upstream rate limited: {message}", retry_after=retry_after)
    if status == 408 or status >= 500:
        return OverloadedError(f"upstream unavailable ({status}): {message}")
    if status in (401, 403):
        if caller_key:
            return AuthenticationError(f"API key rejected by upstream: {message}")
        return UpstreamRequestError("upstream rejected the configured API key")
    if status == 402:
        return UpstreamRequestError("upstream account has insufficient credits")
    return UpstreamRequestError(f"upstream rejected the request ({status}): {message}")


def _parse_completion(
    data: dict[str, Any], requested_model: str, cost_header: float | None
) -> Completion:
    choice = (data.get("choices") or [{}])[0]
    if choice.get("finish_reason") == "error" or "error" in choice:
        raise OverloadedError(f"upstream provider error: {_error_message(choice)}")
    content = (choice.get("message") or {}).get("content")
    if isinstance(content, list):  # content-part arrays from some providers
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    usage = data.get("usage") or {}
    cost = usage.get("cost")  # OpenRouter reports cost in the body, LiteLLM in a header
    return Completion(
        content=content,
        finish_reason=choice.get("finish_reason"),
        model=data.get("model") or requested_model,
        provider=data.get("provider"),
        input_tokens=int(usage.get("prompt_tokens") or 0),
        output_tokens=int(usage.get("completion_tokens") or 0),
        cost=float(cost) if cost is not None else cost_header,
        cached_tokens=_details(usage, "prompt_tokens_details", "cached_tokens"),
        reasoning_tokens=_details(usage, "completion_tokens_details", "reasoning_tokens"),
    )


class UpstreamClient:
    def __init__(self, settings: Settings, http: httpx.AsyncClient):
        self.settings = settings
        self.http = http

    def _headers(self, api_key: str | None) -> dict[str, str]:
        if api_key is None and self.settings.upstream_api_key is not None:
            api_key = self.settings.upstream_api_key.get_secret_value()
        if api_key is None:  # excluded by Settings validation; kept as a guard
            raise UpstreamRequestError("no upstream API key")
        headers = {"Authorization": f"Bearer {api_key}"}
        if self.settings.flavor == "openrouter":
            if self.settings.app_title:
                headers["X-Title"] = self.settings.app_title
            if self.settings.app_referer:
                headers["HTTP-Referer"] = self.settings.app_referer
        return headers

    def build_body(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: dict[str, Any],
        max_tokens: int,
        seed: int | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "response_format": response_format,
        }
        if seed is not None:
            body["seed"] = seed
        effort = self.settings.reasoning_effort
        if self.settings.flavor == "openrouter":
            body["max_tokens"] = max_tokens
            # Only route to providers that honour every parameter, i.e. the strict JSON schema.
            body["provider"] = {"require_parameters": True}
            body["usage"] = {"include": True}
            if effort:
                body["reasoning"] = {"effort": effort, "exclude": True}
        else:
            body["max_completion_tokens"] = max_tokens  # reasoning models reject max_tokens
            if effort:
                body["reasoning_effort"] = effort
        return body

    async def _post_once(self, body: dict[str, Any], api_key: str | None) -> Completion:
        url = f"{self.settings.upstream_base_url.rstrip('/')}/chat/completions"
        headers = self._headers(api_key)
        try:
            resp = await self.http.post(
                url, json=body, headers=headers, timeout=self.settings.timeout_s
            )
        except httpx.TimeoutException as e:
            raise OverloadedError(f"upstream timed out after {self.settings.timeout_s}s") from e
        except httpx.TransportError as e:
            raise OverloadedError(f"upstream connection failed: {e}") from e
        try:
            data = resp.json()
        except ValueError:
            data = None
        caller_key = api_key is not None
        if resp.status_code != 200:
            raise error_for_status(
                resp.status_code,
                _error_message(data),
                _header_float(resp, "retry-after"),
                caller_key=caller_key,
            )
        if not isinstance(data, dict):
            raise OverloadedError("upstream returned a non-JSON response")
        if "error" in data and not data.get("choices"):
            err = data["error"] if isinstance(data["error"], dict) else {}
            code = err.get("code")
            status = code if isinstance(code, int) else 502
            raise error_for_status(status, _error_message(data), caller_key=caller_key)
        return _parse_completion(data, body["model"], _header_float(resp, LITELLM_COST_HEADER))

    async def complete(self, *, api_key: str | None = None, **kwargs: Any) -> Completion:
        """POST one completion, retrying rate limits and transient failures with backoff.

        `api_key` overrides the configured key (forwarded caller keys).
        """
        body = self.build_body(**kwargs)
        attempt = 0
        while True:
            try:
                return await self._post_once(body, api_key)
            except (RateLimitedError, OverloadedError) as e:
                if attempt >= self.settings.max_retries:
                    raise
                delay = max(e.retry_after or 0.0, self.settings.retry_backoff_s * 2**attempt)
                await asyncio.sleep(min(delay, MAX_RETRY_DELAY_S))
                attempt += 1
