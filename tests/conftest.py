import inspect
import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from decisions_mw.config import Settings
from decisions_mw.main import create_app

# The request from Jev's API reference.
JEV_EXAMPLE = {
    "model": "jev-latest",
    "state": "Please refund the duplicate payment.",
    "questions": {
        "refund": {"type": "noul", "instructions": "Does the customer request a refund?"},
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this message?",
            "criteria": {"billing": "Payments and refunds", "technical": "Software errors"},
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this incident?",
            "criteria": ["Low urgency", "Needs attention today", "Requires immediate action"],
        },
    },
}


def completion(
    content: str | None,
    *,
    model: str = "openai/gpt-6-luna-20260922",
    provider: str | None = "OpenAI",
    finish_reason: str = "stop",
    cost: float | None = 0.00001,
) -> dict[str, Any]:
    usage: dict[str, Any] = {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}
    if cost is not None:
        usage["cost"] = cost
    body: dict[str, Any] = {
        "id": "gen-test",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage,
    }
    if provider is not None:  # OpenRouter names the provider; OpenAI-compatible upstreams don't
        body["provider"] = provider
    return body


def auto_answer(schema: dict[str, Any]) -> dict[str, Any]:
    """A valid answer for any decision schema: 70 for noul, 80 on the first option or level."""
    out: dict[str, Any] = {}
    for alias, prop in schema["properties"].items():
        if prop["type"] == "integer":
            out[alias] = 70
        elif "top" in prop["properties"]:
            first = prop["properties"]["top"]["items"]["properties"]["o"]["enum"][0]
            out[alias] = {"top": [{"o": first, "w": 80}]}
        else:
            slots = list(prop["properties"])
            rest = 20 // max(1, len(slots) - 1)
            out[alias] = {s: 80 if i == 0 else rest for i, s in enumerate(slots)}
    return out


Reply = dict[str, Any] | httpx.Response | Exception | Callable[[dict[str, Any]], Any]


class FakeUpstream:
    """Scripted upstream: replays queued replies, then answers validly from the request schema.

    A queued callable gets the request body and may be async (e.g. to simulate a slow upstream).
    """

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.headers: list[httpx.Headers] = []
        self.queue: list[Reply] = []

    def push(self, *replies: Reply) -> None:
        self.queue.extend(replies)

    def push_content(self, *contents: Any) -> None:
        for c in contents:
            self.queue.append(completion(c if isinstance(c, str) else json.dumps(c)))

    async def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        self.headers.append(request.headers)
        reply: Any
        if self.queue:
            reply = self.queue.pop(0)
        else:
            schema = body["response_format"]["json_schema"]["schema"]
            reply = completion(json.dumps(auto_answer(schema)))
        if callable(reply) and not isinstance(reply, dict):
            reply = reply(body)
            if inspect.isawaitable(reply):
                reply = await reply
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, httpx.Response):
            return reply
        return httpx.Response(200, json=reply)


@pytest.fixture
def fake() -> FakeUpstream:
    return FakeUpstream()


@pytest.fixture
def make_client(fake: FakeUpstream) -> Callable[..., Any]:
    @contextmanager
    def make(*, raise_server_exceptions: bool = True, **overrides: Any) -> Iterator[TestClient]:
        values: dict[str, Any] = {
            "upstream_api_key": "test-key",
            "allow_anonymous": True,
            "retry_backoff_s": 0.0,
        }
        values.update(overrides)
        settings = Settings(_env_file=None, **values)
        http = httpx.AsyncClient(transport=httpx.MockTransport(fake.handler))
        app = create_app(settings, http)
        with TestClient(app, raise_server_exceptions=raise_server_exceptions) as client:
            yield client

    return make


@pytest.fixture
def client(make_client: Callable[..., Any]) -> Iterator[TestClient]:
    with make_client() as c:
        yield c
