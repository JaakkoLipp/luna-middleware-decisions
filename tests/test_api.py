import asyncio
import json
import logging
import math
from contextlib import contextmanager

import httpx
import pytest

from decisions_mw.calibration import Calibration, Platt

from .conftest import JEV_EXAMPLE, auto_answer, completion

JEV_LIKE_OUTPUT = {"q1": 99, "q2": {"o1": 100, "o2": 0}, "q3": {"l0": 0, "l1": 0, "l2": 100}}


def post(client, body=JEV_EXAMPLE, **headers):
    return client.post("/v1/systemone", json=body, headers=headers)


def error_type(resp) -> str:
    return resp.json()["error"]["type"]


@contextmanager
def captured_events():
    """JSON events logged by the app during the block."""
    events: list[dict] = []
    handler = logging.Handler()
    handler.emit = lambda record: events.append(json.loads(record.getMessage()))
    log = logging.getLogger("decisions_mw")
    log.addHandler(handler)
    try:
        yield events
    finally:
        log.removeHandler(handler)


def valid_reply(body):
    return completion(json.dumps(auto_answer(body["response_format"]["json_schema"]["schema"])))


def test_reproduces_jev_reference_response(client, fake):
    fake.push_content(JEV_LIKE_OUTPUT)
    resp = post(client)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert list(body) == [
        "answers",
        "id",
        "model",
        "provider",
        "request_id",
        "service_tier",
        "usage",
    ]
    assert body["answers"] == {
        "refund": {"type": "noul", "noul": 0.99},
        "department": {
            "type": "choice",
            "choice": "billing",
            "confidence": 1.0,
            "probabilities": {"billing": 1.0, "technical": 0.0},
        },
        "urgency": {
            "type": "score",
            "score": 2.0,
            "confidence": 1.0,
            "legend": {
                "0": "Low urgency",
                "1": "Needs attention today",
                "2": "Requires immediate action",
            },
            "probabilities": {"0": 0.0, "1": 0.0, "2": 1.0},
        },
    }
    assert body["id"] == body["request_id"]
    assert body["model"] == "openai/gpt-6-luna-20260922"
    assert body["provider"] == "openrouter"
    assert body["usage"] == {"input_tokens": 100, "output_tokens": 10}
    assert resp.headers["X-Upstream-Model"] == "openai/gpt-6-luna-20260922"
    assert resp.headers["X-Upstream-Provider"] == "OpenAI"
    assert resp.headers["X-Upstream-Calls"] == "1"
    assert resp.headers["X-Samples"] == "1"
    assert float(resp.headers["X-Cost-USD"]) == pytest.approx(0.00001)


def test_upstream_request_is_non_reasoning_strict_and_routed(client, fake):
    post(client)
    (sent,) = fake.requests
    assert sent["model"] == "openai/gpt-6-luna"  # jev-latest maps to the configured Luna
    assert sent["reasoning"] == {"effort": "none", "exclude": True}
    assert sent["response_format"]["json_schema"]["strict"] is True
    assert sent["provider"] == {"require_parameters": True}
    assert sent["seed"] == 0
    assert sent["max_tokens"] >= 256
    assert "temperature" not in sent and "logprobs" not in sent


def test_only_allowlisted_models_pass_through(make_client, fake):
    with make_client(allowed_models="openai/gpt-5.6-luna") as c:
        post(c, dict(JEV_EXAMPLE, model="openai/gpt-5.6-luna"))
        post(c, dict(JEV_EXAMPLE, model="sao10k/l3-lunaris-8b"))
    assert [r["model"] for r in fake.requests] == ["openai/gpt-5.6-luna", "openai/gpt-6-luna"]


def test_bad_output_is_retried_with_new_seed(client, fake):
    fake.push_content("I think billing.", JEV_LIKE_OUTPUT)
    resp = post(client)
    assert resp.status_code == 200
    assert [r["seed"] for r in fake.requests] == [0, 10_000]
    assert resp.json()["usage"]["input_tokens"] == 200


def test_truncated_output_is_retried(client, fake):
    fake.push(completion('{"q1": 9', finish_reason="length"))
    fake.push_content(JEV_LIKE_OUTPUT)
    assert post(client).status_code == 200
    first, second = fake.requests
    assert second["max_tokens"] == 2 * first["max_tokens"]


def test_bad_output_twice_is_502(client, fake):
    fake.push_content("nope", "still nope")
    resp = post(client)
    assert resp.status_code == 502
    assert error_type(resp) == "upstream_output_error"


@pytest.mark.parametrize(
    ("status", "expected", "kind", "calls"),
    [
        (429, 429, "rate_limit_error", 3),
        (503, 529, "overloaded_error", 3),
        (401, 502, "upstream_error", 1),
        (402, 502, "upstream_error", 1),
        (400, 502, "upstream_error", 1),
    ],
)
def test_upstream_status_mapping(client, fake, status, expected, kind, calls):
    error = {"error": {"code": status, "message": "boom"}}
    fake.push(*[httpx.Response(status, json=error) for _ in range(3)])
    resp = post(client)
    assert resp.status_code == expected
    assert error_type(resp) == kind
    assert len(fake.requests) == calls  # only 429/5xx are retried (max_retries=2)


def test_transient_failure_recovers_on_retry(client, fake):
    fake.push(httpx.Response(503, json={"error": {"message": "busy"}}))
    fake.push_content(JEV_LIKE_OUTPUT)
    assert post(client).status_code == 200


def test_timeout_is_529(client, fake):
    fake.push(*[httpx.ReadTimeout("slow") for _ in range(3)])
    resp = post(client)
    assert resp.status_code == 529


def test_error_inside_200_body(client, fake):
    fake.push(*[{"error": {"code": 503, "message": "provider down"}} for _ in range(3)])
    assert post(client).status_code == 529


def test_upstream_401_with_configured_key_is_502(client, fake):
    fake.push(httpx.Response(401, json={"error": {"message": "bad key"}}))
    resp = post(client)
    assert resp.status_code == 502
    assert "configured API key" in resp.json()["error"]["message"]


def test_validation_error_is_422_with_details(client, fake):
    body = {
        "model": "jev-latest",
        "state": "s",
        "questions": {"u": {"type": "score", "instructions": "x", "criteria": ["only one"]}},
    }
    resp = post(client, body)
    assert resp.status_code == 422
    err = resp.json()["error"]
    assert err["type"] == "invalid_request_error"
    assert any("criteria" in d["loc"] for d in err["details"])
    assert fake.requests == []


def test_auth(make_client):
    with make_client(api_keys="k1, k2") as c:
        assert post(c).status_code == 401
        assert post(c, Authorization="Bearer nope").status_code == 401
        assert post(c, Authorization="Bearer k2").status_code == 200
        # Auth is checked before the body is validated.
        assert post(c, {"model": "x"}).status_code == 401


def test_samples_are_averaged(client, fake):
    fake.push_content(
        dict(JEV_LIKE_OUTPUT, q1=90), dict(JEV_LIKE_OUTPUT, q1=60), dict(JEV_LIKE_OUTPUT, q1=30)
    )
    resp = post(client, **{"X-Samples": "3"})
    assert resp.status_code == 200
    assert resp.json()["answers"]["refund"]["noul"] == pytest.approx(0.6)
    assert sorted(r["seed"] for r in fake.requests) == [0, 1, 2]
    assert resp.headers["X-Samples"] == "3"
    assert resp.headers["X-Upstream-Calls"] == "3"


@pytest.mark.parametrize("value", ["0", "9", "many"])
def test_bad_samples_header(client, value):
    resp = post(client, **{"X-Samples": value})
    assert resp.status_code == 422


def test_many_questions_are_split_across_calls(make_client, fake):
    questions = {f"q{i}": {"type": "noul", "instructions": f"question {i}?"} for i in range(5)}
    with make_client(max_questions_per_call=2) as c:
        resp = post(c, {"model": "jev-latest", "state": "s", "questions": questions})
    assert resp.status_code == 200
    assert list(resp.json()["answers"]) == list(questions)
    sizes = sorted(
        len(r["response_format"]["json_schema"]["schema"]["properties"]) for r in fake.requests
    )
    assert sizes == [1, 2, 2]


def test_large_choice_uses_topk(client, fake):
    options = {f"label{i}": None for i in range(30)}
    body = {
        "model": "jev-latest",
        "state": "s",
        "questions": {"c": {"type": "choice", "instructions": "pick", "criteria": options}},
    }
    resp = post(client, body)
    assert resp.status_code == 200
    answer = resp.json()["answers"]["c"]
    assert answer["choice"] == "label0"
    assert len(answer["probabilities"]) == 30
    assert math.isclose(sum(answer["probabilities"].values()), 1.0, abs_tol=1e-4)
    schema = fake.requests[0]["response_format"]["json_schema"]["schema"]
    assert "top" in schema["properties"]["q1"]["properties"]


def test_calibration_applies_and_can_be_bypassed(make_client, fake, tmp_path):
    path = tmp_path / "calibration.json"
    cal = Calibration(choice=Platt(a=0.5))
    path.write_text(json.dumps({"openai/gpt-6-luna": cal.to_dict()}))
    output = dict(JEV_LIKE_OUTPUT, q2={"o1": 90, "o2": 10})
    fake.push_content(output, output)
    with make_client(calibration_path=path) as c:
        calibrated = post(c).json()["answers"]["department"]["confidence"]
        raw = post(c, **{"X-Calibration": "off"})
    assert raw.headers["X-Calibration"] == "off"
    assert raw.json()["answers"]["department"]["confidence"] == pytest.approx(0.9)
    assert 0.5 < calibrated < 0.9


def test_rate_limit(make_client):
    with make_client(rate_limit_per_minute=1) as c:
        assert post(c).status_code == 200
        resp = post(c)
    assert resp.status_code == 429
    assert int(resp.headers["Retry-After"]) >= 1


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok", "model": "openai/gpt-6-luna"}


def test_logs_one_json_line_per_decision(client):
    with captured_events() as events:
        resp = post(client)
    (event,) = events
    assert event["event"] == "decision"
    assert event["request_id"] == resp.json()["request_id"] == resp.headers["X-Request-Id"]
    assert event["questions"] == 3 and event["calls"] == 1


def test_errors_are_logged_with_request_id(client, fake):
    fake.push_content("nope", "still nope")
    with captured_events() as events:
        resp = post(client)
    (event,) = events
    assert event["event"] == "decision_error"
    assert event["status"] == 502 and event["type"] == "upstream_output_error"
    assert event["request_id"] == resp.headers["X-Request-Id"]


def test_unexpected_error_is_json_500_with_request_id(make_client, fake):
    fake.push(RuntimeError("bug with request data"))  # not a DecisionError: escapes the service
    with make_client(raise_server_exceptions=False) as client, captured_events() as events:
        resp = post(client)
    assert resp.status_code == 500
    assert resp.json() == {"error": {"type": "api_error", "message": "internal server error"}}
    (event,) = events
    assert event["event"] == "decision_error" and event["status"] == 500
    assert event["exception"] == "RuntimeError"  # the class only, never the message
    assert event["request_id"] == resp.headers["X-Request-Id"]


def test_one_failed_sample_does_not_fail_the_request(client, fake):
    def sample_zero_is_broken(body):
        return completion("not json") if body["seed"] % 10_000 == 0 else valid_reply(body)

    fake.push(*[sample_zero_is_broken] * 4)  # 3 samples + the retry of sample 0
    resp = post(client, **{"X-Samples": "3"})
    assert resp.status_code == 200, resp.text
    assert resp.headers["X-Samples-Failed"] == "1"
    assert resp.headers["X-Upstream-Calls"] == "4"
    assert resp.json()["usage"]["input_tokens"] == 400  # the failed sample was still billed


def test_request_deadline(make_client, fake):
    async def slow(body):
        await asyncio.sleep(2)
        return valid_reply(body)

    fake.push(slow)
    with make_client(request_timeout_s=0.1) as c:
        resp = post(c)
    assert resp.status_code == 529
    assert "0.1s" in resp.json()["error"]["message"]


def test_request_limits(make_client, fake):
    many = {f"q{i}": {"type": "noul", "instructions": "?"} for i in range(5)}
    with make_client(max_questions=4, max_state_chars=10) as c:
        too_many = post(c, {"model": "jev-latest", "state": "s", "questions": many})
        too_long = post(c, dict(JEV_EXAMPLE, state="x" * 11))
    assert too_many.status_code == 422 and "at most 4 questions" in too_many.text
    assert too_long.status_code == 422 and "10 characters" in too_long.text
    assert fake.requests == []


def test_non_ascii_key_is_401_not_500(make_client):
    with make_client(api_keys="k1") as c:
        headers = {"Authorization": "Bearer caf\xe9".encode("latin-1")}
        resp = c.post("/v1/systemone", json=JEV_EXAMPLE, headers=headers)
    assert resp.status_code == 401


def test_openai_flavor_sends_only_openai_fields(make_client, fake):
    with make_client(upstream_base_url="http://litellm:4000/v1") as c:
        resp = post(c)
    assert resp.status_code == 200
    (sent,) = fake.requests
    assert set(sent) == {
        "model",
        "messages",
        "response_format",
        "seed",
        "max_completion_tokens",
        "reasoning_effort",
    }
    assert sent["reasoning_effort"] == "none"
    assert "x-title" not in fake.headers[0]
    assert resp.json()["provider"] == "openai"


def test_cost_from_litellm_header(make_client, fake):
    body = completion(json.dumps(JEV_LIKE_OUTPUT), cost=None, provider=None)
    fake.push(httpx.Response(200, json=body, headers={"x-litellm-response-cost": "0.00042"}))
    with make_client(upstream_base_url="http://litellm:4000/v1") as c:
        resp = post(c)
    assert float(resp.headers["X-Cost-USD"]) == pytest.approx(0.00042)
    assert "X-Upstream-Provider" not in resp.headers


def test_forward_auth_uses_the_callers_key(make_client, fake):
    with make_client(forward_auth=True, upstream_api_key=None) as c:
        missing = post(c)
        ok = post(c, Authorization="Bearer sk-caller")
        fake.push(httpx.Response(401, json={"error": {"message": "invalid key"}}))
        rejected = post(c, Authorization="Bearer sk-wrong")
    assert missing.status_code == 401
    assert ok.status_code == 200
    assert fake.headers[0]["authorization"] == "Bearer sk-caller"
    assert rejected.status_code == 401
    assert error_type(rejected) == "authentication_error"
