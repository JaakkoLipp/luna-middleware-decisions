"""POST /v1/decisions, driven by the official OpenAI SDK against the app in-process."""

import json
from contextlib import contextmanager

import openai
import pytest
from openai import OpenAI
from openai.types import Decision

from .conftest import completion

QUESTIONS = [
    {"type": "predicate", "name": "refund", "instructions": "Does the customer request a refund?"},
    {
        "type": "choice",
        "name": "department",
        "instructions": "Which team should handle this message?",
        "choices": [
            {"value": "billing", "description": "Payments and refunds"},
            {"value": "technical", "description": "Software errors"},
        ],
    },
    {
        "type": "score",
        "name": "urgency",
        "instructions": "How urgent is this incident?",
        "levels": [
            {"label": "low", "description": "Low urgency"},
            {"label": "today"},
            {"label": "now", "description": "Requires immediate action"},
        ],
    },
]
OUTPUT = {"q1": 99, "q2": {"o1": 100, "o2": 0}, "q3": {"l0": 0, "l1": 0, "l2": 100}}


@pytest.fixture
def make_sdk(make_client):
    @contextmanager
    def make(key: str = "test-key", **overrides):
        with make_client(**overrides) as test_client:
            yield OpenAI(
                base_url="http://testserver/v1",
                api_key=key,
                http_client=test_client,
                max_retries=0,
            )

    return make


@pytest.fixture
def sdk(make_sdk):
    with make_sdk() as client:
        yield client


def raw_post(sdk: OpenAI, body: dict):
    return sdk._client.post("/v1/decisions", json=body)


def test_sdk_round_trip(sdk, fake):
    fake.push_content(OUTPUT)
    decision = sdk.decisions.create(
        model="gpt-6-luna", input="Please refund the duplicate payment.", questions=QUESTIONS
    )
    assert isinstance(decision, Decision)
    refund, department, urgency = decision.answers

    assert (refund.type, refund.name, refund.probability) == ("predicate", "refund", 0.99)

    assert department.type == "choice" and department.choice == "billing"
    assert department.confidence == 1.0
    assert [(p.value, p.probability) for p in department.probabilities] == [
        ("billing", 1.0),
        ("technical", 0.0),
    ]

    assert urgency.type == "score" and urgency.score == 2.0
    assert [(p.value, p.label, p.probability) for p in urgency.probabilities] == [
        (0, "low", 0.0),
        (1, "today", 0.0),
        (2, "now", 1.0),
    ]
    assert decision.usage.input_tokens == 100
    assert decision.usage.total_tokens == 110
    assert decision.model == "openai/gpt-6-luna-20260922"

    # What the model saw: option and level text built from value/label and description.
    prompt = fake.requests[0]["messages"][1]["content"]
    assert "o1 = billing: Payments and refunds" in prompt
    assert "l0: low: Low urgency" in prompt and "l1: today\n" in prompt


def test_names_are_optional_and_answers_follow_question_order(sdk, fake):
    fake.push_content({"q1": 10, "q2": 90})
    questions = [
        {"type": "predicate", "instructions": "Is it raining?"},
        {"type": "predicate", "instructions": "Is it cloudy?"},
    ]
    resp = raw_post(sdk, {"model": "gpt-6-luna", "input": "Grey sky.", "questions": questions})
    assert resp.status_code == 200
    answers = resp.json()["answers"]
    assert [a["probability"] for a in answers] == [0.1, 0.9]
    assert all("name" not in a for a in answers)


def test_boolean_choice_values_keep_their_type(sdk, fake):
    fake.push_content({"q1": {"o1": 80, "o2": 20}})
    question = {
        "type": "choice",
        "name": "eligible",
        "instructions": "Is the customer eligible?",
        "choices": [{"value": True}, {"value": False}],
    }
    resp = raw_post(sdk, {"model": "gpt-6-luna", "input": "Gold member.", "questions": [question]})
    answer = resp.json()["answers"][0]
    assert answer["choice"] is True
    assert [p["value"] for p in answer["probabilities"]] == [True, False]


def test_message_input_text_is_joined(sdk, fake):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "first part"},
                {"type": "input_text", "text": "second part"},
            ],
        },
        {"role": "user", "type": "message", "content": "third part"},
    ]
    sdk.decisions.create(model="gpt-6-luna", input=messages, questions=QUESTIONS[:1])
    prompt = fake.requests[0]["messages"][1]["content"]
    assert "first part\n\nsecond part\n\nthird part" in prompt


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            {
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_image", "image_url": "data:image/png;base64,AA"}
                        ],
                    }
                ],
                "questions": QUESTIONS[:1],
            },
            "image input is not supported",
        ),
        (
            {"input": "x", "questions": [QUESTIONS[0], QUESTIONS[0]]},
            "names must be unique",
        ),
        (
            {
                "input": "x",
                "questions": [
                    {
                        "type": "choice",
                        "instructions": "?",
                        "choices": [{"value": True}, {"value": "true"}],
                    }
                ],
            },
            "distinct as text",
        ),
        (
            {
                "input": "x",
                "questions": [{"type": "score", "instructions": "?", "levels": [{"label": "one"}]}],
            },
            "request failed validation",
        ),
        ({"input": "x", "questions": []}, "request failed validation"),
    ],
    ids=["image", "duplicate-names", "bool-and-string-collide", "one-level", "no-questions"],
)
def test_invalid_requests_are_400_bad_request(sdk, fake, body, message):
    with pytest.raises(openai.BadRequestError) as err:
        sdk.decisions.create(model="gpt-6-luna", **body)
    assert err.value.status_code == 400
    assert message in str(err.value)
    assert fake.requests == []


def test_usage_details_come_from_upstream(sdk, fake):
    body = completion(json.dumps({"q1": 50}))
    body["usage"]["prompt_tokens_details"] = {"cached_tokens": 64}
    body["usage"]["completion_tokens_details"] = {"reasoning_tokens": 0}
    fake.push(body)
    decision = sdk.decisions.create(model="gpt-6-luna", input="x", questions=QUESTIONS[:1])
    assert decision.usage.input_tokens_details.cached_tokens == 64
    assert decision.usage.output_tokens_details.reasoning_tokens == 0


def test_auth_and_extension_headers(make_sdk, fake):
    with make_sdk(key="wrong", api_keys="k1") as client, pytest.raises(openai.AuthenticationError):
        client.decisions.create(model="gpt-6-luna", input="x", questions=QUESTIONS[:1])
    with make_sdk(key="k1", api_keys="k1") as client:
        raw = client.decisions.with_raw_response.create(
            model="gpt-6-luna",
            input="x",
            questions=QUESTIONS[:1],
            extra_headers={"X-Samples": "3"},
        )
    assert raw.headers["X-Samples"] == "3"
    assert raw.headers["X-Upstream-Calls"] == "3"
    assert raw.headers["X-Request-Id"]
    assert raw.parse().answers[0].type == "predicate"


def test_jev_endpoint_keeps_422(sdk):
    resp = sdk._client.post("/v1/systemone", json={"model": "m", "state": "s", "questions": {}})
    assert resp.status_code == 422
