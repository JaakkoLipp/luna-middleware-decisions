"""Unit tests for the deployment check script's own logic (scripts/live_check.py)."""

import copy
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "live_check.py"
spec = importlib.util.spec_from_file_location("live_check", SCRIPT)
live_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(live_check)

QUESTIONS = live_check.JEV_EXAMPLE["questions"]
# Jev's reference response for its reference request.
GOOD = {
    "answers": {
        "refund": {"type": "noul", "noul": 0.99},
        "department": {
            "type": "choice",
            "choice": "billing",
            "confidence": 1.0,
            "probabilities": {"technical": 0.0, "billing": 1.0},
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
    },
    "id": "x",
    "model": "m",
    "provider": "p",
    "request_id": "x",
    "service_tier": "standard",
    "usage": {"input_tokens": 279, "output_tokens": 20},
}


def broken(mutate):
    body = copy.deepcopy(GOOD)
    mutate(body)
    return live_check.contract_problems(QUESTIONS, body)


def test_jev_reference_response_passes():
    assert live_check.contract_problems(QUESTIONS, GOOD) == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda b: b.pop("model"),
        lambda b: b["usage"].update(output_tokens="20"),
        lambda b: b["answers"].pop("urgency"),
        lambda b: b["answers"]["refund"].update(noul=1.5),
        lambda b: b["answers"]["department"].update(choice="sales"),
        lambda b: b["answers"]["department"]["probabilities"].update(billing=0.5),
        lambda b: b["answers"]["department"].update(confidence=0.5),
        lambda b: b["answers"]["urgency"].update(score=1.0),
        lambda b: b["answers"]["urgency"]["legend"].update({"0": "Low"}),
        lambda b: b["answers"]["urgency"].update(type="choice"),
    ],
    ids=[
        "missing-field",
        "usage-not-int",
        "missing-answer",
        "noul-out-of-range",
        "unknown-choice",
        "probabilities-dont-sum",
        "confidence-not-top",
        "score-not-weighted",
        "legend-mismatch",
        "wrong-type",
    ],
)
def test_contract_violations_are_reported(mutate):
    assert broken(mutate)


def test_non_object_body():
    assert live_check.contract_problems(QUESTIONS, None) == ["response is not a JSON object"]


@pytest.mark.parametrize(
    ("answer", "check", "value", "ok"),
    [
        ({"noul": 0.9}, "noul>=", 0.8, True),
        ({"noul": 0.7}, "noul>=", 0.8, False),
        ({"noul": 0.1}, "noul<=", 0.2, True),
        ({"noul": 0.5}, "noul_between", (0.1, 0.9), True),
        ({"noul": 0.99}, "noul_between", (0.1, 0.9), False),
        ({"choice": "billing", "confidence": 0.9}, "choice", "billing", True),
        ({"choice": "sales", "confidence": 0.9}, "choice", "billing", False),
        ({"score": 1.8}, "score>=", 1.5, True),
    ],
)
def test_expectations(answer, check, value, ok):
    assert live_check.check_expectation(answer, check, value)[0] is ok


def test_cases_are_valid_requests():
    from decisions_mw.schemas import DecisionRequest

    for case in live_check.CASES:
        DecisionRequest.model_validate(
            {"model": "m", "state": case.state, "questions": case.questions}
        )
        assert set(case.expect) <= set(case.questions)
    DecisionRequest.model_validate({"model": "m", **live_check.AMBIGUOUS})


DECISIONS_GOOD = {
    "answers": [
        {"type": "predicate", "name": "refund", "probability": 0.99},
        {
            "type": "choice",
            "name": "department",
            "choice": "billing",
            "confidence": 0.9,
            "probabilities": [
                {"value": "billing", "probability": 0.9},
                {"value": "technical", "probability": 0.1},
            ],
        },
        {
            "type": "score",
            "name": "urgency",
            "score": 1.1,
            "confidence": 0.7,
            "probabilities": [
                {"value": 0, "label": "low", "probability": 0.1},
                {"value": 1, "label": "today", "probability": 0.7},
                {"value": 2, "label": "now", "probability": 0.2},
            ],
        },
    ],
    "model": "m",
    "usage": {
        "input_tokens": 10,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        "output_tokens": 5,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 15,
    },
}
DQ = live_check.DECISIONS_QUESTIONS


def test_decisions_reference_response_passes():
    assert live_check.decisions_contract_problems(DQ, DECISIONS_GOOD) == []


def test_decisions_validator_accepts_the_real_endpoint(make_client):
    with make_client() as client:
        body = {"model": "gpt-6-luna", "input": "Refund please.", "questions": DQ}
        resp = client.post("/v1/decisions", json=body)
    assert resp.status_code == 200
    assert live_check.decisions_contract_problems(DQ, resp.json()) == []


def test_decisions_refusals_are_allowed():
    body = copy.deepcopy(DECISIONS_GOOD)
    body["answers"][1] = {"type": "refusal", "name": "department"}
    assert live_check.decisions_contract_problems(DQ, body) == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda b: b.pop("usage"),
        lambda b: b["answers"].pop(),
        lambda b: b["answers"][0].pop("name"),
        lambda b: b["answers"][0].update(probability=2),
        lambda b: b["answers"][1]["probabilities"].reverse(),
        lambda b: b["answers"][1].update(choice="sales"),
        lambda b: b["answers"][1].update(confidence=0.5),
        lambda b: b["answers"][2]["probabilities"][0].update(label="LOW"),
        lambda b: b["answers"][2].update(score=2.0),
    ],
    ids=[
        "no-usage",
        "missing-answer",
        "name-not-echoed",
        "probability-out-of-range",
        "choices-out-of-order",
        "unknown-choice",
        "confidence-not-top",
        "level-label-mismatch",
        "score-not-weighted",
    ],
)
def test_decisions_violations_are_reported(mutate):
    body = copy.deepcopy(DECISIONS_GOOD)
    mutate(body)
    assert live_check.decisions_contract_problems(DQ, body)


def test_decisions_boolean_values_must_stay_booleans():
    questions = [
        {
            "type": "choice",
            "name": "paid",
            "instructions": "?",
            "choices": [{"value": True}, {"value": False}],
        }
    ]
    answer = {
        "type": "choice",
        "name": "paid",
        "choice": 1,  # equals True in Python, but the wrong JSON type
        "confidence": 0.8,
        "probabilities": [{"value": 1, "probability": 0.8}, {"value": 0, "probability": 0.2}],
    }
    body = {**DECISIONS_GOOD, "answers": [answer]}
    assert live_check.decisions_contract_problems(questions, body)
