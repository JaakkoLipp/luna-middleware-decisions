import copy

import pytest
from pydantic import ValidationError

from decisions_mw.schemas import ChoiceQuestion, DecisionRequest, NoulQuestion, ScoreQuestion

from .conftest import JEV_EXAMPLE


def request_with(question: dict) -> dict:
    return {"model": "jev-latest", "state": "s", "questions": {"q": question}}


def test_jev_reference_example_parses():
    req = DecisionRequest.model_validate(JEV_EXAMPLE)
    assert isinstance(req.questions["refund"], NoulQuestion)
    assert isinstance(req.questions["department"], ChoiceQuestion)
    assert isinstance(req.questions["urgency"], ScoreQuestion)


def test_structured_state_and_instructions_are_allowed():
    body = request_with({"type": "noul", "instructions": {"rule": "Is `limit` exceeded?"}})
    body["state"] = {"spend": 120, "limit": 100}
    DecisionRequest.model_validate(body)
    body["state"] = [{"role": "user", "text": "hi"}]
    DecisionRequest.model_validate(body)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda b: b.update(state=None),
        lambda b: b["questions"]["refund"].update(instructions=None),
        lambda b: b.update(questions={}),
        lambda b: b["questions"]["refund"].update(type="maybe"),
        lambda b: b["questions"]["refund"].update(criteria={"true": "yes"}),
        lambda b: b["questions"]["refund"].update(criteria={"true": "y", "false": "n", "x": "?"}),
        lambda b: b["questions"]["department"].update(criteria={}),
        lambda b: b["questions"]["department"].pop("criteria"),
        lambda b: b["questions"]["urgency"].update(criteria=["only one"]),
        lambda b: b["questions"]["urgency"].update(criteria=[str(i) for i in range(11)]),
    ],
    ids=[
        "null-state",
        "null-instructions",
        "no-questions",
        "unknown-type",
        "noul-criteria-missing-false",
        "noul-criteria-extra-key",
        "choice-no-options",
        "choice-missing-criteria",
        "score-one-level",
        "score-eleven-levels",
    ],
)
def test_jev_validation_rules(mutate):
    body = copy.deepcopy(JEV_EXAMPLE)
    mutate(body)
    with pytest.raises(ValidationError):
        DecisionRequest.model_validate(body)


def test_choice_option_limit():
    ok = {f"o{i}": None for i in range(255)}
    DecisionRequest.model_validate(
        request_with({"type": "choice", "instructions": "x", "criteria": ok})
    )
    too_many = {f"o{i}": None for i in range(256)}
    with pytest.raises(ValidationError):
        DecisionRequest.model_validate(
            request_with({"type": "choice", "instructions": "x", "criteria": too_many})
        )


def test_score_level_bounds():
    for n in (2, 10):
        DecisionRequest.model_validate(
            request_with({"type": "score", "instructions": "x", "criteria": ["l"] * n})
        )
