"""OpenAI Decisions API format (`POST /v1/decisions`), translated onto the Jev engine.

Mirrors the request and response types of the official SDK (`openai` 3.26): questions are an
array and answers come back in the same order. `name` is optional and echoed when given. Choice
values are strings or booleans and are returned with their original type.

Differences from OpenAI's hosted API:
- Image parts (`input_image`) are rejected: this deployment answers from text only.
- No `refusal` answers. Upstream refusals surface as errors instead.
- A boolean and a string with the same text (`true` and `"true"`) can't be used as two options of
  one question, because the model sees option text.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StrictBool, StrictStr

from .errors import InvalidRequestError
from .schemas import (
    MAX_CHOICE_OPTIONS,
    MAX_SCORE_LEVELS,
    MIN_SCORE_LEVELS,
    ChoiceAnswer,
    DecisionRequest,
    DecisionResponse,
    NoulAnswer,
    ScoreAnswer,
)
from .service import DecisionMeta

ChoiceValue = StrictStr | StrictBool


# --- request -----------------------------------------------------------------------------------


class InputText(BaseModel):
    type: Literal["input_text"]
    text: str


class InputImage(BaseModel):
    type: Literal["input_image"]
    image_url: str
    detail: Literal["low", "high", "auto", "original"] | None = None


InputPart = Annotated[InputText | InputImage, Field(discriminator="type")]


class InputMessage(BaseModel):
    role: Literal["user"]
    content: str | list[InputPart]
    type: Literal["message"] | None = None


class PredicateQuestion(BaseModel):
    type: Literal["predicate"]
    instructions: str
    name: str | None = None


class ChoiceOption(BaseModel):
    value: ChoiceValue
    description: str | None = None


class ChoiceQuestion(BaseModel):
    type: Literal["choice"]
    instructions: str
    name: str | None = None
    choices: Annotated[list[ChoiceOption], Field(min_length=1, max_length=MAX_CHOICE_OPTIONS)]


class ScoreLevel(BaseModel):
    label: str
    description: str | None = None


class ScoreQuestion(BaseModel):
    type: Literal["score"]
    instructions: str
    name: str | None = None
    levels: Annotated[
        list[ScoreLevel], Field(min_length=MIN_SCORE_LEVELS, max_length=MAX_SCORE_LEVELS)
    ]


Question = Annotated[
    PredicateQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")
]


class DecisionsRequest(BaseModel):
    model: str
    input: str | Annotated[list[InputMessage], Field(min_length=1)]
    questions: Annotated[list[Question], Field(min_length=1)]
    safety_identifier: str | None = None


# --- response ----------------------------------------------------------------------------------


class PredicateAnswer(BaseModel):
    type: Literal["predicate"] = "predicate"
    name: str | None = None
    probability: float


class ChoiceProbability(BaseModel):
    value: ChoiceValue
    probability: float


class ChoiceAnswerOut(BaseModel):
    type: Literal["choice"] = "choice"
    name: str | None = None
    choice: ChoiceValue
    confidence: float
    probabilities: list[ChoiceProbability]


class ScoreProbability(BaseModel):
    value: int
    label: str
    probability: float


class ScoreAnswerOut(BaseModel):
    type: Literal["score"] = "score"
    name: str | None = None
    score: float
    confidence: float
    probabilities: list[ScoreProbability]


Answer = Annotated[PredicateAnswer | ChoiceAnswerOut | ScoreAnswerOut, Field(discriminator="type")]


class InputTokensDetails(BaseModel):
    cache_write_tokens: int = 0
    cached_tokens: int = 0


class OutputTokensDetails(BaseModel):
    reasoning_tokens: int = 0


class DecisionsUsage(BaseModel):
    input_tokens: int
    input_tokens_details: InputTokensDetails
    output_tokens: int
    output_tokens_details: OutputTokensDetails
    total_tokens: int


class DecisionsResponse(BaseModel):
    answers: list[Answer]
    model: str
    usage: DecisionsUsage


# --- translation -------------------------------------------------------------------------------


def _option_text(value: str | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _state_text(input: str | list[InputMessage]) -> str:
    if isinstance(input, str):
        return input
    texts = []
    for message in input:
        if isinstance(message.content, str):
            texts.append(message.content)
            continue
        for part in message.content:
            if isinstance(part, InputImage):
                raise InvalidRequestError(
                    "image input is not supported by this deployment; send text only"
                )
            texts.append(part.text)
    return "\n\n".join(texts)


def _qid(index: int) -> str:
    return f"q{index}"


def to_jev(req: DecisionsRequest) -> DecisionRequest:
    names = [q.name for q in req.questions if q.name is not None]
    if len(names) != len(set(names)):
        raise InvalidRequestError("question names must be unique")
    questions: dict[str, dict] = {}
    for i, q in enumerate(req.questions):
        if isinstance(q, PredicateQuestion):
            questions[_qid(i)] = {"type": "noul", "instructions": q.instructions}
        elif isinstance(q, ChoiceQuestion):
            criteria: dict[str, str | None] = {}
            for option in q.choices:
                text = _option_text(option.value)
                if text in criteria:
                    raise InvalidRequestError(
                        f"questions[{i}]: choice values must be distinct as text; "
                        f"{text!r} appears more than once"
                    )
                criteria[text] = option.description
            questions[_qid(i)] = {
                "type": "choice",
                "instructions": q.instructions,
                "criteria": criteria,
            }
        else:
            levels = [
                f"{lv.label}: {lv.description}" if lv.description else lv.label for lv in q.levels
            ]
            questions[_qid(i)] = {
                "type": "score",
                "instructions": q.instructions,
                "criteria": levels,
            }
    return DecisionRequest.model_validate(
        {"model": req.model, "state": _state_text(req.input), "questions": questions}
    )


def from_jev(
    req: DecisionsRequest, result: DecisionResponse, meta: DecisionMeta
) -> DecisionsResponse:
    answers: list[PredicateAnswer | ChoiceAnswerOut | ScoreAnswerOut] = []
    for i, q in enumerate(req.questions):
        a = result.answers[_qid(i)]
        if isinstance(q, PredicateQuestion):
            assert isinstance(a, NoulAnswer)
            answers.append(PredicateAnswer(name=q.name, probability=a.noul))
        elif isinstance(q, ChoiceQuestion):
            assert isinstance(a, ChoiceAnswer)
            by_text = {_option_text(o.value): o.value for o in q.choices}
            answers.append(
                ChoiceAnswerOut(
                    name=q.name,
                    choice=by_text[a.choice],
                    confidence=a.confidence,
                    probabilities=[
                        ChoiceProbability(
                            value=o.value, probability=a.probabilities[_option_text(o.value)]
                        )
                        for o in q.choices
                    ],
                )
            )
        else:
            assert isinstance(a, ScoreAnswer)
            answers.append(
                ScoreAnswerOut(
                    name=q.name,
                    score=a.score,
                    confidence=a.confidence,
                    probabilities=[
                        ScoreProbability(
                            value=j, label=lv.label, probability=a.probabilities[str(j)]
                        )
                        for j, lv in enumerate(q.levels)
                    ],
                )
            )
    usage = result.usage
    return DecisionsResponse(
        answers=answers,
        model=result.model,
        usage=DecisionsUsage(
            input_tokens=usage.input_tokens,
            input_tokens_details=InputTokensDetails(cached_tokens=meta.cached_tokens),
            output_tokens=usage.output_tokens,
            output_tokens_details=OutputTokensDetails(reasoning_tokens=meta.reasoning_tokens),
            total_tokens=usage.input_tokens + usage.output_tokens,
        ),
    )
