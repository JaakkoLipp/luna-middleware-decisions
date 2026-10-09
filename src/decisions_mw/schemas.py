"""Jev System One wire format (`POST /v1/systemone`)."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# Jev accepts text or structured JSON for `state` and `instructions`, but never null.
Content = str | dict[str, Any] | list[Any]

MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10


class NoulCriteria(BaseModel):
    model_config = ConfigDict(extra="forbid")

    true: str
    false: str


class NoulQuestion(BaseModel):
    type: Literal["noul"]
    instructions: Content
    criteria: NoulCriteria | None = None


class ChoiceQuestion(BaseModel):
    type: Literal["choice"]
    instructions: Content
    criteria: Annotated[dict[str, str | None], Field(min_length=1, max_length=MAX_CHOICE_OPTIONS)]


class ScoreQuestion(BaseModel):
    type: Literal["score"]
    instructions: Content
    criteria: Annotated[list[str], Field(min_length=MIN_SCORE_LEVELS, max_length=MAX_SCORE_LEVELS)]


Question = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]


class DecisionRequest(BaseModel):
    model: str
    state: Content
    questions: Annotated[dict[str, Question], Field(min_length=1)]


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    confidence: float
    probabilities: dict[str, float]


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    confidence: float
    legend: dict[str, str]
    probabilities: dict[str, float]


Answer = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class DecisionResponse(BaseModel):
    answers: dict[str, Answer]
    id: str
    model: str
    provider: str
    request_id: str
    service_tier: str = "standard"
    usage: Usage
