"""Orchestrates one decision request: fan out upstream calls, combine samples, calibrate."""

import asyncio
import secrets
import string
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime

from .aggregate import OutputParseError, build_answer, ensemble, parse_output
from .calibration import IDENTITY, CalibrationStore
from .config import Settings
from .errors import BadUpstreamOutputError, DecisionError, InvalidRequestError, OverloadedError
from .output_schema import build_response_format, estimate_output_tokens
from .prompt import QuestionSpec, build_messages, build_specs, chunk, render_content
from .schemas import Content, DecisionRequest, DecisionResponse, Usage
from .upstream import Completion, UpstreamClient

PARSE_ATTEMPTS = 2
RETRY_SEED_OFFSET = 10_000
MIN_MAX_TOKENS = 256
REASONING_TOKEN_ALLOWANCE = 2048
_ALNUM = string.ascii_letters + string.digits


def new_request_id() -> str:
    """Timestamp plus random suffix, shaped like Jev's request ids."""
    suffix = "".join(secrets.choice(_ALNUM) for _ in range(10))
    return datetime.now(UTC).strftime("%Y%m%d%H%M%S%f") + suffix


@dataclass(frozen=True)
class DecisionMeta:
    upstream_model: str
    upstream_providers: list[str]
    samples: int
    samples_failed: int
    calls: int
    cost_usd: float | None
    latency_ms: float
    calibrated: bool
    cached_tokens: int = 0
    reasoning_tokens: int = 0


class DecisionService:
    def __init__(self, settings: Settings, client: UpstreamClient, calibration: CalibrationStore):
        self.settings = settings
        self.client = client
        self.calibration = calibration
        self._sem = asyncio.Semaphore(settings.max_concurrency)

    def resolve_model(self, requested: str) -> str:
        """Allowlisted models pass through; anything else (e.g. `jev-latest`) uses the default."""
        return (
            requested if requested in self.settings.allowed_model_set else self.settings.luna_model
        )

    def _check_limits(self, req: DecisionRequest, samples: int) -> None:
        s = self.settings
        if not 1 <= samples <= s.max_samples:
            raise InvalidRequestError(f"samples must be between 1 and {s.max_samples}")
        if len(req.questions) > s.max_questions:
            raise InvalidRequestError(f"at most {s.max_questions} questions per request")
        if len(render_content(req.state)) > s.max_state_chars:
            raise InvalidRequestError(f"state is longer than {s.max_state_chars} characters")

    def _max_tokens(self, group: list[QuestionSpec]) -> int:
        n = max(MIN_MAX_TOKENS, 2 * estimate_output_tokens(group))
        if self.settings.reasoning_effort not in ("", "none"):
            n += REASONING_TOKEN_ALLOWANCE
        return n

    async def _sample(
        self,
        model: str,
        state: Content,
        group: list[QuestionSpec],
        index: int,
        api_key: str | None,
    ) -> tuple[dict[str, list[float]], list[Completion]]:
        messages = build_messages(state, group)
        response_format = build_response_format(group)
        completions: list[Completion] = []
        max_tokens = self._max_tokens(group)
        problem = ""
        for attempt in range(PARSE_ATTEMPTS):
            seed = self.settings.seed
            if seed is not None:
                seed += index + attempt * RETRY_SEED_OFFSET
            async with self._sem:
                completion = await self.client.complete(
                    model=model,
                    messages=messages,
                    response_format=response_format,
                    max_tokens=max_tokens,
                    seed=seed,
                    api_key=api_key,
                )
            completions.append(completion)
            if completion.finish_reason == "length":
                problem = "completion was truncated (finish_reason=length)"
                max_tokens *= 2
                continue
            try:
                return parse_output(completion.content, group), completions
            except OutputParseError as e:
                problem = str(e)
        err = BadUpstreamOutputError(
            f"upstream output did not match the decision schema after {PARSE_ATTEMPTS} "
            f"attempts: {problem}"
        )
        err.completions = completions  # still billed; counted in usage if other samples succeed
        raise err

    async def decide(
        self,
        req: DecisionRequest,
        *,
        request_id: str,
        samples: int | None = None,
        calibrate: bool = True,
        api_key: str | None = None,
    ) -> tuple[DecisionResponse, DecisionMeta]:
        """Answer every question. Samples that fail are skipped as long as each group of
        questions keeps at least one good sample; otherwise the group's first error is raised."""
        start = time.perf_counter()
        k = samples if samples is not None else self.settings.samples
        self._check_limits(req, k)
        model = self.resolve_model(req.model)
        specs = build_specs(
            req.questions, topk_threshold=self.settings.topk_threshold, topk=self.settings.topk
        )
        groups = chunk(specs, self.settings.max_questions_per_call)
        jobs = [(g, i) for g in range(len(groups)) for i in range(k)]

        timeout = self.settings.request_timeout_s
        try:
            async with asyncio.timeout(timeout):
                results = await asyncio.gather(
                    *(self._sample(model, req.state, groups[g], i, api_key) for g, i in jobs),
                    return_exceptions=True,
                )
        except TimeoutError:
            raise OverloadedError(f"request did not finish within {timeout:g}s") from None

        per_question: dict[str, list[list[float]]] = defaultdict(list)
        completions: list[Completion] = []
        ok = [0] * len(groups)
        first_error: dict[int, DecisionError] = {}
        for (g, _), result in zip(jobs, results, strict=True):
            if isinstance(result, BaseException):
                if not isinstance(result, DecisionError):
                    raise result
                first_error.setdefault(g, result)
                completions.extend(getattr(result, "completions", []))
                continue
            distributions, calls = result
            ok[g] += 1
            completions.extend(calls)
            for qid, dist in distributions.items():
                per_question[qid].append(dist)
        for g, successes in enumerate(ok):
            if successes == 0:
                raise first_error[g]

        cal = self.calibration.for_model(model) if calibrate else IDENTITY
        answers = {
            s.qid: build_answer(s, cal.apply(s.kind, ensemble(per_question[s.qid]))) for s in specs
        }

        costs = [c.cost for c in completions]
        upstream_model = completions[0].model
        response = DecisionResponse(
            answers=answers,
            id=request_id,
            model=upstream_model,
            provider="openrouter" if self.settings.flavor == "openrouter" else "openai",
            request_id=request_id,
            usage=Usage(
                input_tokens=sum(c.input_tokens for c in completions),
                output_tokens=sum(c.output_tokens for c in completions),
            ),
        )
        meta = DecisionMeta(
            upstream_model=upstream_model,
            upstream_providers=sorted({c.provider for c in completions if c.provider}),
            samples=k,
            samples_failed=len(jobs) - sum(ok),
            calls=len(completions),
            cost_usd=sum(costs) if all(c is not None for c in costs) else None,
            latency_ms=(time.perf_counter() - start) * 1000,
            calibrated=calibrate,
            cached_tokens=sum(c.cached_tokens for c in completions),
            reasoning_tokens=sum(c.reasoning_tokens for c in completions),
        )
        return response, meta
