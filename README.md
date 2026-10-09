# decisions-middleware

A decisions API backed by a non-reasoning **GPT Luna** model. You send a state and bounded
questions (yes/no, choice, score). You get back typed answers with a probability for every option.
It speaks two wire formats, both on the same engine:

| Endpoint | Format | Clients |
|---|---|---|
| `POST /v1/decisions` | [OpenAI Decisions API](https://developers.openai.com/api/docs/guides/decisions): `predicate`, `choice`, `score` | the official OpenAI SDK (`client.decisions.create`) |
| `POST /v1/systemone` | TypeSafe's Jev: `noul`, `choice`, `score` | Jev clients |

The model can be reached through **OpenRouter**, or through any **OpenAI-compatible endpoint**:
a LiteLLM proxy in front of an Azure deployment, Azure directly, or OpenAI. This matters because
OpenAI's own Decisions API isn't available on Azure. Microsoft has announced no date (checked
2026-10-09).

> **Status:**
> - 152 tests pass against a mocked upstream, including the official OpenAI SDK driving
>   `/v1/decisions`.
> - Both LiteLLM setups below have been run over real HTTP, with a real LiteLLM 1.104.2 and a fake
>   Azure deployment. LiteLLM's own OpenAI Decisions client accepted the middleware's responses.
> - It has **not yet run against the real Luna**, so latency, accuracy and calibration on real
>   traffic are unmeasured. Start with `scripts/live_check.py` and `eval/run_eval.py` (below).

## Quickstart (OpenRouter)

```bash
uv sync
export OPENROUTER_API_KEY=sk-or-...
export DMW_API_KEYS=my-client-key          # inbound auth; the service won't start without it
uv run uvicorn --factory decisions_mw.main:create_app --port 8000
```

With the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="my-client-key")
decision = client.decisions.create(
    model="gpt-6-luna",
    input="Please refund the duplicate payment.",
    questions=[
        {
            "type": "predicate",
            "name": "refund",
            "instructions": "Does the customer request a refund?",
        },
        {
            "type": "choice",
            "name": "department",
            "instructions": "Which team should handle this?",
            "choices": [
                {"value": "billing", "description": "Payments and refunds"},
                {"value": "technical", "description": "Software errors"},
            ],
        },
        {
            "type": "score",
            "name": "urgency",
            "instructions": "How urgent is this?",
            "levels": [{"label": "low"}, {"label": "today"}, {"label": "now"}],
        },
    ],
)
for answer in decision.answers:
    print(answer)
```

The same request in Jev format:

```bash
curl -s localhost:8000/v1/systemone -H 'Authorization: Bearer my-client-key' \
  -H 'Content-Type: application/json' -d '{
  "model": "jev-latest",
  "state": "Please refund the duplicate payment.",
  "questions": {
    "refund":     {"type": "noul",   "instructions": "Does the customer request a refund?"},
    "department": {"type": "choice", "instructions": "Which team should handle this message?",
                   "criteria": {"billing": "Payments and refunds", "technical": "Software errors"}},
    "urgency":    {"type": "score",  "instructions": "How urgent is this incident?",
                   "criteria": ["Low urgency", "Needs attention today", "Requires immediate action"]}
  }
}'
```

```json
{
  "answers": {
    "refund": {"type": "noul", "noul": 0.97},
    "department": {"type": "choice", "choice": "billing", "confidence": 0.95,
                   "probabilities": {"billing": 0.95, "technical": 0.05}},
    "urgency": {"type": "score", "score": 0.9, "confidence": 0.5,
                "legend": {"0": "Low urgency", "1": "Needs attention today", "2": "Requires immediate action"},
                "probabilities": {"0": 0.3, "1": 0.5, "2": 0.2}}
  },
  "id": "20261009112725611516OuOxgwuuPy",
  "model": "openai/gpt-6-luna-20260922",
  "provider": "openrouter",
  "request_id": "20261009112725611516OuOxgwuuPy",
  "service_tier": "standard",
  "usage": {"input_tokens": 412, "output_tokens": 31}
}
```

(The numbers above are illustrative.) For local experiments without auth, set
`DMW_ALLOW_ANONYMOUS=true`. To run with Docker: `docker build -t decisions-middleware .` and then
`docker run -p 8000:8000 -e OPENROUTER_API_KEY -e DMW_API_KEYS decisions-middleware`.

## Running behind LiteLLM (e.g. Azure)

In both setups below, the middleware calls your Azure deployment through LiteLLM's chat endpoint.
The setups differ in how clients reach the middleware.

The shared part of LiteLLM's `config.yaml`:

```yaml
model_list:
  # The chat model the middleware calls: your Azure deployment
  - model_name: luna-decisions
    litellm_params:
      model: azure/<deployment-name>
      api_base: https://<resource>.openai.azure.com
      api_key: os.environ/AZURE_API_KEY
      api_version: "<api-version>"
    model_info:
      base_model: azure/gpt-6-luna      # required unless the deployment is named gpt-6-luna
```

And the middleware's environment:

```bash
DMW_UPSTREAM_BASE_URL=http://litellm:4000/v1   # not openrouter.ai, so requests use plain OpenAI fields
DMW_FORWARD_AUTH=true                          # use the key LiteLLM sends for the model calls
DMW_LUNA_MODEL=luna-decisions                  # the chat model above
DMW_MAX_RETRIES=0                              # LiteLLM already retries; don't multiply them
```

### Option A: LiteLLM's built-in Decisions endpoints (recommended)

LiteLLM 1.104.2 has its own `/v1/decisions` (OpenAI format) and `/v1/systemone` (Jev) endpoints.
It translates between the two formats and sends requests to a backend chosen by model name. Its
built-in backends are OpenAI, OpenRouter and Perplexity, not Azure. A model entry that points at
the middleware fills that gap:

```yaml
  # add to model_list
  - model_name: gpt-6-luna                       # what clients put in "model"
    litellm_params:
      model: openai/gpt-6-luna
      api_base: http://decisions-mw:8000/v1      # LiteLLM calls <api_base>/decisions
      api_key: os.environ/DECISIONS_SERVICE_KEY  # a LiteLLM key allowed to use luna-decisions
```

Clients then use LiteLLM's standard endpoints with `model: "gpt-6-luna"`, either
`OpenAI(base_url="https://litellm.example/v1")` or Jev at `https://litellm.example/v1/systemone`.
Code written for OpenAI's hosted Decisions API runs unchanged. If Azure ships Decisions later,
only this model entry changes; clients don't.

What you give up with this route:
- **LiteLLM drops the extension headers.** It doesn't forward `X-Samples`/`X-Calibration`, and
  it doesn't return the middleware's `X-*` response headers. Cost comes back as LiteLLM's
  `x-litellm-response-cost` instead.
- **Jev validation errors are 400**, not Jev's 422.
- **Spend is recorded twice, under different keys.** LiteLLM charges the caller for the decisions
  call, and charges the middleware's model calls to `DECISIONS_SERVICE_KEY`. Per-caller spend is
  right, since it's computed from the token usage the middleware reports. Leave the service key out
  of budgets and totals.

### Option B: a forwarding route under its own prefix

```yaml
general_settings:
  pass_through_endpoints:
    - path: /decisions-mw
      target: http://decisions-mw:8000
      include_subpath: true
      forward_headers: true
      timeout: 60
```

Clients call `https://litellm.example/decisions-mw/v1/decisions` (OpenAI SDK
`base_url=".../decisions-mw/v1"`) or `.../decisions-mw/v1/systemone`. Everything works as if they
called the middleware directly:
- extension headers and the middleware's response headers go through;
- Jev errors are 422;
- the caller's own key is used for the model calls, so spend is counted once, per caller.

Don't forward `/v1/decisions` or `/v1/systemone` themselves: those paths belong to LiteLLM's
built-in endpoints, and which one wins depends on LiteLLM's route order.

### Pitfalls in either setup

- **`base_model` is required for a custom deployment name.** Without it, LiteLLM doesn't know the
  deployment is Luna and rejects `reasoning_effort` with a 400. Worse, with `drop_params: true` it
  silently drops `reasoning_effort`, so Luna runs at its default `medium` reasoning: slower and
  more expensive, with no error anywhere.
- **Don't configure fallbacks to other models** for the chat model. Calibration is keyed by model
  name, and other models may not support the strict schema.
- **Make the middleware reachable only through LiteLLM**, using network isolation.

**Verified** with LiteLLM 1.104.2:
- Azure receives only plain OpenAI fields: `reasoning_effort: "none"`, `max_completion_tokens` and
  the strict schema.
- Luna on Azure supports `reasoning_effort` values `none`, `low`, `medium`, `high` and `xhigh`,
  but not `minimal`.
- Through either option, the OpenAI SDK and Jev requests both return correct answers, with boolean
  choice values staying booleans.

## Testing a live deployment

`scripts/live_check.py` is an acceptance check you can run against the deployed endpoint, the
same URL your clients use. It needs only Python 3 (standard library), so you can run it from
anywhere without installing anything:

```bash
export LITELLM_KEY=sk-...
# Option A (LiteLLM's built-in endpoints): pass the LiteLLM model name
python3 scripts/live_check.py --url https://litellm.example/v1/systemone --key-env LITELLM_KEY \
  --model gpt-6-luna

# Option B (prefixed forwarding route), plus a direct probe of LiteLLM's chat endpoint:
# is reasoning really off, and does Azure return logprobs?
python3 scripts/live_check.py --url https://litellm.example/decisions-mw/v1/systemone \
  --key-env LITELLM_KEY \
  --chat-url https://litellm.example/v1/chat/completions --chat-model luna-decisions
```

| Area | What it checks |
|---|---|
| contract | Jev response shape, consistency of probabilities, confidence and score, request ids |
| auth | requests without a valid key are rejected |
| quality | clear-cut questions get the obvious answer: paid invoice, billing vs technical, outage urgency, 30-option top-k, structured JSON state |
| behaviour | prompt injection in the state is ignored; missing information gives an uncertain answer (WARN only) |
| paths | invalid requests and the request limit are rejected, many questions are split across calls, `X-Samples` |
| signals | reasoning looks off (output tokens per call), cost is reported, samples actually differ |
| latency | 10 requests, 2 at a time: client and server p50/p95 (WARN above `--max-p95-ms`) |
| decisions | `/v1/decisions` in OpenAI format: response shape, answers, boolean choice values, 400 on invalid requests |
| upstream | with `--chat-url`: `reasoning_effort: "none"` accepted with 0 reasoning tokens; whether `logprobs` come back |

Each check is PASS, WARN or FAIL. The exit code is 1 if anything fails, so it can gate a deploy.
The script recognises Option A's documented limitations (dropped headers, 400 errors) and reports
them as WARN, not FAIL. A run makes about 30 requests (roughly 35–45 model calls).

Useful options:
- `--json report.json` saves the results.
- `--skip-latency` makes a quicker run.
- `--max-questions` / `--max-questions-per-call` match non-default `DMW_*` limits.
- `--decisions-url` is needed if the Decisions endpoint isn't at the `--url` path with
  `/v1/systemone` replaced by `/v1/decisions`; `--skip-decisions` skips those checks.
- `--health-url` checks the middleware's `/healthz`, if it's reachable.

## API

### `POST /v1/decisions` (OpenAI format)

Requests and responses match the types in the official OpenAI SDK (`openai` 3.26):
- `input` is a string or user messages with `input_text` parts.
- `questions` is an array of `predicate`, `choice` (with `choices: [{value, description}]`) and
  `score` (with `levels: [{label, description}]`) questions, each with an optional `name`.
- `answers` come back in question order and echo the `name`. Choice values keep their type: a
  boolean option comes back as a boolean.
- `usage` includes `input_tokens_details.cached_tokens` and `output_tokens_details.reasoning_tokens`,
  as reported by the upstream.

Differences from OpenAI's hosted API:

| | OpenAI hosted | This middleware |
|---|---|---|
| Image input (`input_image`) | supported | rejected with 400: text only |
| `refusal` answers | possible | never; upstream refusals become errors |
| Options `true` and `"true"` in one question | distinct | rejected with 400, since the model sees option text |
| Minimum choices | 2 (also enforced by LiteLLM) | 1 |
| Probabilities | from the model | stated weights, calibrated (see "How it works") |
| Invalid request | 400 | 400 |

### `POST /v1/systemone` (Jev format)

The request and response bodies follow Jev's System One contract:

- `state` and `instructions` are text or JSON, never null.
- `noul` criteria, if given, need both `true` and `false`.
- `choice` takes 1–255 options.
- `score` takes 2–10 ordered levels. Its `score` is the probability-weighted level index (0-based).
- `confidence` is the probability of the top answer.
- Unknown body fields are ignored.

### Shared by both endpoints

Limits per request: `DMW_MAX_QUESTIONS` questions (64) and `DMW_MAX_STATE_CHARS` characters of
state (200,000). With `DMW_MAX_SAMPLES` = 8 and 8 questions per call, one request makes at most 64
upstream calls, plus retries.

`model`: `jev-latest`, or anything not listed in `DMW_ALLOWED_MODELS`, runs on `DMW_LUNA_MODEL`.
Callers can't route to arbitrary models.

Extensions live in headers, so the bodies stay compatible:

| Request header | Effect |
|---|---|
| `Authorization: Bearer <key>` | inbound auth (see `DMW_API_KEYS` / `DMW_FORWARD_AUTH`) |
| `X-Samples: k` | average k upstream samples (1..`DMW_MAX_SAMPLES`) |
| `X-Calibration: off` | return raw, uncalibrated probabilities |

| Response header | Meaning |
|---|---|
| `X-Request-Id` | on every response, errors included; matches the log line |
| `X-Upstream-Model`, `X-Upstream-Provider` | model and provider(s) that answered (provider: OpenRouter only) |
| `X-Upstream-Calls`, `X-Samples`, `X-Samples-Failed` | calls made (including retries), samples requested, samples that failed |
| `X-Cost-USD`, `X-Latency-Ms` | upstream-reported cost (OpenRouter or LiteLLM), server-side latency |
| `X-Calibration` | whether calibration was applied |

Errors are `{"error": {"type", "message"}}`:

| Status | When |
|---|---|
| 401 | missing or invalid API key, including a forwarded key the upstream rejects |
| 422 (400 on `/v1/decisions`) | request fails validation or limits (`details` lists each schema problem) |
| 429 | our rate limit, or the upstream still rate-limiting after retries (`Retry-After` set) |
| 529 | upstream 5xx, timeout or connection failure after retries, or `DMW_REQUEST_TIMEOUT_S` exceeded |
| 502 | not in Jev: the upstream rejected the request (configured key, credits, params) or Luna's output didn't fit the schema twice |

## How it works

On OpenRouter, no Luna provider offers `logprobs`, `top_logprobs`, `logit_bias` or `temperature`.
Token probabilities, the usual basis for a classifier like this, therefore aren't available.
Instead:

1. **One upstream call per request.** All questions go into one strict JSON schema
   (`response_format: json_schema, strict: true`) with reasoning off.
   - On OpenRouter this is `reasoning: {effort: "none"}`, plus `provider.require_parameters` so it
     only routes to providers that enforce the schema.
   - On OpenAI-compatible upstreams it is `reasoning_effort: "none"` with `max_completion_tokens`,
     and nothing else, because OpenAI and Azure reject unknown fields.
2. **Short aliases.** Question ids become `q1..qn`, because Jev never shows ids to the model.
   Options become `o1..on` and levels `l0..lN`, each shown next to its name and description. The
   state goes first, inside delimiters, as untrusted data. Requests that share a long state (over
   about 1,024 tokens, the minimum for OpenAI's automatic caching) can then reuse a cached prefix.
3. **The model returns integer weights from 0–100**, e.g. `{"q1": 97, "q2": {"o1": 95, "o2": 5}}`.
   Weights are clamped and normalized; if all are zero, the answer is uniform.
4. **Choices with more than `DMW_TOPK_THRESHOLD` options** ask for a top-k list instead. Weight
   not assigned to the list is spread evenly over the unlisted options, which keeps output short.
5. **More than `DMW_MAX_QUESTIONS_PER_CALL` questions** are split into parallel calls.
6. **`X-Samples: k`** makes k calls with different seeds and averages the distributions. A sample
   that fails is skipped, as long as every group of questions keeps at least one good sample.
7. **Retries.**
   - Unparseable output is retried once with a new seed. Truncated output is retried once with
     twice the token budget.
   - Upstream 429/5xx/timeouts are retried with backoff (`DMW_MAX_RETRIES`).
   - The whole request is capped at `DMW_REQUEST_TIMEOUT_S`.
8. **Calibration.** Probabilities a model states about itself tend to be overconfident, so
   calibration is fitted offline and applied per model name, using Platt scaling throughout.
   - `noul` calibrates p(true).
   - `choice` and `score` calibrate the top answer's probability and rescale the other options to
     share the rest ("top-label"). The result depends only on the top probability, never on how
     many options there are, and the top answer never changes.

## Evaluation and calibration

`eval/run_eval.py` runs a labeled JSONL dataset against any Jev-compatible endpoints. It reports
accuracy, Brier score, NLL, expected calibration error (ECE), score MAE, p50/p95 latency, tokens,
cost, and top-answer agreement between endpoints. `eval/datasets/sample.jsonl` is a 24-case smoke
set (53 labeled questions) covering support routing, moderation, agent next-step and review
sentiment. Replace it with real traffic before trusting any numbers.

```bash
# middleware at k=1 vs k=3, raw probabilities
uv run python eval/run_eval.py --raw --key k1=MY_KEY --key k3=MY_KEY \
  --target k1=http://localhost:8000/v1/systemone \
  --target k3=http://localhost:8000/v1/systemone --samples k3=3

# compare with TypeSafe's Jev
uv run python eval/run_eval.py --key luna=MY_KEY --key jev=TYPESAFE_API_KEY \
  --target luna=http://localhost:8000/v1/systemone \
  --target jev=https://api.typesafe.ai/v1/systemone

# fit calibration from the raw k=1 run, then serve it
uv run python eval/fit_calibration.py --target k1 --model openai/gpt-6-luna
DMW_CALIBRATION_PATH=calibration.json uv run uvicorn --factory decisions_mw.main:create_app
```

`--key NAME=ENV_VAR` names an environment variable holding that target's bearer key.

Fitting rules:
- Fit with the same `X-Samples` you serve with.
- Refit whenever the model snapshot changes.
- Use the model name the middleware uses (`DMW_LUNA_MODEL`).
- A type with fewer than 10 examples keeps the identity.
- A fit with a negative slope falls back to the base rate instead of inverting answers.

## Configuration

All settings are environment variables (or `.env`, see `.env.example`). Invalid or unsafe
combinations stop the service at startup.

| Variable | Default | |
|---|---|---|
| `DMW_UPSTREAM_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter or any OpenAI-compatible endpoint (alias `DMW_OPENROUTER_BASE_URL`) |
| `DMW_UPSTREAM_API_KEY` | — | upstream key (alias `OPENROUTER_API_KEY`); not needed with `DMW_FORWARD_AUTH` |
| `DMW_UPSTREAM_FLAVOR` | `auto` | `openrouter` for openrouter.ai URLs, otherwise `openai`; or set explicitly |
| `DMW_LUNA_MODEL` | `openai/gpt-6-luna` | default model; calibration is keyed on it |
| `DMW_ALLOWED_MODELS` | empty | extra models a request may name explicitly |
| `DMW_REASONING_EFFORT` | `none` | Luna: `none`, `low`, `medium`, `high`, `xhigh` (not `minimal`); empty omits it |
| `DMW_API_KEYS` | empty | comma-separated inbound bearer keys |
| `DMW_FORWARD_AUTH` | `false` | forward the caller's key upstream (LiteLLM); excludes `DMW_API_KEYS` |
| `DMW_ALLOW_ANONYMOUS` | `false` | run without inbound auth (local development only) |
| `DMW_RATE_LIMIT_PER_MINUTE` | `0` (off) | per inbound key, in-process |
| `DMW_SAMPLES` / `DMW_MAX_SAMPLES` | `1` / `8` | default and cap for `X-Samples` |
| `DMW_SEED` | `0` | sample i uses seed+i; empty for unseeded |
| `DMW_TOPK_THRESHOLD` / `DMW_TOPK` | `20` / `5` | when large choices switch to a top-k list |
| `DMW_MAX_QUESTIONS_PER_CALL` | `8` | questions per upstream call |
| `DMW_MAX_QUESTIONS` / `DMW_MAX_STATE_CHARS` | `64` / `200000` | per-request limits |
| `DMW_REQUEST_TIMEOUT_S` | `30` | deadline for a whole request |
| `DMW_CALIBRATION_PATH` | unset | calibration JSON from `fit_calibration.py` |
| `DMW_TIMEOUT_S` / `DMW_MAX_RETRIES` / `DMW_MAX_CONCURRENCY` | `20` / `2` / `16` | per upstream call |
| `DMW_LOG_LEVEL` | `INFO` | one JSON log line per request on stderr |

## Development

```bash
uv run pytest                                      # live tests skip without a key
OPENROUTER_API_KEY=... uv run pytest -m live -s    # real Luna: Jev example at k=1 and k=3
DMW_UPSTREAM_BASE_URL=http://litellm:4000/v1 DMW_UPSTREAM_API_KEY=sk-... \
  DMW_LUNA_MODEL=luna-decisions uv run pytest -m live -s
uv run ruff check . && uv run ruff format --check .
```

CI (`.github/workflows/ci.yml`) runs lint, format check and tests on every push.

Layout: `src/decisions_mw/`

| File | Contents |
|---|---|
| `schemas.py` | Jev wire format |
| `openai_format.py` | OpenAI Decisions wire format and translation to and from Jev |
| `config.py` | settings and startup validation |
| `prompt.py` | aliasing and messages |
| `output_schema.py` | strict JSON schema |
| `upstream.py` | OpenRouter / OpenAI-compatible client, retries, error mapping |
| `aggregate.py` | weights to answers |
| `calibration.py` | top-label Platt calibration |
| `service.py` | fan-out, partial failures, deadline |
| `ratelimit.py` | rate limiting |
| `main.py` | HTTP (both endpoints) and auth |
| `evaluation.py` | metrics and fitting |

## Known limitations

- **Slower than the dedicated APIs.** Jev claims 70–500 ms, and OpenAI says its hosted Decisions
  API is about 10× faster than a regular Luna call (reported at around 1.6 s). This service makes
  regular structured-output calls, so expect it to be noticeably slower; measure it with
  `scripts/live_check.py`.
- **More expensive than the dedicated APIs.** Jev and OpenAI's Decisions API bill input tokens
  only. Here Luna's output tokens are billed too, roughly 3–5× the cost at k=1, and cost grows
  linearly with k.
- **Probabilities are an approximation.** They are the model's own stated weights, calibrated
  after the fact, and only as good as the calibration data. Azure's documentation contradicts
  itself on whether Luna returns `logprobs`. If your deployment does, token probabilities would be
  a better basis, but that isn't implemented.
- **Sampling may not add diversity.** Luna doesn't let us set `temperature`. If its default
  sampling is close to deterministic, `X-Samples` costs k× for little gain. Check with the k1 vs
  k3 eval.
- **Questions in one call can influence each other.** Lower `DMW_MAX_QUESTIONS_PER_CALL` to
  isolate them, at the cost of more calls.
- **Rate limiting is per process.** Behind LiteLLM, use its key limits instead.
