"""Run a labeled dataset against Jev-compatible endpoints and report decision quality.

Any endpoint speaking Jev's `POST /v1/systemone` works, so this middleware and real Jev can be
compared side by side:

    # this middleware, raw probabilities (input for fit_calibration.py)
    uv run python eval/run_eval.py --target luna=http://localhost:8000/v1/systemone --raw

    # one sample vs. three averaged samples
    uv run python eval/run_eval.py --target k1=http://localhost:8000/v1/systemone \
        --target k3=http://localhost:8000/v1/systemone --samples k3=3

    # middleware vs. TypeSafe's Jev
    uv run python eval/run_eval.py \
        --target luna=http://localhost:8000/v1/systemone \
        --target jev=https://api.typesafe.ai/v1/systemone --key jev=TYPESAFE_API_KEY

Writes <out>/predictions.jsonl and <out>/report.json and prints a summary.
"""

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import httpx

from decisions_mw.evaluation import Prediction, summarize


def pairs(values: list[str] | None, flag: str) -> dict[str, str]:
    out = {}
    for value in values or []:
        name, sep, rest = value.partition("=")
        if not sep:
            raise SystemExit(f"{flag} expects NAME=VALUE, got {value!r}")
        out[name] = rest
    return out


def to_prediction(question: dict[str, Any], label: Any, answer: dict[str, Any]) -> Prediction:
    kind = question["type"]
    if kind == "noul":
        p = float(answer["noul"])
        return Prediction("noul", [1 - p, p], int(bool(label)))
    if kind == "choice":
        options = list(question["criteria"])
        probs = [float(answer["probabilities"].get(o, 0.0)) for o in options]
        return Prediction("choice", probs, options.index(label))
    n = len(question["criteria"])
    probs = [float(answer["probabilities"].get(str(i), 0.0)) for i in range(n)]
    return Prediction("score", probs, int(label))


def percentile(values: list[float], q: int) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[q - 1]


async def run_case(
    http: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    target: str,
    url: str,
    headers: dict[str, str],
    model: str,
    case: dict[str, Any],
) -> dict[str, Any]:
    body = {"model": model, "state": case["state"], "questions": case["questions"]}
    record: dict[str, Any] = {"target": target, "case": case["id"]}
    async with sem:
        start = time.perf_counter()
        try:
            resp = await http.post(url, json=body, headers=headers)
        except httpx.HTTPError as e:
            record.update(error=f"{type(e).__name__}: {e}")
            return record
        record["latency_ms"] = (time.perf_counter() - start) * 1000
    record["status"] = resp.status_code
    record["calibration"] = resp.headers.get("x-calibration")
    cost = resp.headers.get("x-cost-usd")
    record["cost_usd"] = float(cost) if cost else None
    if resp.status_code != 200:
        record["error"] = resp.text[:500]
        return record
    data = resp.json()
    record["usage"] = data.get("usage")
    record["answers"] = data.get("answers", {})
    return record


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", default="eval/datasets/sample.jsonl")
    parser.add_argument("--target", action="append", required=True, help="NAME=URL")
    parser.add_argument("--key", action="append", help="NAME=ENV_VAR holding a bearer key")
    parser.add_argument("--model", action="append", help="NAME=MODEL (default jev-latest)")
    parser.add_argument("--samples", action="append", help="NAME=K, X-Samples for that target")
    parser.add_argument("--raw", action="store_true", help="X-Calibration: off")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--out", default="eval/out")
    args = parser.parse_args()

    targets = pairs(args.target, "--target")
    keys = pairs(args.key, "--key")
    models = pairs(args.model, "--model")
    samples = pairs(args.samples, "--samples")
    cases = [json.loads(line) for line in Path(args.dataset).read_text().splitlines() if line]
    by_id = {c["id"]: c for c in cases}

    sem = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(timeout=args.timeout) as http:
        jobs = []
        for name, url in targets.items():
            headers = {}
            if name in keys:
                headers["Authorization"] = f"Bearer {os.environ[keys[name]]}"
            if name in samples:
                headers["X-Samples"] = samples[name]
            if args.raw:
                headers["X-Calibration"] = "off"
            model = models.get(name, "jev-latest")
            jobs += [run_case(http, sem, name, url, headers, model, c) for c in cases]
        records = await asyncio.gather(*jobs)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    preds: dict[str, dict[tuple[str, str], Prediction]] = defaultdict(dict)
    with (out / "predictions.jsonl").open("w") as f:
        for r in records:
            case = by_id[r["case"]]
            for qid, label in case["labels"].items():
                answer = r.get("answers", {}).get(qid)
                if answer is None:
                    continue
                p = to_prediction(case["questions"][qid], label, answer)
                preds[r["target"]][(r["case"], qid)] = p
                line = {
                    "target": r["target"],
                    "case": r["case"],
                    "qid": qid,
                    "kind": p.kind,
                    "probs": p.probs,
                    "label": p.label,
                    "calibration": r.get("calibration"),
                }
                f.write(json.dumps(line) + "\n")

    report: dict[str, Any] = {"dataset": args.dataset, "targets": {}, "agreement": {}}
    for name in targets:
        rs = [r for r in records if r["target"] == name]
        latencies = [r["latency_ms"] for r in rs if "latency_ms" in r and "error" not in r]
        costs = [r.get("cost_usd") for r in rs if "error" not in r]
        target_preds = list(preds[name].values())
        report["targets"][name] = {
            "requests": len(rs),
            "errors": [{"case": r["case"], "error": r["error"]} for r in rs if "error" in r],
            "latency_ms": {"p50": percentile(latencies, 50), "p95": percentile(latencies, 95)},
            "cost_usd": sum(costs) if costs and all(c is not None for c in costs) else None,
            "input_tokens": sum((r.get("usage") or {}).get("input_tokens", 0) for r in rs),
            "output_tokens": sum((r.get("usage") or {}).get("output_tokens", 0) for r in rs),
            "by_type": {
                kind: summarize([p for p in target_preds if p.kind == kind])
                for kind in ("noul", "choice", "score")
            },
        }
    for a, b in combinations(targets, 2):
        shared = preds[a].keys() & preds[b].keys()
        if shared:
            same = sum(
                max(range(len(preds[a][k].probs)), key=preds[a][k].probs.__getitem__)
                == max(range(len(preds[b][k].probs)), key=preds[b][k].probs.__getitem__)
                for k in shared
            )
            report["agreement"][f"{a}~{b}"] = {"n": len(shared), "top_answer": same / len(shared)}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print_report(report)


def fmt(value: Any, spec: str = ".3f") -> str:
    return "-" if value is None else format(value, spec)


def print_report(report: dict[str, Any]) -> None:
    for name, t in report["targets"].items():
        lat = t["latency_ms"]
        print(
            f"\n== {name}: {t['requests']} requests, {len(t['errors'])} errors, "
            f"latency p50 {fmt(lat['p50'], '.0f')} ms / p95 {fmt(lat['p95'], '.0f')} ms, "
            f"cost ${fmt(t['cost_usd'], '.5f')}, tokens {t['input_tokens']} in / "
            f"{t['output_tokens']} out"
        )
        print(f"   {'type':<7}{'n':>4}{'acc':>8}{'brier':>8}{'nll':>8}{'ece':>8}{'mae':>8}")
        for kind, m in t["by_type"].items():
            if m.get("n"):
                print(
                    f"   {kind:<7}{m['n']:>4}{fmt(m['accuracy']):>8}{fmt(m['brier']):>8}"
                    f"{fmt(m['nll']):>8}{fmt(m['ece']):>8}{fmt(m.get('mae')):>8}"
                )
        for err in t["errors"][:5]:
            print(f"   ! {err['case']}: {err['error'][:160]}", file=sys.stderr)
    for pair, a in report["agreement"].items():
        print(f"\nagreement {pair}: {a['top_answer']:.3f} on {a['n']} questions")


if __name__ == "__main__":
    asyncio.run(main())
