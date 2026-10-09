"""Fit calibration parameters from raw predictions and merge them into calibration.json.

    uv run python eval/run_eval.py --target luna=http://localhost:8000/v1/systemone --raw
    uv run python eval/fit_calibration.py --target luna --model openai/gpt-6-luna

Use the same X-Samples value when collecting predictions as you serve with: averaging samples
changes how confident the outputs are. Serve the result with DMW_CALIBRATION_PATH.
"""

import argparse
import json
from pathlib import Path

from decisions_mw.evaluation import Prediction, ece, fit_calibration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--predictions", default="eval/out/predictions.jsonl")
    parser.add_argument("--target", required=True, help="target name used in run_eval.py")
    parser.add_argument("--model", required=True, help="Luna slug the parameters apply to")
    parser.add_argument("--out", default="calibration.json")
    parser.add_argument("--min-n", type=int, default=10)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.predictions).read_text().splitlines() if line]
    rows = [r for r in rows if r["target"] == args.target]
    if not rows:
        raise SystemExit(f"no predictions for target {args.target!r}")
    if any(r.get("calibration") == "on" for r in rows):
        raise SystemExit("predictions were calibrated; re-run run_eval.py with --raw")
    preds = [Prediction(r["kind"], r["probs"], r["label"]) for r in rows]

    calibration, notes = fit_calibration(preds, min_n=args.min_n)
    for note in notes:
        print(f"note: {note}")
    for kind in ("noul", "choice", "score"):
        items = [p for p in preds if p.kind == kind]
        if items:
            after = [Prediction(kind, calibration.apply(kind, p.probs), p.label) for p in items]
            print(f"{kind:<7} n={len(items):<4} ece {ece(items):.3f} -> {ece(after):.3f}")

    out = Path(args.out)
    table = json.loads(out.read_text()) if out.exists() else {}
    table[args.model] = calibration.to_dict()
    out.write_text(json.dumps(table, indent=2) + "\n")
    print(f"wrote {args.model} -> {out}: {json.dumps(calibration.to_dict())}")


if __name__ == "__main__":
    main()
