#!/usr/bin/env bash
# Proof-of-concept comparison: this middleware (Luna) vs. Jev on OpenRouter, on a synthetic set.
#
#   scripts/compare_luna_jev.sh            # 150 cases
#   N=300 scripts/compare_luna_jev.sh      # bigger set
#
# Needs OPENROUTER_API_KEY and DMW_API_KEYS in .env. Starts the middleware on :8000 if it isn't
# already running, and stops it again afterwards. Results land in eval/out/compare/.
set -euo pipefail
cd "$(dirname "$0")/.."

N="${N:-150}"
PORT="${PORT:-8000}"
DATASET="eval/datasets/synthetic.jsonl"
OUT="eval/out/compare"

set -a; . ./.env; set +a
export DMW_KEY="${DMW_API_KEYS%%,*}"
[ -n "$DMW_KEY" ] || { echo "DMW_API_KEYS is empty in .env" >&2; exit 1; }
[ -n "${OPENROUTER_API_KEY:-}" ] || { echo "OPENROUTER_API_KEY is empty in .env" >&2; exit 1; }

uv run python eval/make_synthetic.py --n "$N" --out "$DATASET"

SERVER_PID=""
if ! curl -sf "localhost:$PORT/healthz" >/dev/null; then
  echo "starting middleware on :$PORT"
  mkdir -p "$OUT"
  uv run uvicorn --factory decisions_mw.main:create_app --port "$PORT" >"$OUT/server.log" 2>&1 &
  SERVER_PID=$!
  trap '[ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null' EXIT
  for _ in $(seq 30); do curl -sf "localhost:$PORT/healthz" >/dev/null && break; sleep 1; done
  curl -sf "localhost:$PORT/healthz" >/dev/null || { echo "server failed, see $OUT/server.log" >&2; exit 1; }
fi

uv run python eval/run_eval.py --dataset "$DATASET" --out "$OUT" \
  --key luna=DMW_KEY --key jev=OPENROUTER_API_KEY \
  --target luna="http://localhost:$PORT/v1/systemone" \
  --target jev=https://openrouter.ai/api/v1/systemone
