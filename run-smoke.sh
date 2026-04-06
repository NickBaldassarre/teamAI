#!/bin/bash
set -e
mkdir -p LOGS
> LOGS/telemetry.jsonl
echo "# Smoke run $(date)" >> LOGS/eval_history.md

./.venv/bin/python -m unittest tests.test_supervisor
./.venv/bin/python -m unittest tests.test_evals

export TEAMAI_TELEMETRY=1
export HF_HOME="$PWD/.teamai/hf_cache"
export TEAMAI_MODEL_ID="mlx-community/gemma-4-2b-it-4bit"

echo "=> Running teamai doctor preflight..."
if ! ./.venv/bin/python -m teamai doctor --probe-mode generate; then
    echo "=> local runtime unhealthy or doctor failed. See output above."
    exit 1
fi

mkdir -p .teamai/evals
echo "=> Running terminal_bridge live eval..."
./.venv/bin/python -m teamai eval \
  --suite-file evals/teamai_smoke.json \
  --allow-write-cases \
  --runner-mode terminal_bridge \
  --output-format full_json \
  --output-file .teamai/evals/live-smoke-report.json

# Append Telemetry Snapshot
echo -e "\n## Telemetry Snapshot\n\`\`\`json" >> LOGS/eval_history.md
head -20 LOGS/telemetry.jsonl >> LOGS/eval_history.md 2>/dev/null || echo "No telemetry frames" >> LOGS/eval_history.md
echo -e "\`\`\`" >> LOGS/eval_history.md
