#!/usr/bin/env bash
set -e

mkdir -p LOGS
rm -f LOGS/telemetry.jsonl

echo "Running Unit Tests: Supervisor"
./.venv/bin/python -m unittest tests.test_supervisor

echo "Running Unit Tests: Evals"
./.venv/bin/python -m unittest tests.test_evals

echo "Running Smoke Eval Suite"
./.venv/bin/python -m teamai eval \
    --suite-file evals/teamai_smoke.json \
    --allow-write-cases \
    --output-format summary_markdown > LOGS/smoke_summary.md

echo "--- TELEMETRY TRACE ---" >> LOGS/eval_history.md
if [ -f LOGS/telemetry.jsonl ]; then
    cat LOGS/telemetry.jsonl >> LOGS/eval_history.md
else
    echo "No telemetry generated." >> LOGS/eval_history.md
fi
echo "--- SMOKE SUMMARY ---" >> LOGS/eval_history.md
cat LOGS/smoke_summary.md >> LOGS/eval_history.md
echo "" >> LOGS/eval_history.md

echo "Smoke evaluation complete. Appended trace to LOGS/eval_history.md"

