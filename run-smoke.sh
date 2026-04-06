#!/bin/bash
export TEAMAI_TELEMETRY=1
mkdir -p LOGS
> LOGS/telemetry.jsonl
echo "# Smoke run $(date)" >> LOGS/eval_history.md

./.venv/bin/python -m unittest tests.test_supervisor
./.venv/bin/python -m unittest tests.test_evals
./.venv/bin/python -m teamai eval --suite-file evals/teamai_smoke.json --allow-write-cases --output-format summary_markdown
