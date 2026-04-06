#!/bin/bash
export TEAMAI_TELEMETRY=1
mkdir -p LOGS
> LOGS/telemetry.jsonl
echo "# Smoke run $(date)" >> LOGS/eval_history.md

./.venv/bin/python -m unittest tests.test_supervisor
./.venv/bin/python -m unittest tests.test_evals
./.venv/bin/python -m teamai eval --suite-file evals/teamai_smoke.json --allow-write-cases --output-format summary_markdown

# Append Telemetry Snapshot
echo -e "\n## Telemetry Snapshot\n\`\`\`json" >> LOGS/eval_history.md
head -20 LOGS/telemetry.jsonl >> LOGS/eval_history.md 2>/dev/null || echo "No telemetry frames" >> LOGS/eval_history.md
echo -e "\`\`\`" >> LOGS/eval_history.md
