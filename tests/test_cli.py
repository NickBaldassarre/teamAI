from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from datetime import datetime, timezone
from pathlib import Path

from teamai.cli import _build_run_stream_handlers
from teamai.schemas import RunEvent


class CLIStreamingTest(unittest.TestCase):
    def test_stream_handlers_write_jsonl_to_stderr_and_log_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            stderr = io.StringIO()
            progress_callback, event_callback, close_stream = _build_run_stream_handlers(
                project_root=project_root,
                stream_format="jsonl",
                event_log_file="events.jsonl",
            )
            event = RunEvent(
                sequence=1,
                timestamp=datetime.now(timezone.utc),
                kind="round_stage",
                message="Round 1/2: planner",
                round_number=1,
                stage="planner",
                data={"total_rounds": 2},
            )
            try:
                with redirect_stderr(stderr):
                    assert progress_callback is not None
                    progress_callback("Round 1/2: planner")
                    assert event_callback is not None
                    event_callback(event)
            finally:
                close_stream()

            stderr_payload = stderr.getvalue().strip().splitlines()
            self.assertEqual(len(stderr_payload), 1)
            self.assertEqual(json.loads(stderr_payload[0])["kind"], "round_stage")

            log_lines = (project_root / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(log_lines), 1)
            self.assertEqual(json.loads(log_lines[0])["stage"], "planner")


if __name__ == "__main__":
    unittest.main()
