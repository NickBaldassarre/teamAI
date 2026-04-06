from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from teamai.config import Settings
from teamai.runtime import run_runtime_doctor, select_runtime_python


class RuntimeSelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_root = Path(self.temp_dir.name)
        self.settings = Settings(
            model_id="dummy-model",
            model_revision=None,
            force_download=False,
            trust_remote_code=False,
            enable_thinking=False,
            workspace_root=self.project_root,
            max_rounds=1,
            max_actions_per_round=1,
            max_tokens_per_turn=64,
            temperature=0.0,
            allow_shell=False,
            allow_writes=False,
            command_timeout_seconds=5,
            max_file_bytes=4096,
            max_command_output_chars=4096,
            host="127.0.0.1",
            port=8000,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_select_runtime_python_prefers_project_virtualenv(self) -> None:
        selected_python = self.project_root / ".venv" / "bin" / "python"
        selected_python.parent.mkdir(parents=True, exist_ok=True)
        selected_python.write_text("", encoding="utf-8")

        selection = select_runtime_python(
            self.project_root,
            current_python=Path("/usr/bin/python3"),
        )

        self.assertEqual(selection.source, "project_venv")
        self.assertEqual(Path(selection.selected_python).resolve(), selected_python.resolve())
        self.assertFalse(selection.using_selected_python)

    def test_select_runtime_python_honors_env_override(self) -> None:
        override = self.project_root / "custom-python"
        override.write_text("", encoding="utf-8")

        with patch.dict("os.environ", {"TEAMAI_PYTHON_EXECUTABLE": str(override)}, clear=False):
            selection = select_runtime_python(
                self.project_root,
                current_python=Path("/usr/bin/python3"),
            )

        self.assertEqual(selection.source, "env_override")
        self.assertEqual(Path(selection.selected_python).resolve(), override.resolve())

    def test_run_runtime_doctor_uses_selected_python_for_probe(self) -> None:
        selected_python = self.project_root / ".venv" / "bin" / "python"
        selected_python.parent.mkdir(parents=True, exist_ok=True)
        selected_python.write_text("", encoding="utf-8")
        captured: dict[str, object] = {}

        def subprocess_runner(
            command: list[str],
            env: dict[str, str],
            cwd: Path,
            timeout_seconds: float | None,
        ) -> subprocess.CompletedProcess[str]:
            captured["command"] = command
            captured["cwd"] = cwd
            captured["timeout_seconds"] = timeout_seconds
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(
                    {
                        "status": "healthy",
                        "reason": "mlx_generate_ok",
                        "summary": "MLX model warmup generation succeeded.",
                        "checked_at": datetime.now(timezone.utc).isoformat(),
                        "probe_mode": "generate",
                        "python_executable": str(selected_python),
                        "model_id": self.settings.model_id,
                        "model_revision": self.settings.model_revision,
                        "warnings": [],
                        "details": {"default_device": "Device(gpu, 0)"},
                    }
                ),
                stderr="",
            )

        report = run_runtime_doctor(
            settings=self.settings,
            project_root=self.project_root,
            current_python=Path("/usr/bin/python3"),
            subprocess_runner=subprocess_runner,
        )

        self.assertEqual(report.selection.source, "project_venv")
        self.assertEqual(report.probe.status, "healthy")
        self.assertEqual(Path(captured["command"][0]).resolve(), selected_python.resolve())
        self.assertEqual(Path(captured["cwd"]).resolve(), self.project_root.resolve())


if __name__ == "__main__":
    unittest.main()
