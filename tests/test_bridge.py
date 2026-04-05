from __future__ import annotations

from dataclasses import replace
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from teamai.bridge import (
    BridgeArtifacts,
    BridgeLaunchConfig,
    BridgePreflightError,
    default_bridge_artifacts,
    launch_bridge,
    load_bridge_status,
    render_bridge_script,
)


class BridgeLauncherTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_root = Path(self.temp_dir.name)
        self.artifacts = default_bridge_artifacts(self.project_root)
        self.config = BridgeLaunchConfig(
            task="Inspect this repository and identify the next engineering tasks.",
            project_root=self.project_root,
            python_executable=Path("/tmp/fake-python"),
            workspace=".",
            max_rounds=5,
            max_actions=3,
            max_tokens=192,
            temperature=0.2,
            execution_mode="read_only",
            inject_write_env=False,
            terminal_app="Terminal",
            artifacts=self.artifacts,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_render_bridge_script_contains_run_command_and_status_updates(self) -> None:
        rendered = render_bridge_script(self.config)

        self.assertIn("-m teamai run", rendered)
        self.assertIn("--output-format handoff_json", rendered)
        self.assertIn(str(self.artifacts.handoff_file), rendered)
        self.assertIn('"state": "running"', rendered)
        self.assertIn('"state": state', rendered)
        self.assertIn('"inject_write_env": false', rendered)
        self.assertIn('"bridge_run_id": "', rendered)
        self.assertIn('"memory_profile": "default"', rendered)
        self.assertIn('"retry_on_memory_pressure": true', rendered)
        self.assertIn('"retry_profile": "emergency_default"', rendered)

    def test_render_bridge_script_injects_temporary_write_env_when_requested(self) -> None:
        rendered = render_bridge_script(
            replace(self.config, execution_mode="workspace_write", inject_write_env=True, memory_profile="light_write")
        )

        self.assertIn("TEAMAI_ALLOW_WRITES=true /tmp/fake-python -u -m teamai run", rendered)
        self.assertIn('"inject_write_env": true', rendered)
        self.assertIn('"memory_profile": "light_write"', rendered)

    def test_launch_bridge_dry_run_writes_script_and_queued_status(self) -> None:
        status = launch_bridge(self.config, dry_run=True)

        self.assertEqual(status["state"], "queued")
        self.assertTrue(status["bridge_run_id"])
        self.assertTrue(self.artifacts.script_file.exists())
        self.assertTrue(self.artifacts.status_file.exists())
        self.assertEqual(status["memory_profile"], "inspection")
        self.assertTrue(status["guardrail_notes"])
        self.assertTrue(status["retry_on_memory_pressure"])
        self.assertEqual(status["retry_profile"], "emergency_inspection")

        stored_status = json.loads(self.artifacts.status_file.read_text(encoding="utf-8"))
        self.assertEqual(stored_status["task"], self.config.task)
        self.assertEqual(stored_status["handoff_file"], str(self.artifacts.handoff_file))
        self.assertEqual(stored_status["bridge_run_id"], status["bridge_run_id"])
        self.assertEqual(stored_status["memory_profile"], "inspection")
        self.assertEqual(stored_status["retry_profile"], "emergency_inspection")

    def test_launch_bridge_clears_previous_handoff_and_log_artifacts(self) -> None:
        self.artifacts.handoff_file.parent.mkdir(parents=True, exist_ok=True)
        self.artifacts.handoff_file.write_text("old handoff", encoding="utf-8")
        self.artifacts.log_file.write_text("old log", encoding="utf-8")

        status = launch_bridge(self.config, dry_run=True)

        self.assertEqual(status["state"], "queued")
        self.assertFalse(self.artifacts.handoff_file.exists())
        self.assertFalse(self.artifacts.log_file.exists())

    def test_launch_bridge_dry_run_blocks_workspace_write_when_writes_disabled(self) -> None:
        config = replace(self.config, execution_mode="workspace_write")

        self.artifacts.handoff_file.parent.mkdir(parents=True, exist_ok=True)
        self.artifacts.handoff_file.write_text("old handoff", encoding="utf-8")
        self.artifacts.log_file.write_text("old log", encoding="utf-8")

        with patch.dict("os.environ", {"TEAMAI_ALLOW_WRITES": "false"}, clear=False):
            with self.assertRaises(BridgePreflightError) as context:
                launch_bridge(config, dry_run=True)

        self.assertFalse(self.artifacts.script_file.exists())
        self.assertFalse(self.artifacts.handoff_file.exists())
        self.assertFalse(self.artifacts.log_file.exists())
        stored_status = json.loads(self.artifacts.status_file.read_text(encoding="utf-8"))
        self.assertEqual(stored_status["state"], "preflight_failed")
        self.assertIn("TEAMAI_ALLOW_WRITES", stored_status["error"])
        self.assertEqual(context.exception.payload["state"], "preflight_failed")

    def test_launch_bridge_dry_run_allows_workspace_write_when_writes_enabled(self) -> None:
        config = replace(self.config, execution_mode="workspace_write")

        with patch.dict("os.environ", {"TEAMAI_ALLOW_WRITES": "true"}, clear=False):
            status = launch_bridge(config, dry_run=True)

        self.assertEqual(status["state"], "queued")
        self.assertTrue(self.artifacts.script_file.exists())

    def test_launch_bridge_dry_run_allows_workspace_write_with_temporary_env_override(self) -> None:
        config = replace(self.config, execution_mode="workspace_write", inject_write_env=True)

        with patch.dict("os.environ", {"TEAMAI_ALLOW_WRITES": "false"}, clear=False):
            status = launch_bridge(config, dry_run=True)

        self.assertEqual(status["state"], "queued")
        self.assertEqual(status["memory_profile"], "light_write")
        self.assertTrue(self.artifacts.script_file.exists())
        rendered = self.artifacts.script_file.read_text(encoding="utf-8")
        self.assertIn("TEAMAI_ALLOW_WRITES=true /tmp/fake-python -u -m teamai run", rendered)

    def test_launch_bridge_dry_run_uses_light_recon_profile_for_broad_task(self) -> None:
        config = replace(
            self.config,
            task="Improve the local model's broad self-improvement loop and harden the bridge.",
        )

        status = launch_bridge(config, dry_run=True)

        self.assertEqual(status["state"], "queued")
        self.assertEqual(status["memory_profile"], "light_recon")
        self.assertEqual(status["retry_profile"], "emergency_recon")
        self.assertIn("--max-rounds 2", status["command"])
        self.assertIn("--max-actions 2", status["command"])
        self.assertIn("--max-tokens 128", status["command"])
        self.assertIn("--temperature 0.1", status["command"])

    def test_render_bridge_script_includes_automatic_memory_retry_path(self) -> None:
        rendered = render_bridge_script(replace(self.config, memory_profile="light_recon"))

        self.assertIn('"retry_profile": "emergency_recon"', rendered)
        self.assertIn("retrying_after_memory_pressure", rendered)
        self.assertIn("Memory pressure detected. Retrying with", rendered)

    def test_load_bridge_status_preserves_memory_pressure_payload(self) -> None:
        self.artifacts.status_file.parent.mkdir(parents=True, exist_ok=True)
        self.artifacts.status_file.write_text(
            json.dumps(
                {
                    "state": "failed",
                    "memory_pressure": True,
                    "error": "Bridge run failed with local MLX memory pressure; inspect the log file and consider an even lighter bridge profile.",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        status = load_bridge_status(self.artifacts)

        self.assertEqual(status["state"], "failed")
        self.assertTrue(status["memory_pressure"])
        self.assertIn("memory pressure", status["error"])

    def test_load_bridge_status_reports_missing_files(self) -> None:
        artifacts = BridgeArtifacts(
            handoff_file=self.project_root / ".teamai" / "handoff.json",
            status_file=self.project_root / ".teamai" / "status.json",
            log_file=self.project_root / ".teamai" / "run.log",
            script_file=self.project_root / ".teamai" / "launch.sh",
        )

        status = load_bridge_status(artifacts)

        self.assertEqual(status["state"], "missing")
        self.assertFalse(status["handoff_exists"])
        self.assertFalse(status["log_exists"])


if __name__ == "__main__":
    unittest.main()
