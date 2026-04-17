from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from teamai import daemon


class DaemonTest(unittest.TestCase):
    def test_build_plist_inherits_xai_and_grok_keys(self) -> None:
        with patch.dict(
            os.environ,
            {
                "XAI_API_KEY": "xai-test-key",
                "GROK_API_KEY": "grok-test-key",
                "PATH": "/usr/bin",
            },
            clear=False,
        ):
            payload = daemon._build_plist(workspace="/tmp/demo")  # noqa: SLF001

        env_vars = payload["EnvironmentVariables"]
        self.assertEqual(env_vars["XAI_API_KEY"], "xai-test-key")
        self.assertEqual(env_vars["GROK_API_KEY"], "grok-test-key")

    def test_stop_daemon_reports_timeout_without_removing_pid_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pid_file = Path(temp_dir) / "daemon.pid"
            pid_file.write_text("4242", encoding="utf-8")

            with patch.object(daemon, "_pid_path", return_value=pid_file):
                with patch.object(daemon, "_read_pid", return_value=4242):
                    with patch.object(daemon, "_pid_is_alive", return_value=True):
                        with patch("teamai.daemon.os.kill") as mock_kill:
                            with patch("teamai.daemon.time.sleep"):
                                with patch(
                                    "teamai.daemon.time.monotonic",
                                    side_effect=[0.0, 0.0, 5.0],
                                ):
                                    result = daemon.stop_daemon()

            mock_kill.assert_called_once_with(4242, daemon.signal.SIGTERM)
            self.assertEqual(result["status"], "stop_timeout")
            self.assertEqual(result["pid"], 4242)
            self.assertTrue(pid_file.exists())


if __name__ == "__main__":
    unittest.main()
