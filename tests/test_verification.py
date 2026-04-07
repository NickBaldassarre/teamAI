from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from teamai.sandbox import Sandbox
from teamai.verification import verify_patch


class VerificationTest(unittest.TestCase):
    def test_verify_patch_returns_success_when_patch_applies_and_tests_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir) / "repo"
            patch_file = self._create_repo_fixture(project_root, replacement="42")

            with Sandbox(project_root) as sandbox:
                result = verify_patch(patch_file, sandbox)

            self.assertTrue(result.success)
            self.assertEqual(result.patch_returncode, 0)
            self.assertEqual(result.test_returncode, 0)
            self.assertIn("== Patch Apply ==", result.log_output)
            self.assertIn("== Test Run ==", result.log_output)
            self.assertIn("exit_code: 0", result.log_output)

    def test_verify_patch_returns_failure_when_tests_fail_after_patch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir) / "repo"
            patch_file = self._create_repo_fixture(project_root, replacement="0")

            with Sandbox(project_root) as sandbox:
                result = verify_patch(patch_file, sandbox)

            self.assertFalse(result.success)
            self.assertEqual(result.patch_returncode, 0)
            self.assertNotEqual(result.test_returncode, 0)
            self.assertIn("FAILED", result.log_output)
            self.assertIn("exit_code:", result.log_output)

    def _create_repo_fixture(self, project_root: Path, *, replacement: str) -> Path:
        project_root.mkdir(parents=True, exist_ok=True)
        (project_root / "calc.py").write_text("def answer():\n    return 1\n", encoding="utf-8")
        tests_dir = project_root / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_calc.py").write_text(
            "\n".join(
                [
                    "from __future__ import annotations",
                    "",
                    "import unittest",
                    "",
                    "from calc import answer",
                    "",
                    "",
                    "class CalcTest(unittest.TestCase):",
                    "    def test_answer(self) -> None:",
                    "        self.assertEqual(answer(), 42)",
                    "",
                    "",
                    "if __name__ == '__main__':",
                    "    unittest.main()",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        venv_bin = project_root / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "python").symlink_to(Path(sys.executable))

        patch_file = project_root / "change.patch"
        patch_file.write_text(
            "\n".join(
                [
                    "diff --git a/calc.py b/calc.py",
                    "--- a/calc.py",
                    "+++ b/calc.py",
                    "@@ -1,2 +1,2 @@",
                    " def answer():",
                    "-    return 1",
                    f"+    return {replacement}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return patch_file


if __name__ == "__main__":
    unittest.main()
