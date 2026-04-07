from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from teamai.sandbox import Sandbox


class SandboxTest(unittest.TestCase):
    def test_sandbox_run_isolates_file_changes_from_source_repo(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir) / "repo"
            project_root.mkdir()
            source_file = project_root / "tracked.txt"
            source_file.write_text("original\n", encoding="utf-8")

            with Sandbox(project_root) as sandbox:
                sandbox_path = sandbox.path
                result = sandbox.run("printf 'sandbox\\n' > tracked.txt")

                self.assertEqual(result.returncode, 0)
                self.assertEqual(source_file.read_text(encoding="utf-8"), "original\n")
                self.assertEqual((sandbox_path / "tracked.txt").read_text(encoding="utf-8"), "sandbox\n")

            self.assertFalse(sandbox_path.exists())

    def test_sandbox_run_executes_within_isolated_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir) / "repo"
            project_root.mkdir()
            (project_root / "tracked.txt").write_text("original\n", encoding="utf-8")

            with Sandbox(project_root) as sandbox:
                sandbox_path = sandbox.path
                result = sandbox.run(["/bin/pwd"])

            self.assertEqual(result.returncode, 0)
            self.assertEqual(Path(result.stdout.strip()).resolve(), sandbox_path.resolve())
            self.assertFalse(sandbox_path.exists())


if __name__ == "__main__":
    unittest.main()
