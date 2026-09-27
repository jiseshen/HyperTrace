import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hypertrace_mcp.__main__ import install


class InstallTest(unittest.TestCase):
    def test_registers_both_clients_with_checkout_path(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as root:
            with patch("hypertrace_mcp.__main__.Path.home", return_value=Path(root)), \
                 patch("hypertrace_mcp.__main__.Path.cwd", return_value=project.parent), \
                 patch("hypertrace_mcp.__main__.shutil.which", side_effect=lambda name: "/bin/" + name), \
                 patch("hypertrace_mcp.__main__.subprocess.run") as run:
                self.assertEqual(install(), 0)
            self.assertEqual(run.call_count, 2)
            for call in run.call_args_list:
                args = call.args[0]
                self.assertIn(str(project), args)
                self.assertEqual(args[-1], "serve")
            self.assertTrue((Path(root) / ".codex" / "skills" / "hypertrace" / "SKILL.md").is_file())
            self.assertTrue((Path(root) / ".claude" / "skills" / "hypertrace" / "SKILL.md").is_file())


if __name__ == "__main__":
    unittest.main()
