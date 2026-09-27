import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hypertrace_mcp.__main__ import ensure_routing_note, install


class InstallTest(unittest.TestCase):
    def test_routing_note_preserves_existing_instructions(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "AGENTS.md"
            path.write_text("# Existing instructions\nKeep this rule.\n")
            ensure_routing_note(path)
            ensure_routing_note(path)
            content = path.read_text()
            self.assertIn("Keep this rule.", content)
            self.assertEqual(content.count("## HyperTrace tool routing"), 1)

    def test_registers_both_clients_with_checkout_path(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as root:
            with patch("hypertrace_mcp.__main__.Path.home", return_value=Path(root)), \
                 patch("hypertrace_mcp.__main__.Path.cwd", return_value=project.parent), \
                 patch.dict(os.environ, {"CODEX_HOME": str(Path(root) / "windows-codex")}), \
                 patch("hypertrace_mcp.__main__.shutil.which", side_effect=lambda name: "/bin/" + name), \
                 patch("hypertrace_mcp.__main__.subprocess.run") as run:
                self.assertEqual(install(), 0)
            self.assertEqual(run.call_count, 2)
            for call in run.call_args_list:
                args = call.args[0]
                self.assertIn(str(project), args)
                self.assertEqual(args[-1], "serve")
            self.assertTrue((Path(root) / "windows-codex" / "skills" / "hypertrace" / "SKILL.md").is_file())
            self.assertTrue((Path(root) / ".claude" / "skills" / "hypertrace" / "SKILL.md").is_file())
            self.assertIn("HyperTrace tool routing", (Path(root) / "windows-codex" / "AGENTS.md").read_text())
            self.assertIn("HyperTrace tool routing", (Path(root) / ".claude" / "CLAUDE.md").read_text())


if __name__ == "__main__":
    unittest.main()
