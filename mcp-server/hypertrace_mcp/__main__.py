"""Run the server or register it with installed coding agents."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROUTING_NOTE = """## HyperTrace tool routing

For coding work, use the `hypertrace` skill after learning a new project's context and when a follow-up explicitly signals a reusable preference, correction, or coding habit. Let its evidence gate decide whether to save anything. Current user instructions take precedence over memory; do not copy existing agent instructions into memory.
"""


def ensure_routing_note(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text() if path.exists() else ""
    if "## HyperTrace tool routing" in existing:
        return
    with path.open("a", encoding="utf-8") as output:
        if existing:
            output.write("\n" if existing.endswith("\n") else "\n\n")
        output.write(ROUTING_NOTE)


def install() -> int:
    uv = shutil.which("uv")
    if not uv:
        print("Install uv first: https://docs.astral.sh/uv/", file=sys.stderr)
        return 1
    cwd = Path.cwd().resolve()
    project = cwd / "mcp-server" if (cwd / "mcp-server" / "pyproject.toml").is_file() else cwd
    if not (project / "pyproject.toml").is_file() or project.name != "mcp-server":
        print("Run install from the HyperTrace repository root or its mcp-server directory.", file=sys.stderr)
        return 1
    launch = [uv, "run", "--project", str(project), "hypertrace-mcp", "serve"]
    cache = str(Path(tempfile.gettempdir()) / "hypertrace-uv-cache")
    bundled = Path(__file__).resolve().parent / "skill" / "SKILL.md"
    codex_home = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
    found = 0
    registered = 0
    for host, root, entry, command in (
        ("Codex", codex_home / "skills", codex_home / "AGENTS.md", ["codex", "mcp", "add", "hypertrace", "--env", f"UV_CACHE_DIR={cache}", "--", *launch]),
        ("Claude", Path.home() / ".claude" / "skills", Path.home() / ".claude" / "CLAUDE.md", ["claude", "mcp", "add", "--scope", "user", "--env", f"UV_CACHE_DIR={cache}", "--transport", "stdio", "hypertrace", "--", *launch]),
    ):
        executable = shutil.which(command[0])
        if not executable:
            print(f"{host}: CLI not found; skipped")
            continue
        found += 1
        target = root / "hypertrace" / "SKILL.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(bundled, target)
        try:
            subprocess.run([executable, *command[1:]], check=True)
        except subprocess.CalledProcessError as exc:
            print(f"{host}: MCP registration failed ({exc}); skill installed at {target}", file=sys.stderr)
            continue
        registered += 1
        ensure_routing_note(entry)
        print(f"{host}: registered MCP and installed skill at {target}; enabled routing in {entry}")
    if not found:
        print("Neither Codex nor Claude CLI was found. Install one, then rerun this command.", file=sys.stderr)
        return 1
    return 0 if registered == found else 1


def main() -> None:
    parser = argparse.ArgumentParser(prog="hypertrace-mcp")
    parser.add_argument("command", choices=["serve", "install"])
    args = parser.parse_args()
    if args.command == "install":
        raise SystemExit(install())
    from .server import create_server
    create_server().run(transport="stdio")


if __name__ == "__main__":
    main()
