"""Run the server or register it with installed coding agents."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


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
    bundled = Path(__file__).resolve().parent / "skill" / "SKILL.md"
    found = 0
    registered = 0
    for host, root, command in (
        ("Codex", Path.home() / ".codex" / "skills", ["codex", "mcp", "add", "hypertrace", "--", *launch]),
        ("Claude", Path.home() / ".claude" / "skills", ["claude", "mcp", "add", "--scope", "user", "--transport", "stdio", "hypertrace", "--", *launch]),
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
        print(f"{host}: registered MCP and installed skill at {target}")
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
