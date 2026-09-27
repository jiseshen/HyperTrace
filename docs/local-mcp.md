# Local MCP prototype

## Goal and current decisions

The prototype makes HyperTrace useful during ordinary coding work. It stores tentative hypotheses about user habits across projects, using short verbal feedback rather than requiring chosen and rejected candidate responses. The research pipeline remains the source of the hypothesis-based design; the MCP has a separate live interaction path because a follow-up has no candidate-response comparison.

Feedback scoring uses the current host's small subagents, one per hypothesis. Each returns an expectedness score for the observed follow-up with and without that hypothesis. A new hypothesis needs a clear two-point advantage over its baseline; the server then applies a bounded evidence update and keeps it provisional. Scores are ranking signals, not calibrated probabilities. If subagents are unavailable, the main agent can score serially with reduced weight.

Each hypothesis has a short description, actionable detail, weight, project-kind category, and preference-axis category. Both category sets can grow. The server reuses normalized labels and accepts reviewed aliases. One unfiltered browse returns both categories and top ranked short entries. The agent filters categories only when that first page misses relevant entries, and opens details only when they could change the work. This keeps the common retrieval path to one MCP call; it uses no embeddings. Current user instructions always override stored beliefs.

## Install and use

Requires Python 3.11 or later, [uv](https://docs.astral.sh/uv/), and the Claude Code or Codex CLI. From the repository root:

```bash
uv run --project mcp-server hypertrace-mcp install
```

The installer copies the same `hypertrace` skill to each detected client's user skill directory, registers a stdio server pointing at this checkout, and appends one short routing note to Codex `AGENTS.md` or Claude `CLAUDE.md` if absent. It preserves existing instructions. Restart the clients after installation. The checkout must remain at the same path; rerun installation if it moves. To check registration, use `codex mcp list` or `claude mcp list`. The server command can also run directly with `uv run --project mcp-server hypertrace-mcp serve`.

For a Windows Codex installation whose `CODEX_HOME` is on `/mnt/c/...` but whose server runs in WSL, set `CODEX_HOME` to that path when running the installer. It will put the skill there and register the Linux `uv` path in that home's `config.toml`, with a writable temporary `UV_CACHE_DIR`. The desktop app needs a restart to discover a newly added MCP server or skill; an already-running task may retain its old tool list.

Codex and Claude share one database on the **same operating system**. Linux uses `${XDG_DATA_HOME:-~/.local/share}/hypertrace/memory.sqlite3`; Windows uses `%LOCALAPPDATA%\hypertrace\memory.sqlite3`. `HYPERTRACE_DB` can override the path for testing or an intentionally shared location. Windows and WSL are separate environments unless pointed to the same accessible file; SQLite on a network share is not supported by this prototype.

At a new project, the skill asks the agent to understand the work and discover categories. It opens promising hypotheses only when details could change the next action, and briefly names a principle only when retrieved memory is actually applied. On explicit feedback, it gathers compact pre-feedback context and independent scores, then saves one event. The MCP accepts a feedback excerpt of at most 300 characters and does not store full conversations. It rejects common credential patterns, but the agent must still avoid supplying secrets. There is no automatic transcript capture.

The server queues likely duplicate hypotheses using inexpensive word overlap inside the same categories. A host background agent may review a small queue when convenient; normal responses do not wait for it. Source records and evidence remain available after a merge. A merge can be undone until new feedback is attached to its merged record. Users can ask the agent to inspect, correct, or forget a memory.

## MCP tools

| Tool | Purpose |
| --- | --- |
| `browse_memory` | Discover categories and weight-ranked short descriptions, optionally within selected categories. |
| `open_memory` | Inspect selected hypotheses and recent short evidence. |
| `save_feedback` | Atomically save one feedback event with 1–5 scored proposals. |
| `manage_memory` | Correct, forget, or undo a merge. |
| `next_consolidation`, `finish_consolidation` | Claim and review small background duplicate batches. |
| `alias_category` | Reconcile confirmed equivalent category labels. |

The skill is the workflow layer; the server validates inputs and owns persistence. A repeated `event_id` is ignored. Errors remain visible to the agent. SQLite writes are transactional across concurrent client processes.

## Verification and next iteration

Run the focused checks with `uv run --project mcp-server python -m unittest discover -s mcp-server/tests -v`. They cover weight changes, duplicate events, category aliases, forgetting, merge recovery, and a real stdio MCP handshake. A WSL-backed Windows Codex CLI session has also discovered the installed skill and MCP and completed an isolated follow-up with subagent scoring and memory retrieval. Native Windows execution, Claude integration, and discovery in a restarted Codex desktop task still need hands-on checks. The next iteration should use real follow-ups to inspect category drift, ranking usefulness, acknowledgment quality, and scoring cost before adding custom processing models.
