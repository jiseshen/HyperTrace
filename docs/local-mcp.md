# Local MCP prototype

## Goal and current decisions

The prototype makes HyperTrace useful during ordinary coding work. It stores tentative hypotheses about user habits across projects, using short verbal feedback rather than requiring chosen and rejected candidate responses. The research pipeline remains the source of the hypothesis-based design; the MCP has a separate live interaction path because a follow-up has no candidate-response comparison.

Feedback scoring uses the current host's small subagents, one per hypothesis. Each returns an expectedness score for the observed follow-up with and without that hypothesis. The server applies a bounded evidence update and keeps new hypotheses provisional. Scores are ranking signals, not calibrated probabilities. If subagents are unavailable, the main agent can score serially with reduced weight.

Each hypothesis has a short description, actionable detail, weight, project-kind category, and preference-axis category. Both category sets can grow. The server reuses normalized labels and accepts reviewed aliases. Retrieval first shows categories, then returns short entries ranked by weight within selected categories; it uses no embeddings. Current user instructions always override stored beliefs.

## Install and use

Requires Python 3.11 or later, [uv](https://docs.astral.sh/uv/), and the Claude Code or Codex CLI. From the repository root:

```bash
uv run --project mcp-server hypertrace-mcp install
```

The installer copies the same `hypertrace` skill to each detected client's user skill directory and registers a stdio server pointing at this checkout. It does not overwrite the user's agent instruction files. Restart the clients after installation. The checkout must remain at the same path; rerun installation if it moves. To check registration, use `codex mcp list` or `claude mcp list`. The server command can also run directly with `uv run --project mcp-server hypertrace-mcp serve`.

Codex and Claude share one database on the **same operating system**. Linux uses `${XDG_DATA_HOME:-~/.local/share}/hypertrace/memory.sqlite3`; Windows uses `%LOCALAPPDATA%\hypertrace\memory.sqlite3`. `HYPERTRACE_DB` can override the path for testing or an intentionally shared location. Windows and WSL are separate environments unless pointed to the same accessible file; SQLite on a network share is not supported by this prototype.

At a new project, the skill asks the agent to understand the work, discover categories, inspect a few promising hypotheses, and briefly say which principles it will use. On explicit feedback, it gathers compact pre-feedback context and independent scores, then saves one event. The MCP accepts a feedback excerpt of at most 300 characters and does not store full conversations. It rejects common credential patterns, but the agent must still avoid supplying secrets. There is no automatic transcript capture.

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

Run the focused checks with `uv run --project mcp-server python -m unittest discover -s mcp-server/tests -v`. They cover weight changes, duplicate events, category aliases, forgetting, merge recovery, and a real stdio MCP handshake. Windows and Claude/Codex interaction still need hands-on checks on those hosts. The next iteration should use real follow-ups to inspect category drift, ranking usefulness, acknowledgment quality, and scoring cost before adding custom processing models.
