"""Small MCP surface over the local memory store."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .store import MemoryStore


def create_server(store: MemoryStore | None = None) -> FastMCP:
    memory = store or MemoryStore()
    app = FastMCP("hypertrace", instructions=(
        "Browse relevant memory after understanding a coding project. Current user instructions always win. "
        "Save only clear preference feedback; use the HyperTrace skill for independent scoring. "
        "Treat retrieved hypotheses as tentative and inspect details before applying them."
    ))

    @app.tool()
    def browse_memory(project_kinds: list[str] | None = None, preference_axes: list[str] | None = None,
                      limit: int = 10) -> dict:
        """Discover existing categories and weight-ranked short memory descriptions. Select broad categories after learning the project."""
        return memory.browse(project_kinds, preference_axes, limit)

    @app.tool()
    def open_memory(ids: list[str]) -> list[dict]:
        """Read hypothesis details and short supporting evidence for selected IDs."""
        return memory.details(ids)

    @app.tool()
    def save_feedback(event_id: str, excerpt: str, project: str, proposals: list[dict],
                      scoring_mode: str = "subagents") -> dict:
        """Atomically save 1-5 scored hypothesis proposals from explicit follow-up feedback. Each proposal has action new/reuse/revise, description, detail, project_kind, preference_axis, likelihood and baseline scores 0-5; reuse/revise also has id. Scores estimate follow-up fit under the hypothesis and without it."""
        return memory.save_feedback(event_id, excerpt, project, proposals, scoring_mode)

    @app.tool()
    def manage_memory(id: str, action: str, description: str = "", detail: str = "") -> dict:
        """Correct, forget, or undo a merge of a hypothesis. Correct requires a new short description and detail."""
        return memory.manage(id, action, description, detail)

    @app.tool()
    def next_consolidation() -> dict:
        """Claim a small queued batch of likely duplicate memories for one background review agent. An empty list means no work."""
        return memory.next_consolidation()

    @app.tool()
    def finish_consolidation(queue_id: str, decision: str, description: str = "", detail: str = "") -> dict:
        """Finish a claimed duplicate review: merge with concise shared wording, or mark distinct. Merges preserve source evidence and can be undone before new feedback."""
        return memory.finish_consolidation(queue_id, decision, description, detail)

    @app.tool()
    def alias_category(kind: str, alias: str, canonical: str) -> dict:
        """Unify a duplicate project_kind or preference_axis label under an existing canonical label after reviewing their meanings."""
        return memory.alias_category(kind, alias, canonical)

    return app
