---
name: hypertrace
description: Use HyperTrace local memory during coding projects to retrieve relevant user habits, and learn from explicit follow-up feedback or requirements that signal a reusable preference. Do not use for greetings or routine task details with no preference signal.
---

# HyperTrace

The MCP stores tentative, weighted hypotheses. The current user's instructions take precedence. Keep the visible task moving; memory work must not delay a response just to improve a score.

## Retrieve

After learning enough about a new project or a changed task, call `browse_memory` without filters to see project-kind and preference-axis categories. Choose broad relevant labels, including a general project kind when present, then browse again with those labels. Read details with `open_memory` for the few entries that might change your work. Use only applicable beliefs. Briefly tell the user the main principles you are applying, in ordinary language; do not claim certainty or recite scores. Do not mention irrelevant memories.

## Learn from verbal feedback

Consider each explicit correction, follow-up requirement, or statement of a coding habit. Skip simple acknowledgments, one-off facts, and requirements that reveal nothing reusable. Avoid saving secrets, sensitive personal facts, or text the user asked you to forget. Keep only a short excerpt of the feedback.

Browse related memories before proposing 1-5 distinct explanations. Prefer `reuse` or a narrow `revise` when an existing hypothesis expresses the same habit. Use `new` for a genuinely different explanation. Make every description a short, useful retrieval cue; put the actionable meaning and limits in `detail`. Assign project-kind and preference-axis labels, reusing existing labels where they fit. A project-specific requirement may be remembered with its project-kind label, but do not broaden it into a universal rule without evidence.

For each proposed hypothesis, ask one small host-native subagent to score it, using the fastest suitable host model when available. Give each worker only compact **pre-feedback** project context, the observed follow-up, and that one hypothesis. Ask: “If this hypothesis were true, how expected is this follow-up? How expected would it be without this hypothesis? Return `likelihood` and `baseline` from 0 to 5, with one short reason. Judge the underlying correction or choice, not exact wording. Keep scores close when the evidence is ambiguous. Do not propose edits or call memory tools.” These are relative evidence scores, not probabilities. New proposals use the observed event for discovery, so score conservatively; one event must not establish a firm habit.

Collect all worker outputs before one `save_feedback` call. Discard new proposals unless their `likelihood` exceeds `baseline` by at least two points; a plausible but indirect implication is not enough to create memory. Each remaining proposal needs `action`, `description`, `detail`, `project_kind`, `preference_axis`, `likelihood`, and `baseline`; `reuse` and `revise` also need `id`. Use a stable unique `event_id` for the follow-up so a retry cannot save it twice, a short `excerpt`, and a concise `project` label. Set `scoring_mode` to `subagents`. If the first subagent launch fails, do not retry other launches for this event: score serially yourself using the same rubric and set `scoring_mode` to `main_agent`. Do not hide failed scoring with invented values.

## Consolidate when convenient

If a save reports queued consolidation and the host can run a background subagent, launch **one** small review worker while you continue useful user work. It calls `next_consolidation`, reviews each claimed pair against its evidence, and calls `finish_consolidation` with `merge` only if the claims genuinely mean the same thing; otherwise `distinct`. A merge's description and detail must preserve the shared meaning and limits. If a background agent is unavailable, leave the queue for another opportunity. Never delay the current response to clear it.

Use `manage_memory` when the user asks to correct or forget a hypothesis. `alias_category` may unify labels only after confirming they mean the same thing.
