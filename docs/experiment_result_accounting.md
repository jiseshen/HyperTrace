# Experiment Result Accounting

This note documents the current result aggregation policy for the paper figures,
CSVs, and LaTeX tables generated from `visualize_experiments.py`. It is intended
for future Codex sessions or collaborators who need to update experiment results
without silently changing the statistical protocol.

## Entry Point

Use:

```bash
python3 visualize_experiments.py prism-baselines --no-profile-eval
python3 visualize_experiments.py prism-claude-eval --no-profile-eval
python3 visualize_experiments.py personamem-baselines --no-profile-eval
python3 visualize_experiments.py model-ablation --no-profile-eval
python3 visualize_experiments.py prism-ablation --no-profile-eval
```

Use `--refresh-profile-eval` only when intentionally recomputing PersonaMem-v2
profile scores with the current rubric. Do not use it casually while only
restyling plots or tables.

## Shared Online Metric Protocol

Online metrics are computed by `summary_rows()` from `visualize_experiments.py`.
For each metric:

- Build a turn-wise series using only turns with at least `MIN_USERS_PER_TURN`
  users. The current value is `10`.
- Smooth the turn-wise series with the configured rolling window.
- `*_after20` is the mean over all smoothed turn points with `turn >= 20`.
  It is not just the value at turn 20.
- `*_delta` subtracts the smoothed turn-0 value from that same after-20 mean.
- The plotted main curves and the CSV/table use the same assembled run objects.

The three main online metrics are:

- `prediction_accuracy`: preference prediction accuracy.
- `adapt_relative_gpt_score`: relative GPT judge score for response alignment.
- `adapt_relative_score`: relative embedding score for response alignment.

Hydra only has prediction accuracy in the main comparison; its response
alignment fields are left blank.

## PRISM Main Comparison

Output directory:

- `result/plots/prism/`

Main files:

- `baselines.csv`
- `baselines_online_metrics_main.pdf`
- `baselines_profile*.pdf`

Current cohort:

- `prism_legacy_user_ids()`, which is based on
  `result/openrouter-gpt5-trace-gemini3-flash-eval-legacy` when that directory
  exists.
- Current expected size: 28 users.

Source policy:

- `PT (ours)`:
  - Accuracy, GPT response score, and profile score come from
    `result/openrouter-gpt5-trace-gemini3-flash-eval`.
  - Embedding response score comes from
    `result/openrouter-gpt5-trace-gemini3-flash-eval-legacy` when available.
  - PT legacy embedding turn 0 is offset by
    `LEGACY_EMBEDDING_TURN0_OFFSET = -0.04`.
  - This offset only changes turn 0 and therefore affects `DeltaEmb`; it does
    not change `Emb>20`.
- Baselines:
  - Prefer `baseline_results_legacy/<baseline-name>` if it exists.
  - Fall back to `baseline_results/<baseline-name>` only if the legacy directory
    is missing or empty.
  - Do not apply the PT embedding overlay to baselines.

Important current baseline behavior:

- Cheatsheet uses `baseline_results_legacy/cheatsheet_gpt5_openrouter`.
- HyperAlign uses `baseline_results/hypogenic_gpt5_openrouter`, because there
  is no `baseline_results_legacy/hypogenic_gpt5_openrouter`.

## PRISM Claude-Eval Comparison

Output directory:

- `result/plots/prism_claude_eval/`

Main files:

- `baselines_claude_eval.csv`
- `baselines_claude_eval_online_metrics_main.pdf`
- `profile_alignment*_claude_eval.pdf`

Current cohort:

- Same 28-user PRISM cohort as the PRISM main comparison.

Source policy:

- `PT (ours)`:
  - Accuracy and GPT response score come from
    `result/openrouter-gpt5-trace-gemini3-flash-eval/metrics_claude_sonnet_46_loose`.
  - Embedding response score comes from the same legacy PT embedding source as
    the PRISM main comparison.
  - The same PT turn-0 offset is applied.
- Baselines:
  - First try
    `baseline_results_legacy/<baseline-name>/metrics_claude_sonnet_46_loose`.
  - If that Claude metrics directory is missing or empty, fall back to
    `baseline_results/<baseline-name>/metrics_claude_sonnet_46_loose`.
  - Baseline embedding is not overlaid separately. The selected metrics
    directory is the source for that baseline's plotted and tabulated values.

This means PRISM/Gemini and PRISM/Claude are aligned on the special PT embedding
overlay, but baseline rows remain legacy-first according to their own available
metrics.

## PersonaMem-v2 Main Comparison

Output directory:

- `result/plots/personamem/`

Main files:

- `baselines.csv`
- `baselines_online_metrics_main.pdf`
- `baselines_profile*.pdf`
- `selected_user_ids.txt`
- `excluded_user_ids.txt`

Current cohort:

- Hard-coded `PERSONAMEM_SELECTED_USER_IDS`.
- Current expected size: 28 users.

Source policy:

- `PT (ours)` uses
  `result/openrouter-hybrid-trace-personamem-v2-gemini3-flash-eval`.
- Baselines use the corresponding `baseline_results_personamem_v2/*` paths.
- No PRISM legacy embedding overlay is applied to PersonaMem-v2.

Profile score rubric:

- PersonaMem-v2 does not use PRISM profile dimensions.
- The PersonaMem profile dimensions are:
  - `preference_coverage`
  - `personalization_utility`
  - `update_and_boundary_handling`
  - `memory_quality`
- PersonaMem overall profile score is computed as:

```text
0.30 * preference_coverage
+ 0.30 * personalization_utility
+ 0.25 * update_and_boundary_handling
+ 0.15 * memory_quality
```

## Model Ablation

Output directory:

- `result/plots/model_ablation/`

Main files:

- `learning_trajectory.csv`
- `learning_trajectory_accuracy_gpt_main.pdf`
- `learning_trajectory_accuracy_gpt_profile_cost_main.pdf`

Current matched cohort:

- Intersection of available model-ablation runs and `prism_user_ids()`.
- The current expected size is 28 users.

The four-panel paper figure contains:

- Accuracy curve.
- Relative GPT score curve.
- Overall profile bar chart.
- Cost per turn bar chart.

Cost policy:

- Prefer provider-level token usage from each run's `provider_report.json`.
- If a model run has no provider report, rescale GPT-5 per-turn token usage by
  that model's configured prices.
- Kimi and GLM currently use GPT-5 usage rescaled by their own prices when their
  provider report is unavailable.
- Skip call accounting uses `infer_skip_call_count()`.

Current model prices live in `MODEL_PRICE_PER_MILLION`.

## PRISM Method Ablation

Output directory:

- `result/plots/ablation/`

Main files:

- `prism_method_ablation.csv`
- `prism_method_ablation_accuracy_gpt_main.pdf`
- `prism_method_ablation_accuracy_gpt_profile_cost_main.pdf`
- `ablation_table.tex`

Current matched cohort:

- Same PRISM selected/matched user policy as `assemble_result_runs()` with the
  PRISM ablation run list.

Cost policy:

- `PT(hybrid)` uses one skip call in cost accounting.
- `PT(GPT-5)` uses one skip call in cost accounting.
- `No-gating` / `no-skip` uses zero skip calls.
- `Flat5`, `No-topic`, and `No-consol` use five skip calls.
- The table records:
  - `Acc>20`
  - `GPT>20`
  - `Prof.`
  - turn-wise cost in USD

## Main LaTeX Table

File:

- `main_table.tex`

The table is manually synced from:

- `result/plots/prism/baselines.csv`
- `result/plots/personamem/baselines.csv`

When changing any aggregation rule, regenerate the CSVs first and then update
`main_table.tex`. The table should not be treated as the source of truth.

Current top-level metric groups:

- Preference Prediction:
  - `Acc>20`
  - `DeltaAcc`
- Response Alignment:
  - `GPT>20`
  - `DeltaGPT`
  - `Emb>20`
  - `DeltaEmb`
- Profile Alignment:
  - `Prof.`
  - `Sim.`

Best and second-best marks are manual. Recheck them after any CSV change.

## Common Pitfalls

- Do not change baseline source selection when adjusting PT. The PT legacy
  embedding overlay is intentionally special-cased to `PT (ours)`.
- Do not use Claude primary rows plus legacy baseline embedding fields unless
  that is explicitly intended. The current Claude baseline policy is directory
  selection, not per-field overlay.
- Do not compare a plotted single turn at 25 with `>20` table values. The table
  averages all smoothed turns after 20 with enough users.
- Do not recompute PersonaMem profile caches unless the rubric intentionally
  changed.
- Do not reuse PRISM profile dimensions for PersonaMem.
- Keep `python3` in runnable commands.

