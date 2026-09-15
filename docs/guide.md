# Repository guide

## Architecture and source of truth

- `main.py`: CLI options, configuration loading, cohort selection, resume, concurrent tracing, offline evaluation, and provider reports.
- `core/preference_tracer.py`: sequential turn-by-turn inference for one user. Stage implementations are separate modules in `core/`; `hypothesis_set.py` owns belief storage and retrieval.
- `core/batch_run.py`: batch lifecycle used by the CLI; `core/batch_tracer.py` coordinates stage barriers with `model/batch_queue_model.py`.
- `data/base.py`: shared user/conversation/turn types; `data/loader.py` routes datasets. Dataset modules provide separate inspection commands.
- `prompt/base.py`: common prompt structure; dataset adapters specialize it. PersonaMem's explicit `legacy_retrieved` prediction profile remains a consumed research setting.
- `model/loader.py`: supported generation providers. `model/credentials.py` validates keys; `model/embed.py` manages worker-local clients and the local encoder cache.
- `eval/runner.py`: record evaluation, aggregation, cache matching, and atomic JSON writing; metric modules own scoring.
- `visualize.py`: plots persisted aggregate metrics without data downloads or provider calls.
- `config/run/prism.yaml` and `personamem_v2.yaml`: public run presets. The main alias points to PRISM. Provider-specific request conventions remain in provider adapters.

## Goals and accepted decisions

Keep one tracing/evaluation entrypoint, concurrent requests by default, optional OpenAI batch tracing, separate utilities, and no experiment-specific cohort or score manipulation. Keep the package structure; do not introduce a packaging migration just for publication. Use a fresh run for changed trace settings and a fresh metric directory for changed evaluation inputs.

## Validation

Use Python 3.11+ and install `requirements.txt`; plotting additionally needs matplotlib. Python 3.9 cannot import this code's typing interfaces. Windows activation: `.\.venv\Scripts\Activate.ps1`.

Run the command-help, compile, and dependency checks listed in the README. Resolve all run/ablation presets through `main.load_run_config`, then construct `GenerationConfig`, `EmbedConfig`, `TracerConfig`, `OverrideConfig`, and the selected prompt adapter.

Temporary offline smoke checks cover sampling overrides, trace-only, evaluation-only, concurrent execution, fixed cohorts, resume, invalid modes/counts, missing records, batch wiring, provider failures, and plots with exact saved values. Mock provider and dataset boundaries; do not substitute mocks for import or config checks. There is no permanent test suite.

## Release limitations and next steps

- Live dataset downloads and provider generation, embeddings, and Batch API completion require separate verification; offline checks do not establish service availability or scientific quality.
- Optional Gemini and local transformer backends require their additional dependencies and are not validated against live services/models.
- Record resume is based on file existence. Metric caching matches model names and scope, not complete inputs. Use new output names for changed experiments.
- Provider reports cover calls made in the current invocation and are not a cumulative run ledger.
- License, author metadata, and citation information have not been supplied. Add the owner's chosen terms and verified publication metadata before public release.
- Git ignore rules do not scrub existing Git history. This cleanup is a current-tree review, not a historical secrets audit.

See [cleanup history](history.md) for the verified publication-cleanup milestone.
