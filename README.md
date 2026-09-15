# Hypothesis-Based Preference Tracing for Online LLM Personalization

Preference Tracing is a training-free online personalization framework. It maintains weighted natural-language hypotheses about a user's preferences, updates them from chosen-versus-rejected response feedback, and uses the current belief to generate personalized responses.

## Quick start

Use **Python 3.11 or newer**. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export OPENROUTER_API_KEY="your-key"
python main.py --result prism-smoke --n-users 20 --users-per-run 2
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1` and set the key with `$env:OPENROUTER_API_KEY="your-key"`.

The default runs PRISM tracing followed by evaluation, using four concurrent users per phase. It uses OpenRouter GPT-5 with stage-specific GPT-5-mini routing, Gemini 3 Flash for evaluation, and OpenRouter embeddings. Dataset downloads require network access; tracing and evaluation make paid provider requests. Keys are read from environment variables; `.env` files are not loaded automatically.

For PersonaMem-v2:

```bash
python main.py --config run/personamem_v2.yaml --result personamem-smoke --n-users 20 --users-per-run 2
```

## Repository layout

```text
HyperTrace/
|-- main.py             # Sole tracing and evaluation CLI
|-- visualize.py        # Plot saved metrics, without API calls
|-- requirements.txt    # Core runtime dependencies
|-- config/             # Run presets, models, embeddings, tracer settings
|-- core/               # Online inference and batch orchestration
|-- data/               # Dataset loading and shared data types
|-- eval/               # Prediction, response, and profile metrics
|-- model/              # Provider clients, batching, embeddings
|-- prompt/             # Shared prompts and dataset adapters
`-- docs/               # Architecture, validation, and cleanup history
```

See [the repository guide](docs/guide.md) for the code map, verification commands, and release limitations.

## Method

Preference Tracing represents the current user state with `K` weighted hypotheses:

$$
B_t = {(h_t^1, w_t^1), ..., (h_t^K, w_t^K)}
$$

Each hypothesis is a compact natural-language explanation of the user's current preference, such as a preference for direct implementation details, broader conceptual framing, cautious wording, or a topic-specific constraint.

At each turn, the tracer runs the following loop:

1. **Adapt.** Generate an answer to the current user message using the current profile or a retrieved response-time profile.
2. **Gate.** Skip turns that do not carry reliable preference evidence, such as greetings, fillers, or candidate sets with no meaningful preference contrast.
3. **Preprocess.** Compress long candidate responses into summaries while preserving the dimensions that distinguish chosen and rejected answers.
4. **Initialize or Branch.** If no working belief exists, propose `K` hypotheses. Otherwise, revise, replace, or retrieve hypotheses that explain the new observation.
5. **Filter.** Ask the LLM to score each candidate response under each hypothesis.
6. **Resample and Perturb.** If effective sample size drops, resample high-weight hypotheses. Similar hypotheses are merged or perturbed to keep the belief set diverse.
7. **Consolidate.** At the end of a conversation, the working belief is merged into long-term memory with an importance weight based on useful turn count and belief entropy.
8. **Evaluate.** The final records are scored for response alignment, preference prediction, and profile alignment.

This keeps personalization interpretable and black-box compatible: no model parameters are updated, and all adaptation happens through structured inference over hypotheses.

## Datasets

- `prism`: PRISM conversations and survey evidence from `HannahRoseKirk/prism-alignment`.
- `personamem_v2`: PersonaMem-v2 text data from `bowen-upenn/PersonaMem-v2`.

Both loaders require at least 20 usable turns by default. Run presets sample up to 1,000 eligible users with seed 42 and select the first 50 from that order. CLI sampling overrides apply before cohort selection.

`UserData` contains a user ID, ground-truth profile, and conversations. Each conversation contains turns with a user message, candidate responses, and the chosen response/index.

Dataset inspection commands (these download data if it is not cached):

```bash
python -m data.prism --n-users 2 --print-stats
python -m data.personamem_v2 --n-users 2 --print-stats
```

## Tracing and evaluation

No model weights are trained. Tracing is the online adaptation pass.

```bash
python main.py --result prism-smoke --n-users 20 --users-per-run 2 --trace-only
python main.py --result prism-smoke --n-users 20 --users-per-run 2 --eval-only
```

Use the same config, seed, sample size, and cohort when resuming or evaluating a run. Existing record files are skipped during tracing. Evaluation requires records for every selected user and ignores records outside that cohort. Use a new result name when changing tracing settings; existing records are not automatically invalidated.

Advanced options:

- `--trace-workers` / `--eval-workers`: concurrent users per phase; default 4.
- `--seed`, `--n-users`, `--users-per-run`: sampling and cohort overrides.
- `--user-ids-file`: unique, newline-delimited IDs in the sampled dataset; blank lines and `#` comments are ignored. File order determines the cohort, capped by `users_per_run`.
- `--prediction-only`: restrict offline evaluation to preference prediction; add `--eval-only` to skip tracing.
- `--prediction-model-from-main`: use the main model for offline prediction.
- `--metrics-name`: choose a separate metric directory when comparing evaluators. Default is `metrics`, or a model-specific directory for prediction-only runs.
- `--config-root`: override the configuration directory. Its default is relative to `main.py`, so the entrypoint also works from another working directory.
- `--result-root`: output root, default `result` relative to the current working directory. `--result` chooses a run name or explicit path.

Metric caches match evaluator model names and evaluation scope. They do not fingerprint prompts, provider settings, or source records. Choose a fresh `--metrics-name` when changing those inputs.

### OpenAI Batch API

Concurrent requests are the default. The included batch preset uses OpenAI for tracing, embeddings, and evaluation:

```bash
python main.py --config run/openai_batch.yaml --use-batch --batch-workers 16
```

Set `OPENAI_API_KEY` for batch tracing and the keys required by the preset's embedding and evaluation providers. Batch mode applies to tracing; offline evaluation still uses concurrent requests. It requires an OpenAI main model, submits remote batch jobs, and may take much longer than an interactive run. Batch completion window, polling interval, and timeout are model configuration fields.

### Metrics

- **Response alignment:** embedding similarity and LLM judging relative to chosen and rejected responses.
- **Preference prediction:** ranking candidate responses using the inferred profile.
- **Profile alignment:** dataset-specific judging of the final profile against user evidence.

Relative response alignment is `S(adapted, chosen) - max S(adapted, rejected)`.

## Configuration

`config/run/main.yaml` points to `run/prism.yaml`. `run/personamem_v2.yaml` selects PersonaMem-v2. Other named presets and `config/ablation/` define explicit experimental variants.

Nested configuration values use `${include:model/openrouter-gpt5.yaml}` to include a YAML file relative to the configuration root.

Key fields:

- `dataset` and optional `prompt_adapter`: dataset loading and prompt selection.
- `main_model`, `eval_model`, optional `prediction_model`: generation configuration. Supported generation backends are `openai` and `openrouter`.
- `embed`: embedding backend, model, and dimension; keep the dimension consistent with the model output.
- `tracer`: belief size, gating, retrieval, consolidation, and profile settings.
- `override`: task-specific model routing within the main model's provider.

Models and method parameters are explicit research defaults, not automatic provider discovery. Availability depends on your provider account. Optional OpenRouter headers use `OPENROUTER_HTTP_REFERER` and `OPENROUTER_APP_TITLE`.

Optional embedding dependencies:

```bash
python -m pip install google-genai          # Gemini; GOOGLE_API_KEY or GEMINI_API_KEY
python -m pip install sentence-transformers # Local transformer embeddings
```

## Outputs and plotting

```text
result/<run>/
|-- records/<user_id>.json
|-- metrics/summary.json
|-- metrics/users/<user_id>.json
|-- provider_report.json
|-- eval_provider_report.json
`-- prediction_provider_report.json
```

Provider reports are emitted for OpenRouter calls when available. Records contain per-turn inference state and responses. Metrics contain per-user scores and aggregate turn metrics. Generated runs are ignored by Git; they can contain user text and should be reviewed before sharing.

Plot saved metrics:

```bash
python -m pip install matplotlib
python visualize.py result/prism-smoke
```

Plots are written to `<run>/plots/<metrics-name>/`. Use `--metrics-name` to select alternate metrics or `--output-dir` to choose a destination. Curves use saved turn indices and scores directly, preserve missing-value gaps, and include sample-size curves. Plotting never selects users by score, adjusts scores, or calls a provider.

## Development checks

```bash
python main.py --help
python visualize.py --help
python -m data.prism --help
python -m data.personamem_v2 --help
python -m compileall -q main.py visualize.py core data eval model prompt
python -m pip check
```

See [the repository guide](docs/guide.md) for architecture and validation limitations.
