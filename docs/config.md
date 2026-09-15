# Configuration

HyperTrace composes a run from YAML files under `config/`. Paths in `${include:...}` expressions are resolved from the configuration root, which defaults to the repository's `config` directory.

## Default experiment

`config/run/main.yaml` selects `config/run/prism.yaml`. Running `uv run main.py` therefore uses:

```yaml
name: "openrouter-hybrid-trace-gemini3-flash-eval"
dataset: "prism"
seed: 42
n_users: 1000
users_per_run: 50
embed: "${include:embed/openrouter.yaml}"
main_model: "${include:model/openrouter-gpt5.yaml}"
eval_model: "${include:model/openrouter-gemini-3-flash.yaml}"
tracer: "${include:tracer/base_summary_retrieved_long_term.yaml}"
override: "${include:override/hybrid.yaml}"
```

This samples up to 1,000 eligible PRISM users with seed 42, traces the first 50, uses OpenRouter-hosted GPT-5 for the main model, routes selected stages to GPT-5-mini, uses Gemini 3 Flash for evaluation, and uses OpenRouter-hosted `text-embedding-3-small` embeddings. It requires `OPENROUTER_API_KEY`.

`config/run/personamem_v2.yaml` is the corresponding PersonaMem-v2 preset. It selects the PersonaMem prompt adapter and retrieval-focused tracer settings.

## Run fields

| Field | Meaning |
| --- | --- |
| `name` | Default result-directory name. |
| `dataset` | `prism` or `personamem_v2`. |
| `prompt_adapter` | Optional prompt adapter; defaults to the dataset name. |
| `seed` | Dataset sampling and candidate-order seed. |
| `n_users` | Maximum eligible users sampled before cohort selection. |
| `users_per_run` | Number of sampled users in the run cohort. |
| `embed` | Included or inline embedding configuration. |
| `main_model` | Model used for tracing and response adaptation. |
| `eval_model` | Model used for offline response and profile evaluation. |
| `prediction_model` | Optional separate offline preference-prediction model. |
| `tracer` | Online belief update, retrieval, and consolidation settings. |
| `override` | Optional model/reasoning overrides for individual tracing stages. |

## Model and embedding settings

Generation configurations accept `backend`, `model`, `base_url`, sampling and token fields, reasoning options, retry settings, timeout, and provider-specific `extra_body`. Supported generation backends are `openai` and `openrouter`.

Embedding configurations accept `backend`, `model`, `dim`, `base_url`, `max_retries`, and `retry_delay`. Supported backends are `openai`, `openrouter`, `gemini`, and `transformer`. The configured dimension must match the selected embedding model.

The default `uv sync` installs the OpenAI/OpenRouter dependencies. Use `uv sync --extra gemini` or `uv sync --extra transformer` for the optional embedding backends. OpenAI uses `OPENAI_API_KEY`; OpenRouter uses `OPENROUTER_API_KEY`; Gemini accepts `GOOGLE_API_KEY` or `GEMINI_API_KEY`.

## Tracer settings

The public tracer presets expose these main groups:

- Belief maintenance: `n_hypotheses`, `hierarchical`, `allow_expand`, `similarity_threshold`, and `bradley_terry_temp`.
- Evidence handling: `allow_skip`, `max_history_turns`, and `hypothesis_update_mode`.
- Long-term memory: `consolidate_alpha`, summary retrieval pool/limit fields, and minimum-prior thresholds.
- Response-time retrieval: `inference_profile_source`, query mode, retrieval pool/limit fields, and content/style limits.
- Profile generation: `profile_top_p`, `summary_profile_source`, and prediction-profile settings.

Omitted fields use the defaults in `TracerConfig`. Start from the nearest preset rather than repeating every default.

## Create a custom run

Copy a run file, give it a distinct `name`, and change only the components needed for the experiment. For example:

```yaml
name: "prism-gpt54"
dataset: "prism"
seed: 42
n_users: 1000
users_per_run: 50
embed: "${include:embed/openrouter.yaml}"
main_model: "${include:model/openrouter-gpt5.4.yaml}"
eval_model: "${include:model/openrouter-gemini-3-flash.yaml}"
tracer: "${include:tracer/base_summary_retrieved_long_term.yaml}"
override: "${include:override/none.yaml}"
```

Run it with `uv run main.py --config run/prism_gpt54.yaml`. The path is relative to `config/`; use `--config-root` only when maintaining a separate configuration tree.

Use a new run name after changing tracing inputs because record resume is based on file existence. Use a new `--metrics-name` after changing evaluator settings or prompts because metric-cache matching covers model names and evaluation scope, not every input.

The files in `config/ablation/` are complete experimental variants and can be passed to `--config` in the same way. `config/run/openai_batch.yaml` is the supported OpenAI Batch API preset.
