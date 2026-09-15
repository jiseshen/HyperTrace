# Commands and modules

`main.py` is the only entrypoint for tracing and evaluation. Dataset adapters also expose small inspection commands for checking source data.

## Tracing and evaluation

```bash
uv run main.py [options]
```

With no options, HyperTrace resolves `config/run/main.yaml`, traces its selected cohort, and evaluates the completed records. Concurrent provider requests are the default.

### Run selection

| Option | Default | Purpose |
| --- | --- | --- |
| `--config PATH` | `run/main.yaml` | Run YAML relative to the configuration root. |
| `--config-root PATH` | Repository `config/` | Root for run paths and `${include:...}` references. |
| `--result NAME_OR_PATH` | Resolved config `name` | Result directory beneath the result root. |
| `--result-root PATH` | `result` | Result root relative to the current working directory. |
| `--seed N` | YAML value | Override dataset sampling seed. |
| `--n-users N` | YAML value | Override the sampled eligible-user count. |
| `--users-per-run N` | YAML value | Override the selected cohort size. |
| `--user-ids-file PATH` | unset | Select ordered, unique user IDs from the sampled dataset. Blank lines and `#` comments are ignored. |

### Execution modes

| Option | Purpose |
| --- | --- |
| `--trace-only` | Trace records without offline evaluation. |
| `--eval-only` | Evaluate an existing complete cohort without tracing. |
| `--prediction-only` | Evaluate preference prediction without response/profile judging. |
| `--prediction-model-from-main` | Use `main_model` as the offline prediction evaluator. |
| `--metrics-name NAME` | Store or reuse metrics in a named directory for evaluator comparisons. |

`--trace-only` and `--eval-only` are mutually exclusive. Evaluation-related options cannot be combined with `--trace-only`.

### Concurrency and batch execution

| Option | Default | Purpose |
| --- | --- | --- |
| `--trace-workers N` | `4` | Users traced concurrently in normal mode. |
| `--eval-workers N` | `4` | User records evaluated concurrently. |
| `--use-batch` | off | Trace through the OpenAI Batch API. Requires an OpenAI main model. |
| `--batch-workers N` | `16` | Users advanced concurrently through batch stages. |

Batch mode applies to tracing; evaluation still uses normal concurrent requests. The included preset can be run with:

```bash
uv run main.py --config run/openai_batch.yaml --use-batch
```

Existing records in the selected cohort are skipped during tracing. For evaluation, every selected user must have a record; unrelated records in the directory are ignored.

## Dataset inspection

HyperTrace supports two source datasets:

- **PRISM** loads conversations and survey-derived user evidence from `HannahRoseKirk/prism-alignment`.
- **PersonaMem-v2** loads benchmark text data from `bowen-upenn/PersonaMem-v2`.

Both loaders filter out users with fewer than 20 usable turns by default and convert source rows into the same internal representation:

- `UserData`: a user ID, ground-truth profile, and one or more conversations.
- `Conversation`: an ordered sequence of turns.
- `Turn`: a user message, candidate responses, the chosen response, and its candidate index.

Run configurations control the sampled user count, random seed, selected cohort size, and prompt adapter. See [Configuration](config.md) for those fields and the dataset-specific presets.

PRISM:

```bash
uv run python -m data.prism --n-users 2 --print-stats
uv run python -m data.prism --n-users 2 --print-preview
uv run python -m data.prism --user-id USER_ID --print-preview
```

PersonaMem-v2:

```bash
uv run python -m data.personamem_v2 --n-users 2 --print-stats
uv run python -m data.personamem_v2 --n-users 2 --print-preview
```

Both commands accept `--seed`. PersonaMem-v2 also accepts `--split`. They download source data through Hugging Face when it is not already cached.

## Python modules

The internal data contract is exported from `data`, model interfaces and configurations from `model`, prompt sets from `prompt`, and belief types/configuration from `core`. These imports support research extensions, but the repository does not currently promise a separately versioned library API. See [Architecture](architecture.md) for ownership and data flow, and [Configuration](config.md) for experiment composition.
