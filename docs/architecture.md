# Architecture

HyperTrace separates online preference inference, provider access, dataset adaptation, prompts, and evaluation. The root `main.py` is the only tracing and evaluation entrypoint.

## Execution flow

1. `main.py` resolves the selected YAML configuration, applies command-line overrides, loads the dataset, and selects the run cohort.
2. For normal runs, one `PreferenceTracer` is created per active user. A thread pool lets users progress concurrently while each user's turns remain sequential.
3. For OpenAI Batch API runs, `core/batch_run.py` owns the batch client lifecycle and `BatchPreferenceTracer` advances users through synchronized request stages.
4. Each completed trace is written atomically as a per-user record. A resumed run skips records already present in its selected cohort.
5. Unless tracing-only mode is selected, evaluation reads exactly that cohort's records, computes per-user metrics concurrently, and aggregates the results.

## Packages

### `core`

`core/preference_tracer.py` coordinates the online loop for one user. The surrounding modules implement preprocessing and low-signal gating, hypothesis initialization and branching, likelihood filtering, perturbation, consolidation, profile retrieval and summarization, and adapted response generation.

`core/hypothesis_set.py` owns working beliefs and long-term hypothesis storage. `core/utils.py` defines tracer settings and task-level model overrides. Batch scheduling lives in `core/batch_tracer.py`; its provider lifecycle adapter lives in `core/batch_run.py`.

### `data`

`data/base.py` defines the shared `UserData`, `Conversation`, and `Turn` contract. `data/loader.py` selects a dataset adapter. `data/prism.py` and `data/personamem_v2.py` transform their source datasets into the shared contract.

### `model`

`model/base.py` defines generation configuration and the language-model interface. `model/loader.py` selects the OpenAI or OpenRouter implementation. Provider modules own request formatting, retry behavior, structured-output parsing, and cleanup. Failed requests remain failures rather than producing placeholder model output.

`model/embed.py` supports OpenAI, OpenRouter, Gemini, and local transformer embeddings. Remote clients are isolated by worker and provider. `model/credentials.py` validates required environment variables before a client is created.

### `prompt`

`prompt/base.py` contains the common prompt set. Dataset adapters replace only the prompts or inserts that differ for PRISM and PersonaMem-v2. Ablation adapters describe intentional prompt variants.

### `eval`

`eval/runner.py` evaluates one user record, validates metric caches, aggregates cohort metrics, and writes JSON atomically. The prediction, response, and profile modules implement their respective metrics.

## Design invariants

- Turns for one user run in order; different users may run concurrently.
- A run cohort is determined by the resolved dataset sample and `users_per_run`, or by an ordered user-ID file.
- Evaluation requires a record for every selected user and ignores unrelated records in the same result directory.
- Configuration files select behavior; provider and dataset modules do not silently substitute unavailable backends or fabricated results.
- Generated records can contain user text and should be reviewed before sharing.
