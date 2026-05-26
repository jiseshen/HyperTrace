# PRISM Experimental Setting Summary

## Main Text Description

We evaluate online personalization on a fixed PRISM cohort. Users are first filtered to have at least 20 usable turns, then sampled with seed 42; the main PRISM run traces 50 users with GPT-5. At each turn, Preference Tracing maintains a working belief of 5 preference hypotheses. The tracer skips low-signal interactions, initializes or branches hypotheses from chosen-vs-rejected evidence, updates hypothesis weights with a Bradley-Terry likelihood temperature of 1.0, and summarizes the current belief into a profile used for response adaptation and offline preference prediction. The final profile additionally retrieves long-term hypotheses from a vector store, using a 30-item retrieval pool, 5 long-term hypotheses, and a minimum prior of 0.1. Embeddings use `text-embedding-3-small`.

We compare against four LLM baselines and one supervised reranker. CoT predicts preferences from the current query and the last 5 observed interactions. RAG retrieves up to 5 previous example interactions by embedding similarity and conditions prediction and response generation on those examples. Cheatsheet maintains a running user-preference cheatsheet updated after each observed gold interaction, capped at 10 bullets. HyperAlign maintains up to 10 natural-language hypotheses and selects up to 5 relevant hypotheses for each prediction. Hydra is a RoBERTa-base supervised reranker over candidate responses; it samples 1,000 out-of-bag turns for training, uses a history window of 5 turns and max sequence length 1024, and is evaluated only on candidate reranking because it does not generate adapted responses. All LLM baselines use GPT-5 as the reasoning/generation model and generate adapted responses with the same response-generation prompt family as the tracing method. For response quality, we report both embedding-based relative similarity and LLM-judge relative score by `gemini-3-flash`; `claude-sonnet-4.6` evaluation reruns the LLM judge and profile-based prediction while reusing the same embedding scores.

## Core Parameters

| Component | Setting |
| --- | --- |
| Dataset | PRISM conversations |
| User filter | at least 20 usable turns |
| Sampling | seed 42, `n_users=1000`, fixed 50-user run prefix |
| Main trace model | `openai/gpt-5` via OpenRouter |
| Default judge | `google/gemini-3-flash-preview` via OpenRouter |
| Additional judge | `anthropic/claude-sonnet-4.6` via OpenRouter |
| Embedding model | `openai/text-embedding-3-small`, 1536 dimensions |
| Working belief size | `n_hypotheses=5` |
| Likelihood model | Bradley-Terry, temperature 1.0 |
| Recent history for prompts | 3 turns |
| Skip gating | enabled |
| Hypothesis consolidation | hierarchical, `consolidate_alpha=1.0` |
| Long-term vector store | FAISS inner-product index, capacity 1000 |
| Final-profile long-term retrieval | pool 30, top 5, min prior 0.1 |

## Baseline Parameters

| Baseline | Main mechanism | Key parameters |
| --- | --- | --- |
| CoT | LLM listwise prediction from recent observed interactions | GPT-5; last 5 previous turns; no current gold in prediction; generated adapted response |
| RAG | LLM listwise prediction with retrieved previous examples | GPT-5; embedding cosine retrieval; `rag_top_k=5`; query is current user message; corpus is previous observed turns |
| Cheatsheet | Running natural-language user-preference profile | GPT-5 reasoning; Gemini update model; max 10 bullets; update after prediction using current gold interaction |
| HyperAlign | Iterative natural-language hypothesis generation and selection | GPT-5 reasoning; Gemini hypothesis/selection models; max 10 hypotheses; select top 5; update every turn |
| Hydra | Supervised candidate reranker | RoBERTa-base; 1,000 out-of-bag training turns; history window 5; max length 1024; user-disjoint split; evaluated for reranking only |

## Model Ablation

The model ablation isolates the effect of the tracing model while keeping the tracing algorithm, evaluation protocol, prompt family, and PRISM user cohort fixed. Each run replaces the main model used by Preference Tracing with one target model and uses the same downstream evaluator for preference prediction and response-quality scoring. For experimental reproducibility, all model-ablation runs use the lowest reasoning-effort setting available for the corresponding model endpoint and the provider-recommended generation parameterization; we do not tune sampling parameters per model.

| Run label | Main tracing model | Purpose |
| --- | --- | --- |
| GPT-5 | `openai/gpt-5` | Primary high-capability reference model for Preference Tracing. |
| Kimi K2.6 | `moonshotai/kimi-k2.6` | Frontier non-OpenAI comparison under the same tracing pipeline. |
| GLM-5.1 | `z-ai/glm-5.1` | Frontier non-OpenAI comparison under the same tracing pipeline. |
| DeepSeek-V4 | `deepseek/deepseek-v4-flash` | Fast frontier-model comparison under the same tracing pipeline. |
| Qwen3.5-397B | `qwen/qwen3.5-397b-a17b` | Large open-weight-style model comparison under the same tracing pipeline. |
| Qwen3.5-9B | `qwen/qwen3.5-9b` | Smaller model comparison testing whether the tracing pipeline remains effective with a lower-capacity reasoner. |

## Method Ablation

The method ablation evaluates which structural components of Preference Tracing contribute to online personalization quality and cost. All method-ablation runs use the same fixed 28-user PRISM cohort selected from the GPT-5-filtered experiment subset, the same evaluator, and the same hybrid model routing as the hybrid main run unless the ablation explicitly changes the traced mechanism. The hybrid reference uses GPT-5 for higher-level reasoning steps and GPT-5-mini for selected cheaper update and summarization steps.

| Run label | Intervention | Interpretation |
| --- | --- | --- |
| PT(GPT-5) | Full Preference Tracing with GPT-5 as the tracing model. | Capability reference for the full method without hybrid model routing. |
| PT(hybrid) | Full Preference Tracing with hybrid GPT-5 / GPT-5-mini routing. | Efficiency reference: preserves the hierarchical tracing design while moving selected lower-risk steps to the smaller model. |
| No-gating | Disables skip gating, so every usable interaction is sent through the trace/update path. | Tests whether the skip module mainly removes sparse or noisy evidence. If many interactions are low-signal, this should increase cost without reliably improving personalization. |
| Flat5 | Removes the hierarchical hypothesis-store structure. The tracer initializes one set of five hypotheses per user and keeps updating those same five slots throughout the conversation; when evidence is unrelated, the branch step directly replaces the irrelevant slot instead of retrieving or consolidating a broader hypothesis store. | Tests whether hierarchical memory, retrieval, and consolidation are necessary beyond simply maintaining five online hypotheses. |
| No-topic | Keeps the hierarchical update pipeline but removes topic/category text from stored, embedded, logged, and summarized hypotheses. | Tests whether topic grounding helps retrieval and profile synthesis distinguish semantically different preference regions. |
| No-consol | Replaces hierarchical consolidation with retrieve-and-replace belief updates. At each usable update turn, the tracer retrieves and reranks candidate hypotheses, normalizes the selected retrieved priors into the current belief, runs the local branch/weight/perturb update, and directly writes the resulting weights back to the selected hypotheses instead of using the normal importance-weighted consolidation path. | Tests whether explicit consolidation is needed, or whether repeatedly retrieving the most relevant hypotheses and overwriting their weights is sufficient. |

## Appendix Parameters

| Parameter | Value |
| --- | --- |
| `allow_skip` | true |
| `allow_expand` | false |
| `similarity_threshold` | 0.8 |
| `profile_top_p` | 0.8 |
| `summary_profile_source` | `belief_with_retrieved_long_term` |
| `summary_retrieve_pool_k` | 30 |
| `summary_long_term_top_k` | 5 |
| `summary_min_prior` | 0.1 |
| `inference_profile_source` | `working` |
| `inference_retrieve_top_k` | 12 |
| `inference_retrieve_pool_k` | 30 |
| `inference_min_prior` | 0.15 |
| GPT-5 trace reasoning effort | minimal globally; low for initialize/branch/response; medium for perturb; minimal for preprocessing/filtering/summary/profile/prediction |
| Baseline response max tokens | 512 |
| Baseline score max tokens | 256 |
| Baseline judge temperature | 0.0 |
