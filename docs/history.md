# Cleanup history

## 2026-09-15 — Publication preparation

Consolidated tracing and evaluation in `main.py` with concurrent requests by default and preserved OpenAI batch mode. Added CLI overrides, cohort-scoped evaluation, supported public presets, runtime dependencies, and corrected documentation. Replaced experiment-specific plots with direct saved-metric plots; removed fixed cohorts, score offsets, unsupported configs, and fabricated provider completions. Removed the batch scheduler’s hardcoded 10% early-stop cutoff so all selected users finish. Removed the superseded sequential evaluation wrapper. Fixed embedding-client isolation and handling of failed asynchronous stage outputs.

Validation: Python 3.12.14; 17 run/ablation presets resolve and instantiate; 14 offline smoke-check groups pass, covering modes, sampling/cohorts, resume, batch wiring, failure propagation, plotting, and working-directory independence. Plot values and missing-data gaps were checked numerically and visually. Additional checks passed for concurrent embedding-client isolation, the real batch scheduler finishing the last long-running user in a 20-user synthetic cohort, explicit invalid counts after branch/filter failures, command help, compilation, documentation links, and dependency consistency. Temporary validation files and generated plots were removed after inspection; no external provider calls were made.

Follow-up: verify live dataset/provider access and scientific results separately; supply owner-approved license, author metadata, and citation details. See `guide.md` for cache and release limitations. Implementation is recorded by the commit containing this entry.

## 2026-09-15 — License and citation

Added the user-selected MIT License, attributed to the HyperTrace authors, and the author-supplied arXiv citation in the README. Corrected the BibTeX URL formatting and aligned the README title with the paper. Updated the repository guide to resolve the missing-metadata item from the publication-preparation milestone.

Validation: checked the license against the Open Source Initiative MIT text, the supplied citation fields, the local license link, and diff whitespace. Live provider/dataset verification remains outstanding.
