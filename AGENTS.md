# Repository Guidelines

## Project Structure & Module Organization
- `evaluate.py` is the main entrypoint for SentEval scoring and orchestrates token-prepending experiments; keep new workflows here or call into it from a thin wrapper.
- Model adapters live in `senllm/` (e.g., `modeling_llama.py`, `modeling_qwen2.py`); mirror the existing class layout when adding families so attention overrides remain compatible.
- Runtime configuration is centralized in `config.yaml`, with auxiliary assets in `head_score/` (precomputed head weights) and `attention_analysis/` (analysis artifacts); treat `SentEval/` as a vendor directory for benchmarks.

## Build, Test, and Development Commands
- `pip install -r requirements.txt` installs the pinned Transformer and PyTorch stack; run inside a Python 3.9+ environment.
- `bash SentEval/data/downstream/download_dataset.sh` fetches benchmark data after `cd SentEval/data/downstream/`; required before any evaluation.
- `bash run.sh llama-2-7b-tp` executes a configured experiment; pass `{model_name} {config_path}` for custom setups.
- `bash run_zero_special_grid.sh` automates zero-special threshold sweeps and streams logs to `logs_zero_special/`.

## Coding Style & Naming Conventions
- Follow PEP 8 defaults: 4-space indentation, snake_case for functions, PascalCase for classes, and explicit imports grouped by standard/library/local origin.
- Type hints are gradually adopted; add them when touching functions in `senllm/` or `evaluate.py` to keep signatures clear.
- Use f-strings for formatting and prefer pure functions over side effects; align configuration keys with the hyphenated patterns used in `config.yaml`.

## Testing Guidelines
- Smoke-test changes with `python evaluate.py --config llama-2-7b --config_file config.yaml`, pinning `CUDA_VISIBLE_DEVICES` if you rely on a specific GPU.
- When adding new configs, confirm SentEval scores regenerate expected tables and stash artifacts under a dated folder inside `attention_analysis/`.
- Document any long-running sweeps by attaching the relevant `logs_zero_special/*.log` paths to your review notes.

## Commit & Pull Request Guidelines
- History uses short imperative summaries (e.g., `attention_se 2.1 整理代码`); keep commits under ~72 characters and scope them to a single change set.
- Reference impacted modules in the body, list config keys touched, and note the evaluation command you ran.
- Pull requests should link issues when available, describe the experiment goal, and include score deltas or sample tables/screenshots to substantiate claims.

## Configuration & Data Tips
- Clone existing entries in `config.yaml` when introducing a model; ensure `model_name_or_path`, `use_which_plan`, and attention overrides are explicitly set.
- Set `HF_ENDPOINT=https://hf-mirror.com` (as in `run.sh`) when working behind restricted networks, and keep credentials out of tracked files.
- Large checkpoints and SentEval corpora stay outside version control; document any new storage locations in the PR description.
