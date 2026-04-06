# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

EventBatchSurv (EBS) is a PyTorch-based research framework for studying small-batch Cox proportional hazards optimization. It is a single Python package (`src/ebs/`) with CLI scripts in `scripts/`. No web server, database, or external services are required.

### Environment

- **Python 3.11** is required (matches `environment.yml`). The VM uses a virtualenv at `/workspace/.venv` instead of conda.
- Activate with: `source /workspace/.venv/bin/activate`
- All dependencies from `environment.yml` are installed via pip into the venv.

### Running the application

See `README.md` for full details. Key commands (all require the venv activated):

| Task | Command |
|------|---------|
| Quick smoke test | `python scripts/run_grid.py --smoke` |
| Mini grid (fastest full run) | `python scripts/run_grid.py --base-config configs/base.yaml --grid-config configs/mini.yaml --num-seeds 1` |
| Summarize results | `python scripts/summarize_results.py --results-dir results` |
| Compact summary | `python scripts/compact_results.py --results-dir results` |
| Generate figures | `python scripts/make_figures.py --results-dir results` |

### Linting

- Run `ruff check src/ scripts/ --ignore E402` — the E402 (module-level import not at top) violations are intentional because scripts use `sys.path.insert` before importing `ebs.*`.
- No test suite or CI pipeline exists in this repo.

### Gotchas

- Training runs on CPU by default (`device: cpu` in `configs/base.yaml`). The `--smoke` flag still runs all 10 batching policies × 2 batch sizes × 2 seeds = 40 runs, which takes ~20 min on CPU. For a faster verification, use the `mini.yaml` grid config with `--num-seeds 1` (9 runs, ~2 min).
- MNIST data is auto-downloaded on first run to `data/mnist/`.
- Results are written to `results/` (gitignored). Delete this directory to reset state.
