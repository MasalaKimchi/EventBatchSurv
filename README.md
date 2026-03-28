# EventBatchSurv

Manuscript-grade PyTorch framework for studying small-batch Cox optimization under event scarcity and censoring, with random and event-aware batch sampling policies.

## Environment

```bash
conda env create -f environment.yml
conda activate ebs
```

## Run Experiments

Full grid (event prevalence x censoring x batch size x policy x seed):

```bash
python scripts/run_grid.py --base-config configs/base.yaml --grid-config configs/grid_event_censor.yaml
```

Smoke test:

```bash
python scripts/run_grid.py --smoke
```

Model backbone is configurable in `configs/base.yaml`:

```yaml
model:
  backbone: cox_mlp   # options: cox_mlp, resnet18
```

`resnet18` uses `weights=None`, grayscale `conv1`, and a single-output Cox head, aligned with the TorchSurv notebook pattern.

## Aggregate, Plot, and Interpret

```bash
python scripts/summarize_results.py --results-dir results
python scripts/make_figures.py --results-dir results
python scripts/write_interpretation.py --results-dir results
```

Manuscript-standard reporting in this repo uses:
- Validation C-index only for model selection (`best_val_c_index`)
- Held-out test metrics for primary inference (`test_harrell_c_index`, `test_uno_c_index`)
- Multiple-testing control (Benjamini-Hochberg q-values) and paired effect sizes

Batching policy names:
- `random`: unconstrained random mini-batches (baseline)
- `event_quota_wor_<pct>`: enforce event quota without replacement (e.g. `event_quota_wor_25`, `event_quota_wor_50`, `event_quota_wor_75`)
- `event_quota_wr_<pct>`: enforce event quota with replacement (e.g. `event_quota_wr_25`, `event_quota_wr_50`, `event_quota_wr_75`)

## Outputs

- `results/<grid>_YYYYmmdd_HHMMSS/`: compact run folder with `run_summaries.csv`, `run_summaries.json`, `seed_aggregates.csv`, and `manifest.json`
- `results/runs/`: raw per-run logs (`epoch_logs.jsonl`, `batch_logs.jsonl`, `run_meta.json`)
- `results/aggregates/`: run-level and condition-level CSV summaries, plus enhanced paired tests for test Harrell/Uno/Brier (`*_enhanced.csv`) with effect sizes and BH q-values
- `results/tables/`: manuscript-ready CSV tables including:
  - `table_main.csv` (test Harrell + test Uno + validation means/std/CI)
  - `table_theory.csv` (empirical-theory batch informativeness checks)
  - `table_feasibility_test_uno.csv` (strict-feasibility breakout)
  - `table_high_value_targets.csv` (high-value manuscript signals)
- `results/figures/`: test-metric heatmaps, gain maps, training curves, event-count histograms, gradient-variance trends, forest deltas, paired seed slopes, validation-test gap, mechanism scatter, and improvement heatmaps
- `results/interpretation.md`: auto-generated theory-facing interpretation

## Notes on Theory Checks

For each run the framework records:
- Empirical zero-event batch frequency and weakly informative frequency (`E <= 1`)
- Theoretical zero-event frequency `(1 - p)^b`
- Theoretical weakly informative probability `(1 - p)^b + b p (1 - p)^(b-1)`

where `p` is the realized train-set event rate and `b` is batch size.
