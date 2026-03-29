# EventBatchSurv

PyTorch framework for studying small-batch Cox optimization under event scarcity and censoring.

## Setup

```bash
conda env create -f environment.yml
conda activate ebs
```

## Quick Start

Run the default grid:

```bash
python scripts/run_grid.py --base-config configs/base.yaml --grid-config configs/grid_event_censor.yaml
```

Write the main summaries:

```bash
python scripts/summarize_results.py --results-dir results
python scripts/compact_results.py --results-dir results
```

For the shortest readout, open `results/compact/report.md`.

Run a tiny smoke check:

```bash
python scripts/run_grid.py --smoke
```

Optional figures:

```bash
python scripts/make_figures.py --results-dir results
```

Backbone is configurable in `configs/base.yaml`:

```yaml
model:
  backbone: cox_mlp   # options: cox_mlp, resnet18
```

`resnet18` uses `weights=None`, grayscale `conv1`, and a single-output Cox head.

## Policies

- `random`: unconstrained random mini-batches
- `event_quota_wor_<pct>` / `event_quota_wr_<pct>`: event-enriched batches by label count
- `riskset_anchor_<pct>`: event-anchor batches with risk-set-aware fillers

## Outputs

- `results/<grid>_YYYYmmdd_HHMMSS/`: compact run snapshot with per-grid summaries
- `results/runs/`: optional raw per-run logs (`epoch_logs.jsonl`, `batch_logs.jsonl`)
- `results/aggregates/`: enriched run summaries plus paired test outputs
- `results/compact/summary.csv`: one compact CSV with overall and condition rows
- `results/compact/report.md`: preferred quick-read markdown summary
- `results/tables/`: manuscript-style tables
- `results/figures/`: optional figures

## Evaluation Notes

- Empirical zero-event and weak-information batch frequencies are tracked for each run
- Theoretical checks use `(1 - p)^b` and `(1 - p)^b + b p (1 - p)^(b-1)`
- Validation C-index (`best_val_c_index`) is used for selection
- Held-out test Harrell and Uno C-index are the main comparison metrics
- Paired summaries use effect sizes and Benjamini-Hochberg q-values

Here `p` is the realized train-set event rate and `b` is batch size.
