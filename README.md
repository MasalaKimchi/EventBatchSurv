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

## Aggregate, Plot, and Interpret

```bash
python scripts/summarize_results.py --results-dir results
python scripts/make_figures.py --results-dir results
python scripts/write_interpretation.py --results-dir results
```

## Outputs

- `results/runs/`: raw per-run logs (`epoch_logs.jsonl`, `batch_logs.jsonl`, `run_meta.json`)
- `results/aggregates/`: run-level and condition-level CSV summaries and paired tests
- `results/tables/`: manuscript-ready table CSVs
- `results/figures/`: heatmaps, training curves, event-count histograms, gradient variance plots, gain map
- `results/interpretation.md`: auto-generated theory-facing interpretation

## Notes on Theory Checks

For each run the framework records:
- Empirical zero-event batch frequency and weakly informative frequency (`E <= 1`)
- Theoretical zero-event frequency `(1 - p)^b`
- Theoretical weakly informative probability `(1 - p)^b + b p (1 - p)^(b-1)`

where `p` is the realized train-set event rate and `b` is batch size.
