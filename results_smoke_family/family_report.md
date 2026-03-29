# Family Summary

- CSV: `results_smoke_family/tables/table_family_summary.csv`
- Figure: `results_smoke_family/figures/family_summary_test_uno.png`

## Overall Mean Test Uno
- random: 0.5889 | selected policies {"random": 1}
- event_quota: 0.4480 | selected policies {"event_quota_wr_50": 1}
- riskset_anchor: 0.5139 | selected policies {"riskset_anchor_25": 1}

## Delta Vs Random
- event_quota (lt1): -0.1409
- riskset_anchor (lt1): -0.0750

## Reading Guide
- `event_quota` collapses all `event_quota_*` variants after within-family selection by validation C-index.
- `riskset_anchor` collapses all `riskset_anchor_*` variants after within-family selection by validation C-index.
- Condition rows in the CSV keep `event_target`, `censor_target`, and `batch_size`; overall rows summarize all selected runs by family.
