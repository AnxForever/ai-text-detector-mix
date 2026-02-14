# V11 Regression Gate

- Generated at: 2026-02-13 12:58:47
- Baseline: `bert_v10_augmented` from `/mnt/c/datacollection/models/bert_v10_augmented/eval_comparison.json`
- Candidate: `bert_v11c_boundary_fix` from `/mnt/c/datacollection/models/bert_v11c_boundary_fix/eval_comparison.json`

## Gate result

- Decision: **keep_candidate**
- Pass gate: `True`
- Reason: three_set_avg degradation -0.2000 <= threshold 0.5000
- Baseline three_set_avg: 98.36
- Candidate three_set_avg: 98.56
- Delta (candidate - baseline): 0.2

## Per-set deltas

| dataset | accuracy_delta | precision_delta | recall_delta | f1_delta |
|---|---:|---:|---:|---:|
| core_v1_test_clean | -0.37 | -0.6 | 0.0 | -0.3 |
| independent_data | 0.88 | 3.99 | 0.67 | 2.46 |
| merged_v2_val_clean | 0.09 | -0.19 | 0.39 | 0.1 |

## Rollback policy

- If `three_set_avg` degrades by more than 0.5 points, keep V10 and rollback.
- Analyze degradation sources (domain, length bucket, unknown-source mix) before retrying.
