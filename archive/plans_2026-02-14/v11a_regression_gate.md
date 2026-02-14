# V11 Regression Gate

- Generated at: 2026-02-12 21:41:49
- Baseline: `bert_v10_augmented` from `/mnt/c/datacollection/models/bert_v10_augmented/eval_comparison.json`
- Candidate: `bert_v11a_clean` from `/mnt/c/datacollection/models/bert_v11a_clean/eval_comparison.json`

## Gate result

- Decision: **rollback_to_baseline**
- Pass gate: `False`
- Reason: three_set_avg degradation 0.6100 > threshold 0.5000
- Baseline three_set_avg: 98.36
- Candidate three_set_avg: 97.75
- Delta (candidate - baseline): -0.61

## Per-set deltas

| dataset | accuracy_delta | precision_delta | recall_delta | f1_delta |
|---|---:|---:|---:|---:|
| core_v1_test_clean | -0.92 | -1.48 | 0.0 | -0.75 |
| independent_data | -0.66 | -3.95 | 1.33 | -1.64 |
| merged_v2_val_clean | -0.26 | -0.57 | 0.0 | -0.29 |

## Rollback policy

- If `three_set_avg` degrades by more than 0.5 points, keep V10 and rollback.
- Analyze degradation sources (domain, length bucket, unknown-source mix) before retrying.
