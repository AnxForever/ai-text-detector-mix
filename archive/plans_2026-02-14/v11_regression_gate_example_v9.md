# V11 Regression Gate

- Generated at: 2026-02-12 19:24:01
- Baseline: `bert_v10_augmented` from `models\bert_v10_augmented\eval_comparison.json`
- Candidate: `bert_v9_p0_supplement` from `models\bert_v10_augmented\eval_comparison.json`

## Gate result

- Decision: **rollback_to_baseline**
- Pass gate: `False`
- Reason: three_set_avg degradation 1.0500 > threshold 0.5000
- Baseline three_set_avg: 98.36
- Candidate three_set_avg: 97.31
- Delta (candidate - baseline): -1.05

## Per-set deltas

| dataset | accuracy_delta | precision_delta | recall_delta | f1_delta |
|---|---:|---:|---:|---:|
| core_v1_test_clean | 0.0 | 0.0 | 0.0 | 0.0 |
| independent_data | -2.96 | -10.44 | -4.67 | -7.96 |
| merged_v2_val_clean | -0.18 | -0.75 | 0.39 | -0.19 |

## Rollback policy

- If `three_set_avg` degrades by more than 0.5 points, keep V10 and rollback.
- Analyze degradation sources (domain, length bucket, unknown-source mix) before retrying.
