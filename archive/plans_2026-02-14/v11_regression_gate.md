# V11 Regression Gate

- Generated at: 2026-02-12 19:21:36
- Baseline: `bert_v10_augmented` from `models\bert_v10_augmented\eval_comparison.json`
- Candidate: `bert_v10_augmented` from `models\bert_v10_augmented\eval_comparison.json`

## Gate result

- Decision: **keep_candidate**
- Pass gate: `True`
- Reason: three_set_avg degradation 0.0000 <= threshold 0.5000
- Baseline three_set_avg: 98.36
- Candidate three_set_avg: 98.36
- Delta (candidate - baseline): 0.0

## Per-set deltas

| dataset | accuracy_delta | precision_delta | recall_delta | f1_delta |
|---|---:|---:|---:|---:|
| core_v1_test_clean | 0.0 | 0.0 | 0.0 | 0.0 |
| independent_data | 0.0 | 0.0 | 0.0 | 0.0 |
| merged_v2_val_clean | 0.0 | 0.0 | 0.0 | 0.0 |

## Rollback policy

- If `three_set_avg` degrades by more than 0.5 points, keep V10 and rollback.
- Analyze degradation sources (domain, length bucket, unknown-source mix) before retrying.
