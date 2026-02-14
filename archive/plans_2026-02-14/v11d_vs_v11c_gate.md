# V11 Regression Gate

- Generated at: 2026-02-13 15:09:11
- Baseline: `bert_v11c_boundary_fix` from `models\bert_v11c_boundary_fix\eval_comparison.json`
- Candidate: `bert_v11d_gemini_patch` from `models\bert_v11d_gemini_patch\eval_comparison.json`

## Gate result

- Decision: **keep_candidate**
- Pass gate: `True`
- Reason: three_set_avg degradation 0.2700 <= threshold 0.5000
- Baseline three_set_avg: 98.56
- Candidate three_set_avg: 98.29
- Delta (candidate - baseline): -0.27

## Per-set deltas

| dataset | accuracy_delta | precision_delta | recall_delta | f1_delta |
|---|---:|---:|---:|---:|
| core_v1_test_clean | -0.18 | -0.29 | 0.0 | -0.15 |
| independent_data | -0.55 | -3.32 | 0.66 | -1.49 |
| merged_v2_val_clean | -0.09 | -0.19 | 0.0 | -0.1 |

## Rollback policy

- If `three_set_avg` degrades by more than 0.5 points, keep V10 and rollback.
- Analyze degradation sources (domain, length bucket, unknown-source mix) before retrying.
