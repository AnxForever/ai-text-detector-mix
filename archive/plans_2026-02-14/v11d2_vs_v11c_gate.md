# V11 Regression Gate

- Generated at: 2026-02-13 16:29:59
- Baseline: `bert_v11c_boundary_fix` from `models\bert_v11c_boundary_fix\eval_comparison.json`
- Candidate: `bert_v11d2_gemini_patch_fast` from `models\bert_v11d2_gemini_patch_fast\eval_comparison.json`

## Gate result

- Decision: **rollback_to_baseline**
- Pass gate: `False`
- Reason: three_set_avg degradation 0.6000 > threshold 0.5000
- Baseline three_set_avg: 98.56
- Candidate three_set_avg: 97.96
- Delta (candidate - baseline): -0.6

## Per-set deltas

| dataset | accuracy_delta | precision_delta | recall_delta | f1_delta |
|---|---:|---:|---:|---:|
| core_v1_test_clean | -0.37 | -0.59 | 0.0 | -0.3 |
| independent_data | -1.43 | -7.45 | 0.66 | -3.81 |
| merged_v2_val_clean | 0.0 | 0.0 | 0.0 | 0.0 |

## Rollback policy

- If `three_set_avg` degrades by more than 0.5 points, keep V10 and rollback.
- Analyze degradation sources (domain, length bucket, unknown-source mix) before retrying.
