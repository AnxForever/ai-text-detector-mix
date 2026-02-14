# Risk Governance Addendum (2026-02-12)

This addendum aligns the V10 -> V11 risk-governance workflow with external review feedback.

## Scope adjustments

1. A1 data health checks expanded
   - Add exact-duplicate and near-duplicate metrics to the dashboard.
   - Keep train-vs-fair_test overlap checks in A1 (not deferred to final testing).

2. B2 weak-domain supplement minimum raised
   - Minimum sample target per weak domain: `>= 300`.
   - Add diversity constraints to avoid single-template generation.

3. API observability fields decoupled from data governance
   - `riskFlags`, `domainHint`, `modelVersion`, and `decisionThreshold` are now
     controlled by `DETECTOR_INCLUDE_RISK_OBSERVABILITY`.
   - Default is off to keep the main data-governance iteration focused.

4. C2 temperature calibration is mandatory for V11
   - Always rerun `scripts/evaluation/calibrate_temperature.py` after V11 training.
   - Do not reuse V10 calibration (`temperature=0.8931`) without re-calibration.

5. Rollback gate added
   - If V11 three-set average degrades by more than `0.5` points, rollback to V10.
   - Use `scripts/evaluation/check_v11_regression_gate.py` for a reproducible decision.

## New/updated scripts

- `scripts/analysis/generate_risk_dashboard.py`
  - Added duplicate / near-duplicate metrics.
  - Added normalized-prefix overlap checks.

- `scripts/analysis/plan_weak_domain_supplement.py`
  - Computes weak-domain supplement targets with `min_per_weak_domain=300`.
  - Includes diversity targets (`unique_prefix_ratio`, `top_prefix_share`).

- `scripts/data_cleaning/build_v11b_with_weak_domain_supplement.py`
  - Builds `weak_domain_supplement_v11b.csv` and `train_v11b_candidate.csv`.
  - Enforces no overlap with base train and fair_test by hash exclusion.

- `scripts/evaluation/check_v11_regression_gate.py`
  - Implements rollback rule (`degradation > 0.5` => rollback).

## Recommended command sequence

```bash
# A1: risk dashboard (includes dedup + overlap)
py -3 scripts/analysis/generate_risk_dashboard.py

# B1: unknown-source triage
py -3 scripts/analysis/triage_unknown_source.py

# B2: weak-domain supplement planning (min 300 + diversity)
py -3 scripts/analysis/plan_weak_domain_supplement.py

# B3: build V11 candidate set
py -3 scripts/data_cleaning/build_train_v11_candidate.py

# B2 implementation: inject 300+300 weak-domain rows into V11b
py -3 scripts/data_cleaning/build_v11b_with_weak_domain_supplement.py

# C2: mandatory calibration after V11 training
py -3 scripts/evaluation/calibrate_temperature.py --model bert_v11_candidate

# Rollback gate
py -3 scripts/evaluation/check_v11_regression_gate.py \
  --baseline-json models/bert_v10_augmented/eval_comparison.json \
  --candidate-json models/bert_v11_candidate/eval_comparison.json \
  --baseline-key bert_v10_augmented \
  --candidate-key bert_v11_candidate
```
