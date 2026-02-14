# Codex + ClaudeCode Sync (V11a/V11b)

## Roles

- Codex: data governance pipeline, supplement build, leakage gates, reproducible artifacts.
- ClaudeCode: model training, calibration decision, final metric interpretation.

## Ready-to-use datasets

- V11a train set: `datasets/merged_v2/train_v11_candidate.csv` (60,456 rows)
- V11b train set: `datasets/merged_v2/train_v11b_candidate.csv` (61,056 rows)
- Weak-domain supplement: `datasets/merged_v2/weak_domain_supplement_v11b.csv` (600 rows)

## Codex completed

- Added weak-domain build script:
  - `scripts/data_cleaning/build_v11b_with_weak_domain_supplement.py`
- Built 300 + 300 weak-domain rows with leakage exclusion:
  - `formal_collected`: 300
  - `real_ai_llama-3.1-405b-instruct`: 300
- Leakage checks:
  - overlap with base train: 0
  - overlap with fair_test: 0

## ClaudeCode next actions

1. C1a: train `bert_v11a` using `train_v11_candidate.csv`
2. C2a: calibrate `bert_v11a`
3. C3a: regression gate vs V10
4. C1b: train `bert_v11b` using `train_v11b_candidate.csv`
5. C2b: calibrate `bert_v11b`
6. C3b: compare V10 vs V11a vs V11b and decide keep/rollback

## Gate tools

- Calibration:
  - `scripts/evaluation/calibrate_temperature.py --model bert_v11_candidate`
  - or `--model-path models/<actual_model_dir>`
- Rollback gate (`>0.5` degradation -> rollback):
  - `scripts/evaluation/check_v11_regression_gate.py`
- Fair-test gate JSON builder:
  - `scripts/evaluation/eval_fair_models_for_gate.py`
  - baseline JSON already prepared: `docs/plans/eval_comparison_v10_baseline_for_gate.json`
