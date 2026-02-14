# Claude Handoff - V11d2 Minimal Patch (2026-02-13)

## What changed from V11d

Codex implemented a V11d2 preset to reduce over-correction:
- AI supplement reduced: 120 -> 80
- Search AI fixed to exact count: 24
- Human supplement increased: 120 -> 160
- Human source quotas enforced:
  - `LCSTS-news`: 70
  - `external_m4_qazh`: 70
  - `模板-通知`: 20

## Artifacts

### Scripts
- `scripts/data_cleaning/build_v11d_gemini_patch.py`
  - new preset: `--preset v11d2`
  - supports source quota and exact search-AI target
- `scripts/training/train_v11d.py`
  - supports `--mode fast|full`

### Data outputs
- `datasets/merged_v2/v11d2_gemini_patch.csv`
- `datasets/merged_v2/train_v11d2_candidate.csv`
- `datasets/merged_v2/train_v11d2_candidate_summary.json`
- `docs/plans/v11d2_gemini_patch_build.md`

Key checks from summary:
- rows: `63187 -> 63427` (+240)
- AI selected: 80 (search=24, related=56)
- Human selected: 160
- overlap with base train: 0
- overlap with fair_test: 0

## Claude run commands

1) Train fast (recommended first)
```bash
py -3 scripts/training/train_v11d.py \
  --mode fast \
  --train-data datasets/merged_v2/train_v11d2_candidate.csv \
  --output models/bert_v11d2_gemini_patch_fast \
  --learning-rate 3e-6 \
  --epochs 1
```

2) Three-set evaluation
```bash
py -3 scripts/evaluation/eval_v11_single.py --model bert_v11d2_gemini_patch_fast
```

3) Calibration
```bash
py -3 scripts/evaluation/calibrate_temperature.py --model-path models/bert_v11d2_gemini_patch_fast
```

4) Gate vs V11c
```bash
py -3 scripts/evaluation/check_v11_regression_gate.py \
  --baseline-json models/bert_v11c_boundary_fix/eval_comparison.json \
  --candidate-json models/bert_v11d2_gemini_patch_fast/eval_comparison.json \
  --baseline-key bert_v11c_boundary_fix \
  --candidate-key bert_v11d2_gemini_patch_fast \
  --output-json docs/plans/v11d2_vs_v11c_gate.json \
  --output-md docs/plans/v11d2_vs_v11c_gate.md
```

## Acceptance targets (strict)
- Gemini-search: `8/8`
- formal_collected: `>= 95.5%`
- three-set drop vs V11c: `<= 0.10%`
- independent total errors: `<= 15`

If all pass, then optionally run full mode for final frozen release artifact.
