# Claude Handoff - V11d Gemini Patch (2026-02-13)

## What Codex prepared

### Data artifacts
- `datasets/merged_v2/v11d_gemini_style_patch.csv`
- `datasets/merged_v2/train_v11d_candidate.csv`
- `datasets/merged_v2/train_v11d_candidate_summary.json`
- `docs/plans/v11d_gemini_patch_build.md`

### Build script
- `scripts/data_cleaning/build_v11d_gemini_patch.py`

### Training wrapper
- `scripts/training/train_v11d.py`

## Patch snapshot
- Base train: `train_v11c_candidate.csv` (63187 rows)
- V11d train: `train_v11d_candidate.csv` (63427 rows)
- Added rows: 240 (AI 120 + Human 120)
- Leakage checks: base overlap = 0, fair_test overlap = 0

AI supplement details:
- Search-model rows: 30
- Other Gemini-family rows: 90
- Length: 128-256 = 118, 64-128 = 2
- Model mix includes:
  - `hyb-optimal/gemini-3-pro-preview-search` (30)
  - `gemini-3-flash-preview` (20)
  - `gemini-3-pro-preview-bs` (15)

Human supplement details:
- Source: `LCSTS-news`
- Length: 64-128 = 83, 128-256 = 37

## Recommended next commands (Claude side)

1) Train V11d (fast validation first, then full retrain only if needed)
```bash
py -3 scripts/training/train_v11d.py --mode fast
```

```bash
# optional full retrain for final report after fast mode passes
py -3 scripts/training/train_v11d.py --mode full
```

2) Three-set evaluation
```bash
py -3 scripts/evaluation/eval_v11_single.py --model bert_v11d_gemini_patch
```

3) Temperature calibration (mandatory rerun)
```bash
py -3 scripts/evaluation/calibrate_temperature.py --model-path models/bert_v11d_gemini_patch
```

4) Regression gate (vs V11c and V10)
```bash
py -3 scripts/evaluation/check_v11_regression_gate.py \
  --baseline-json models/bert_v11c_boundary_fix/eval_comparison.json \
  --candidate-json models/bert_v11d_gemini_patch/eval_comparison.json \
  --baseline-key bert_v11c_boundary_fix \
  --candidate-key bert_v11d_gemini_patch \
  --output-json docs/plans/v11d_vs_v11c_gate.json \
  --output-md docs/plans/v11d_vs_v11c_gate.md
```

```bash
py -3 scripts/evaluation/check_v11_regression_gate.py \
  --baseline-json models/bert_v10_augmented/eval_comparison.json \
  --candidate-json models/bert_v11d_gemini_patch/eval_comparison.json \
  --baseline-key bert_v10_augmented \
  --candidate-key bert_v11d_gemini_patch \
  --output-json docs/plans/v11d_vs_v10_gate.json \
  --output-md docs/plans/v11d_vs_v10_gate.md
```

5) Error diff focus (Gemini-search)
```bash
py -3 scripts/evaluation/error_diff_v10_v11c.py
# then compare with V11d run by cloning script or extending it to V11d
```

## Acceptance criteria (same as agreed)
- Three-set average drop vs V11c <= 0.2%
- `real_ai_gemini-3-pro-preview-search` recovers to 8/8 on current fair split
- No material FP increase on `thucnews` 512+ slice
- If any criterion fails: keep V11c
