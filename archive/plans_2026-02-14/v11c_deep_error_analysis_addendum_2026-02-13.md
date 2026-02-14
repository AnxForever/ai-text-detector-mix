# V11c Deep Error Analysis Addendum (2026-02-13)

## Scope
- Compared `bert_v10_augmented` vs `bert_v11c_boundary_fix` at sample level.
- Focused on:
  - new errors vs fixed errors
  - THUCNews long-human behavior
  - `Gemini-3-pro-preview-search` regression risk

Primary artifact:
- `docs/plans/v11c_error_diff_analysis.json`

## 1) Net Error Movement (Three Fair Sets)
- `core_v1_test_clean`: new=3, fixed=1, net=-2
- `independent_data`: new=4, fixed=12, net=+8
- `merged_v2_val_clean`: new=4, fixed=5, net=+1
- **Total**: new=11, fixed=18, **net=+7** (V11c better overall)

## 2) THUCNews Long-Human Check
- `core_v1_test_clean` thucnews-human:
  - V10=97.39% (112/115), V11c=97.39% (112/115)
  - FP count: V10=3, V11c=3
- `merged_v2_val_clean` thucnews-human:
  - V10=97.92% (94/96), V11c=97.92% (94/96)
  - FP count: V10=2, V11c=2
- Length-slice movement in `merged_v2_val_clean` thucnews-human:
  - 256-512: 90.0% -> 100.0% (+10.0)
  - 512+: 100.0% -> 97.37% (-2.63)
  - net effect cancels at source level.

Conclusion:
- Long-text boundary fix does not worsen THUCNews at aggregate source level.
- Residual 512+ FP risk still exists but is not larger than V10 in total count.

## 3) Gemini-3-pro-search Regression (Focused)
- Source: `real_ai_gemini-3-pro-preview-search`
- Independent set count: n=8
- Accuracy:
  - V10: 8/8 (100%)
  - V11c: 7/8 (87.5%)
- Across all Gemini-family rows in independent set (n=48 AI-only):
  - V10: 48/48 (100%)
  - V11c: 47/48 (97.92%)

Single regressed sample:
- index=803, label=AI, length=155
- Style: short, policy-commentary, highly human-like wording

Model confidence trajectory on that sample (AI probability):
- V10: 0.9762
- V11a: 0.9697
- V11b: 0.8085
- V11c: 0.3898 (crossed decision boundary)

Interpretation:
- This is a narrow-source, small-sample regression.
- Margin collapse started before V11c (already visible in V11b), then became a hard error in V11c.

## 4) Decision Guidance
Given current objective (FN-first on realistic independent set):
- V11c still has strongest practical profile:
  - independent_data best (98.57%)
  - total independent errors lowest (13)
  - LLaMA-405B fixed to 100%
  - regression gate PASS (+0.20% vs V10 baseline)

Recommended release posture:
1. Promote V11c as primary candidate.
2. Mark Gemini-search as a tracked canary slice (n is currently too small for hard conclusions).
3. Start a small V11d patch experiment targeting search-grounded short AI style.

## 5) Minimal V11d Patch Plan (Targeted, Low-Risk)
- Add 120-200 AI samples in "search-grounded policy commentary" style (64-256 chars).
- Add matched 120-200 human samples in same topic/style to preserve boundary.
- Fine-tune from V11c for 1 epoch (low LR) and re-run three-set gate.
- Accept V11d only if:
  - three-set avg does not drop >0.2 vs V11c
  - Gemini-search returns to 8/8 on current fair split
  - no new degradation on THUCNews 512+ slice.
