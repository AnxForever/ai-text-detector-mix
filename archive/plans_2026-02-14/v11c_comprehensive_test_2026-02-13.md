# V11c Comprehensive Test Report (2026-02-13)

## 1. Model Under Test

- Model: `models/bert_v11c_boundary_fix`
- Runtime config: `MAX_LENGTH=256`, `TEMPERATURE=0.8165`
- Test goal: validate single-text detection quality for common real-world scenarios

## 2. Three Fair-Test Sets (Core KPI)

Source run:

- `py -3 scripts/evaluation/eval_v11_single.py --model bert_v11c_boundary_fix`

Results:

| Eval Set | Accuracy | Precision | Recall | F1 | Samples |
|---|---:|---:|---:|---:|---:|
| core_v1_test_clean | 97.98% | 97.87% | 98.77% | 98.32% | 545 |
| independent_data | 98.57% | 93.08% | 98.67% | 95.79% | 910 |
| merged_v2_val_clean | 99.13% | 98.07% | 100.00% | 99.03% | 1144 |
| **Three-set average** | **98.56%** | - | - | - | - |

Independent error split:

- FN = 2
- FP = 11
- Total errors = 13

## 3. Independent Source-Level Detection

Strong:

- `real_ai_llama-3.1-405b-instruct`: 9/9 (100%)
- `real_ai_gpt-4`: 10/10 (100%)
- `real_ai_gpt-5`: 8/8 (100%)
- `real_ai_gemini-3-pro-preview`: 24/24 (100%)

Known weak point:

- `real_ai_gemini-3-pro-preview-search`: 7/8 (87.5%)

Human-source caution point:

- `formal_collected`: 96.5% (still lower than major news sources)

Machine-readable output:

- `docs/plans/v11c_comprehensive_eval_2026-02-13.json`

## 4. Calibration (Confidence Quality)

Source run:

- `py -3 scripts/evaluation/calibrate_temperature.py --model-path models/bert_v11c_boundary_fix`

Recommended temperature:

- `T = 0.8165`

ECE before/after:

| Eval Set | ECE Before | ECE After | Delta |
|---|---:|---:|---:|
| core_v1_test_clean | 0.0100 | 0.0082 | -0.0019 |
| independent_data | 0.0168 | 0.0017 | -0.0151 |
| merged_v2_val_clean | 0.0195 | 0.0054 | -0.0142 |

Output:

- `datasets/eval/fair_test/calibration_results.json`

## 5. Data Leakage Quick Check (Targeted to V11c Train Set)

Method:

- SHA1 exact-overlap check between `datasets/merged_v2/train_v11c_candidate.csv`
  and the three fair-test sets.

Result:

| Eval Set | Unique Samples | Exact Overlap | Rate |
|---|---:|---:|---:|
| core_v1_test_clean | 545 | 0 | 0.0000% |
| independent_data | 910 | 1 | 0.1099% |
| merged_v2_val_clean | 1140 | 1 | 0.0877% |

## 6. Spot Inference Sanity Check

Source run:

- `py -3 scripts/evaluation/test_single_text.py --model-dir models/bert_v11c_boundary_fix --file .tmp_v11c_samples.txt`

Observation:

- 4 sample texts tested (2 news-like / 2 structured-AI-like)
- Predictions: 2 Human, 2 AI
- Average confidence: 97.38%

## 7. Conclusion

V11c remains the best available production model for current single-text goal:

- Three-set average is stable at **98.56%**
- Independent set reaches **98.57%**
- Calibration quality is strong with `T=0.8165`
- Main residual risk remains **Gemini-search style** (7/8)

Decision:

- Keep `bert_v11c_boundary_fix` as production model.
