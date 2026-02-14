# V11 Four-Way Comparison: V10 vs V11a vs V11b vs V11c

Generated: 2026-02-13

## 1. Overview

| Model | Train Data | Strategy | Rows | Gate |
|-------|-----------|----------|------|------|
| V10 (baseline) | train_v10.csv | V9 + targeted augmentation | 62,980 | - |
| V11a | train_v11a_candidate.csv | V10 - hard patterns - unknown | 60,456 | **FAIL** (-0.61%) |
| V11b | train_v11b_candidate.csv | V11a + 600 weak-domain | 61,056 | PASS (+0.22%) |
| **V11c** | train_v11c_candidate.csv | V11b + 2131 long-AI boundary fix | 63,187 | **PASS (+0.20%)** |

## 2. Three-Set Accuracy

| Eval Set (samples) | V10 | V11a | V11b | **V11c** |
|--------------------|-----|------|------|---------|
| core_v1_test_clean (545) | **98.35%** | 97.43% | 97.98% | 97.98% |
| independent_data (910) | 97.69% | 97.03% | 98.46% | **98.57%** |
| merged_v2_val_clean (1144) | 99.04% | 98.78% | **99.30%** | 99.13% |
| **Three-set average** | 98.36% | 97.75% | **98.58%** | 98.56% |

## 3. Independent Data Error Count

| Model | FN | FP | Total Errors |
|-------|----|----|-------------|
| V10 | 3 | 18 | 21 |
| V11a | 1 | 26 | 27 |
| V11b | 1 | 13 | 14 |
| **V11c** | **2** | **11** | **13** |

## 4. AI Model Detection Rates (independent_data)

| AI Source | Count | V10 | V11a | V11b | V11c |
|-----------|-------|-----|------|------|------|
| GPT-5 | 8 | 100% | 100% | 100% | 100% |
| GPT-4 | 10 | 100% | 100% | 100% | 100% |
| GPT-OSS-120B | 8 | 100% | 100% | 100% | 100% |
| DeepSeek-v3.2 | 8 | 100% | 100% | 100% | 100% |
| GLM-4.7 | 9 | 100% | 100% | 100% | 100% |
| Gemini-3-flash | 16 | 100% | 100% | 100% | 100% |
| Gemini-3-pro | 24 | 100% | 100% | 100% | 100% |
| Gemini-3-pro-search | 8 | 100% | 100% | 100% | **87.5%** |
| LLaMA-3.1-405B | 9 | **88.9%** | 100% | 100% | **100%** |
| m4_chatgpt | 50 | 98% | 98% | 98% | 98% |

## 5. Human Source FP Rates (independent_data)

| Human Source | Count | V10 | V11a | V11b | V11c |
|-------------|-------|-----|------|------|------|
| formal_collected | 200 | 96.0% | 90.0% | 95.0% | **96.5%** |
| external_m4_qazh | 49 | 95.9% | 98.0% | 98.0% | **95.9%** |
| Wikipedia_CN | 119 | 99.2% | 99.2% | 99.2% | 99.2% |
| Toutiao_News | 221 | 100% | 99.1% | 100% | 100% |

## 6. Temperature Calibration

| Model | Optimal T | ECE (calibrated) |
|-------|-----------|-----------------|
| V10 | 0.8931 | 0.0058 |
| V11a | 0.8987 | 0.0078 |
| V11b | 0.8226 | 0.0031 |
| **V11c** | **0.8165** | **0.0034** |

## 7. Analysis

### V11c Strengths
- **Best independent_data accuracy** (98.57%) across all four models
- **Fewest total errors** on independent set (13 vs V10's 21)
- **formal_collected recovery**: 96.5% (best, V11a dropped to 90%)
- **LLaMA-405B restored to 100%** (V10 had persistent 88.9% miss)
- Clean data pipeline: risk patterns removed, unknowns triaged, weak domains supplemented, long-AI boundary restored
- Passes regression gate with +0.20% improvement

### V11c vs V11b Trade-offs
- Three-set avg V11c 98.56% vs V11b 98.58% (delta: -0.02%, negligible)
- V11c has +0.11% on independent but -0.17% on merged_v2_val
- V11c has 1 fewer error on independent (13 vs 14)
- V11c has Gemini-3-pro-search regression (87.5% vs 100%, 1 sample)

### V11c vs V10 Trade-offs
- core_v1_test_clean: -0.37% (2 more errors on long thucnews, acceptable)
- Gemini-3-pro-search: 87.5% vs 100% (1/8 new FN)
- independent_data: +0.88% (8 fewer errors)

## 8. Recommendation

**Promote V11c (`bert_v11c_boundary_fix`)** as the new production model.

Rationale:
1. Passes regression gate (+0.20% vs V10 baseline)
2. Best independent_data accuracy (98.57%) — the most representative real-world test
3. Fewest errors on independent set (13, down from V10's 21, -38%)
4. Fixes persistent LLaMA-405B miss (88.9% → 100%)
5. Best formal_collected detection (96.5%) — V11a's main failure point recovered
6. Clean, auditable data pipeline with risk governance throughout
7. Well-calibrated (T=0.8165, ECE=0.0034)

Deployment config:
```python
MODEL_PATH = "models/bert_v11c_boundary_fix"
MAX_LENGTH = 256
TEMPERATURE = 0.8165
```
