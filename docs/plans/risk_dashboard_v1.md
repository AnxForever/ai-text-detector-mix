# Risk Dashboard v1

- Generated at: 2026-02-12 19:23:17
- Train set: `C:\datacollection\datasets\merged_v2\train_v10.csv`

## 1) Basic stats

- Rows: 62,980
- Human / AI: 30,165 / 32,815
- AI ratio: 52.104%

## 2) Source risk

- Unique sources: 128
- Unknown source rows: 3,783 (6.007%)

## 3) Length bucket risk

| bucket | count | human | ai | ai_ratio | risk |
|---|---:|---:|---:|---:|---|
| 0-64 | 3,295 | 2,986 | 309 | 9.378% | high_bias |
| 64-128 | 8,751 | 4,622 | 4,129 | 47.183% | balanced |
| 128-256 | 16,096 | 10,762 | 5,334 | 33.139% | balanced |
| 256-512 | 10,287 | 4,477 | 5,810 | 56.479% | balanced |
| 512-1024 | 8,836 | 4,323 | 4,513 | 51.075% | balanced |
| 1024-2048 | 10,681 | 2,532 | 8,149 | 76.294% | medium_bias |
| 2048+ | 5,034 | 463 | 4,571 | 90.803% | high_bias |

## 4) Duplicate and near-duplicate risk (A1)

- Exact duplicates: 0 (0.0%)
- Normalized duplicates: 62 (0.098%)
- Prefix200 duplicates: 17 (0.027%)
- Normalized prefix200 duplicates: 72 (0.114%)

## 5) Template leakage

- Hard pattern rows: 750 (1.191%)
- Soft pattern rows: 12,158 (19.305%)

## 6) Train vs fair_test overlap (A1)

| eval set | rows | exact | prefix200 | normalized_prefix200 |
|---|---:|---:|---:|---:|
| core_v1_test_clean | 545 | 0 | 0 | 0 |
| independent_data | 910 | 0 | 0 | 1 |
| merged_v2_val_clean | 1,144 | 0 | 0 | 1 |

## 7) Weak domains

| source | accuracy | count |
|---|---:|---:|
| real_ai_llama-3.1-405b-instruct | 88.89 | 9 |
| formal_collected | 92.5 | 200 |

## 8) Top-level risk flags

- `unknown_source_ratio_high`
- `length_bucket_bias_high`
- `template_leakage_present`
- `weak_domain_present`
