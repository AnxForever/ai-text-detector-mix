# V11c Long-AI Boundary Fix

- Generated at: 2026-02-13 10:24:52
- Base train: `C:\datacollection\datasets\merged_v2\train_v11b_candidate.csv`
- Reference train: `C:\datacollection\datasets\merged_v2\train_v10.csv`
- Supplement rows: 2131
- Train rows: 61056 -> 63187

## Bucket restoration

| bucket | reference_ai | base_ai | target_ai | added | after_ai |
|---|---:|---:|---:|---:|---:|
| 0-64 | 309 | 308 | 309 | 0 | 308 |
| 64-128 | 4129 | 4123 | 4129 | 0 | 4123 |
| 128-256 | 5334 | 5332 | 5334 | 0 | 5332 |
| 256-512 | 5810 | 5328 | 5810 | 482 | 5810 |
| 512-1024 | 4513 | 3531 | 4513 | 982 | 4513 |
| 1024-2048 | 8149 | 7615 | 8149 | 534 | 8149 |
| 2048+ | 4571 | 4438 | 4571 | 133 | 4571 |

## Leakage checks

- overlap with base train: 0
- overlap with fair_test: 0

## Model-family distribution (supplement)

- deepseek: 366
- glm: 365
- gpt4: 365
- llama: 364
- gpt-oss: 343
- gpt5: 192
- gemini: 136
