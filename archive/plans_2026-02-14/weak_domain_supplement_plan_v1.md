# Weak Domain Supplement Plan v1

- Generated at: 2026-02-12 19:20:59
- Eval comparison: `C:\datacollection\models\bert_v10_augmented\eval_comparison.json`
- Train baseline: `C:\datacollection\datasets\merged_v2\train_v11_candidate.csv`
- Weak threshold: 95.0
- Minimum per weak domain: 300
- Total required new samples: 600

| source | acc | eval_count | current_train | target | required_new | uniq_prefix_ratio | top_prefix_share |
|---|---:|---:|---:|---:|---:|---:|---:|
| real_ai_llama-3.1-405b-instruct | 88.89 | 9 | 0 | 300 | 300 | 0.0 | 0.0 |
| formal_collected | 92.5 | 200 | 0 | 300 | 300 | 0.0 | 0.0 |

## Diversity constraints

- min_unique_prefix_ratio >= 0.7
- max_top_prefix_share <= 0.2
- If generated data is used, use at least 3 prompt families per weak domain.
