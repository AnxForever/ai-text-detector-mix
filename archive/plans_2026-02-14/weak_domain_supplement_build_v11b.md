# V11b Weak Domain Supplement Build

- Generated at: 2026-02-12 19:40:49
- Base train: `C:\datacollection\datasets\merged_v2\train_v11_candidate.csv`
- Supplement rows: 600
- Train rows: 60456 -> 61056

## Domain targets

| source | target | selected | pool_usable | unique_prefix_ratio | top_prefix_share |
|---|---:|---:|---:|---:|---:|
| formal_collected | 300 | 300 | 4820 | 1.0 | 0.0033 |
| real_ai_llama-3.1-405b-instruct | 300 | 300 | 7645 | 1.0 | 0.0033 |

## Leakage checks

- Supplement overlap with base train: 0
- Supplement overlap with fair_test: 0

## Notes

- formal_collected supplement is sourced from non-fair pools and tagged for weak-domain coverage.
- LLaMA supplement uses `model` filter `llama-3.1-405b-instruct` and excludes fair/test overlap.
