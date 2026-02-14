# V11d Gemini Patch Build

- Generated at: 2026-02-13 13:50:28
- Base train: `C:\datacollection\datasets\merged_v2\train_v11c_candidate.csv`
- Train rows: 63187 -> 63427
- Supplement rows: 240
  - AI rows: 120
  - Human rows: 120

## AI supplement

- search rows selected: 30
- length buckets: {'128-256': 118, '64-128': 2}
- style score dist: {'3': 72, '2': 33, '4': 12, '1': 3}

## Human supplement

- length buckets: {'64-128': 83, '128-256': 37}
- style score dist: {'3': 103, '2': 17}

## Leakage checks

- overlap with base train: 0
- overlap with fair_test: 0

## Diversity

- AI: {'rows': 120, 'unique_prefix_ratio': 1.0, 'top_prefix_share': 0.0083}
- Human: {'rows': 120, 'unique_prefix_ratio': 1.0, 'top_prefix_share': 0.0083}
