# V11d Gemini Patch Build

- Generated at: 2026-02-13 15:37:02
- Preset: v11d2
- Base train: `C:\datacollection\datasets\merged_v2\train_v11c_candidate.csv`
- Train rows: 63187 -> 63427
- Supplement rows: 240
  - AI rows: 80
  - Human rows: 160

## AI supplement

- target search rows: 24
- search rows selected: 24
- length buckets: {'128-256': 78, '64-128': 2}
- style score dist: {'3': 52, '2': 15, '4': 12, '1': 1}

## Human supplement

- source quota requested: {'LCSTS-news': 70, 'external_m4_qazh': 70, '模板-通知': 20}
- length buckets: {'128-256': 81, '64-128': 79}
- style score dist: {'2': 71, '3': 57, '1': 23, '0': 9}

## Leakage checks

- overlap with base train: 0
- overlap with fair_test: 0

## Diversity

- AI: {'rows': 80, 'unique_prefix_ratio': 1.0, 'top_prefix_share': 0.0125}
- Human: {'rows': 160, 'unique_prefix_ratio': 1.0, 'top_prefix_share': 0.0063}
