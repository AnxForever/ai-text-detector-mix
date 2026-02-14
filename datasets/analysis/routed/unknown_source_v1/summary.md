# unknown source triage v1

- generated_at: 2026-02-12 18:52:37
- input: `C:\datacollection\datasets\merged_v2\train_v10.csv`
- unknown_rows: 3,783

## bucket stats

| bucket | rows | ratio |
|---|---:|---:|
| review_needed | 1,767 | 46.709% |
| keep_verified | 1,637 | 43.273% |
| drop_candidate | 379 | 10.019% |

## top reasons

| reason | count |
|---|---:|
| soft:markdown_list | 1,674 |
| clean | 1,637 |
| soft:heading_style | 670 |
| hard:instruction_leak_cn | 261 |
| hard:assistant_meta_cn | 212 |
| soft:markdown_table | 129 |
| soft:markdown_code | 25 |
| soft:excessive_punctuation | 15 |
| hard:roleplay_prompt | 8 |
