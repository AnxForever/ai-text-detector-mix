# Datasets Index (Organized)

All datasets are now grouped by the current plan. No duplicate files are kept
when identical copies exist in evaluation splits.

## Active training (core)

- `datasets/active/core_v1`

## Evaluation splits

- `datasets/eval/splits/v1`

## Mixed-text

- Candidates: `datasets/mixed/candidates`
- Sources: `datasets/mixed/hybrid`

## Analysis outputs

- Classified: `datasets/analysis/classified`
- Routed pools: `datasets/analysis/routed`
- Prediction outputs: `datasets/analysis/pred_probs`
- Audit/metadata: `datasets/analysis/metadata`

## Raw sources

- `datasets/raw`

## Archive (legacy datasets)

- `datasets/archive/combined_v2`
- `datasets/archive/final_clean`

## Archive (pending delete)

- `datasets/archive/_to_delete/sources_20260127`

## Planning outputs

- `datasets/planning/data_fill_runs`

## Logs

- `datasets/logs/augmented_v2`

## Samples

- `datasets/samples/schema_sample`

## Dedup summary

- core_v1 is built by merging `combined_v2_clean` and `final_clean_phrase_clean`.
- Exact-text dedup found `final_clean_phrase_clean` fully contained in `combined_v2_clean`.
- No label conflicts found.

## Duplicate removal

- Removed from candidates (kept in eval splits):
  - `mixed_test_balanced_by_category.csv`
  - `mixed_test_balanced_by_category_length.csv`
