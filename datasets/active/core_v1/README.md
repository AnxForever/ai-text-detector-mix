# Core V1 Dataset

This dataset is the merged, de-duplicated, and re-split core training set.
It was built from source inputs that have been moved to:

- `datasets/archive/_to_delete/sources_20260127`

After exact-text de-duplication:

- All samples from `final_clean_phrase_clean` were found in `combined_v2_clean`
  with the same label.
- No label conflicts were found.

Splits (80/10/10 stratified by label):
- train: 46,849
- val: 5,856
- test: 5,858

Metadata:
- `merge_log.json` records inputs and counts
- Conflicts (if any) are stored at `datasets/analysis/metadata/core_v1_label_conflicts.csv`
- `full_dataset.csv` combines all splits
- `all_human.csv` / `all_ai.csv` are label-specific subsets
