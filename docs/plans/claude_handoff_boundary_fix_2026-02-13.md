# Claude Handoff: THUCNews 512+ FP Boundary Fix

## What was confirmed

- Hard-rule false positive exists (`roleplay_prompt`), but impact is small.
- Main issue is long-AI coverage drop after filtering.

## Codex actions completed

1. Refined hard pattern rules in `scripts/utils/risk_patterns.py`
   - Removed broad roleplay match behavior (no generic `你是一个` style trigger).
2. Built long-AI boundary supplement + merged V11c train set:
   - Script: `scripts/data_cleaning/build_v11c_long_ai_boundary_fix.py`
   - Output supplement: `datasets/merged_v2/long_ai_boundary_supplement_v11c.csv`
   - Output train: `datasets/merged_v2/train_v11c_candidate.csv`
   - Summary: `datasets/merged_v2/train_v11c_boundary_fix_summary.json`

## Key numbers

- Added AI rows: `2131`
- Leakage check:
  - overlap with base train = `0`
  - overlap with fair_test = `0`
- AI bucket restoration (`v11c` vs `v10`, exact match on target buckets):
  - `256-512`: 5810
  - `512-1024`: 4513
  - `1024-2048`: 8149
  - `2048+`: 4571

## Recommended execution order

1. Finish current `v11a` training (already running).
2. Keep planned `v11b` run for ablation.
3. Add `v11c` run with boundary fix dataset:

```bash
python scripts/training/train_v10.py  # or your v11 trainer entry
# replace train input with:
# datasets/merged_v2/train_v11c_candidate.csv
```

4. Run fair-test compare + gate on `v10`, `v11a`, `v11b`, `v11c`.

