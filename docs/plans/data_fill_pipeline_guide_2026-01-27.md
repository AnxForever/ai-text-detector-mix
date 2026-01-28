# Data Fill Pipeline Guide (Dry-run)

This guide explains how to use the data fill pipeline script without
calling any external APIs.

## Files

- Script: `scripts/generation/data_fill_pipeline.py`
- Config template: `configs/data_fill_pipeline_template.json`
- Config (quota-aligned): `configs/data_fill_pipeline_targets_2026-01-27.json`

## Dry-run (no output files)

```
py scripts/generation/data_fill_pipeline.py --config configs/data_fill_pipeline_template.json
```

## Execute (write plan files)

```
py scripts/generation/data_fill_pipeline.py --config configs/data_fill_pipeline_targets_2026-01-27.json --execute
```

Outputs will be written under `datasets/planning/data_fill_runs/<run_name>_<timestamp>/`.

## Notes

- This script only writes plan files. It does not generate data.
- Set `ai_generation.enabled=false` until API access is ready.
- Update `targets` to match the latest quota table (use the quota-aligned config as baseline).
