# Cleanup Phase-2 Candidates

> Updated: 2026-02-13
> Status: model archive and config archive completed

## Why this file

Phase-1 already removed failed V11d/V11d2 artifacts from active paths.
Phase-2 is for larger disk cleanup decisions that need explicit retention policy.

## Completed: old model checkpoints archived

Moved from `models/` to `archive/cleanup_2026-02-13/models_phase2/`:

- `models/bert_v2_balanced`
- `models/bert_v3_core_v2`
- `models/bert_v3_fresh`
- `models/bert_v4_defense_focused`
- `models/bert_v5_paired`
- `models/bert_v6_merged`
- `models/bert_v7_improved`
- `models/bert_v8_calibrated`
- `models/bert_v9_p0_supplement`
- `models/bert_v10_augmented`
- `models/bert_v11a_clean`
- `models/bert_v11b_augmented`

Keep:

- `models/bert_v11c_boundary_fix` (current production baseline)

Archived volume: about `4.57 GB` moved out of active `models/`.

Details:

- `archive/cleanup_2026-02-13/phase2_models_move_report.json`

## Completed: one-off config files in `configs/`

Moved one-off generation configs to:

- `archive/configs_legacy_2026-02-13/`

Kept in active `configs/`:

- `data_fill_pipeline_template.json`
- `data_fill_pipeline_targets_2026-01-27.json`
- `scenario_fill_smoke.json`
- `scenario_fill_smoke_glm.json`
- `scenario_fill_smoke_qwen.json`
- `p0_prompts.json`
- `test_f_spec_013_fix.json`
- `README.md`

Move report:

- `archive/cleanup_2026-02-13/phase2_configs_move_report.json`

## Suggested execution order

1. Freeze "must-keep for thesis" list.
2. Move selected config files to archive (not hard delete). (done)
3. Re-run a smoke check:
   - API starts
   - V11c inference works
   - main eval scripts run
4. Hard delete archive only after final thesis hand-in backup is made.
