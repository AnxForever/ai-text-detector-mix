# Workspace Structure Guide

> Updated: 2026-02-13
> Scope: AI text detection thesis project (single-text detection first)

## 1) What is Active Now

These paths are the current working surface for training, evaluation, and API:

- `api/`: inference API and runtime config
  - `api/api.py` default model points to V11c
- `models/bert_v11c_boundary_fix/`: current production-recommended model
- `datasets/merged_v2/`: main training/evaluation data pool
- `scripts/training/`: training entry scripts (focus on v10/v11 line)
- `scripts/evaluation/`: evaluation, calibration, regression gate
- `scripts/data_cleaning/`: dataset build/clean scripts for active line
- `docs/project/`: thesis-facing status and result documents
- `docs/plans/`: experiment plans and decision records

## 2) Archive Zones (Do Not Use in Default Pipeline)

- `archive/cleanup_2026-02-13/`
  - failed V11d/V11d2 model and dataset artifacts moved here
  - phase-2 archived legacy checkpoints in `models_phase2/`
- `archive/configs_legacy_2026-02-13/`
  - archived one-off generation config snapshots
- `docs/plans/archive_failed_v11d/`
  - failed V11d/V11d2 build/handoff notes
- `archive/`
  - historical scripts/temp files from earlier iterations

If you need historical comparison, read from archive. Do not reuse archive paths as
training defaults.

## 3) Directory Purpose by Layer

- `datasets/`: raw, processed, merged, and evaluation data
- `models/`: trained checkpoints and evaluation outputs
- `scripts/`: data -> train -> eval pipelines
- `docs/`: process, decisions, thesis materials
- `configs/`: generation/training config snapshots
- `logs/`: runtime/training logs

## 4) Cleanup Rules

- Rule 1: failed experiments move to archive first, never hard-delete directly.
- Rule 2: only promote a model to active paths after regression gate pass.
- Rule 3: each cleanup must include a manifest with moved paths and restore commands.
- Rule 4: delete archive files only after explicit retention decision.

## 5) Recommended Working Convention

- Keep one active production model line (`v11c`) in runtime defaults.
- Keep experimental variants in clearly named folders, then archive if rejected.
- Update these docs when paths/status change:
  - `docs/project/DOCS_INDEX.md`
  - `docs/project/DATA_AND_MODELS.md`
  - `archive/*/CLEANUP_MANIFEST.md` (when cleanup is executed)
