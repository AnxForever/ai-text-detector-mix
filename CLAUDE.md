# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Chinese AI-generated text detection system. The current online path is a calibrated
V11c BERT binary classifier that returns Human / AI. Mixed-text boundary detection
exists as historical experimental work, but is not enabled by default because the
mixed dataset is not large or representative enough for stable production use.

**Tech stack**: Python 3.12, PyTorch 2.0+, Transformers 4.30+, FastAPI, BERT-base-chinese (fine-tuned)
**Frontend**: Next.js 16 + React 19 + TailwindCSS 4 (in `frontend/`, independent submodule)

## Common Commands

```bash
# Activate virtualenv (required for all Python commands)
source .venv/bin/activate

# --- Testing ---
pytest                                    # Run all tests (testpaths: tests/, api/tests/)
pytest api/tests/                         # API tests only
pytest -k "test_function_name"            # Single test
pytest --cov --cov-report=term-missing    # With coverage (fail_under=50)

# --- Linting & Type Checking ---
ruff check .                              # Lint (E/W/F/I/UP/T20 rules, line-length=100)
ruff check --fix .                        # Auto-fix lint issues
ruff format .                             # Format code
mypy api/ scripts/                        # Type check (py312, ignore_missing_imports=true)

# --- Training ---
python scripts/training/train_bert_improved.py --epochs 5 --batch_size 16
python scripts/training/train_span_detector.py --epochs 10

# --- Evaluation ---
python scripts/evaluation/eval_complete.py              # Full test set evaluation
python scripts/evaluation/test_single_text.py --interactive  # Interactive single-text test
python scripts/evaluation/comprehensive_eval.py         # Comprehensive evaluation

# --- API Server ---
cd api && python api.py                   # Starts on 0.0.0.0:8000

# --- Frontend ---
cd frontend && pnpm dev                   # Dev server
cd frontend && pnpm build                 # Production build
cd frontend && pnpm lint                  # ESLint

# --- Docker Deployment ---
docker compose up -d                      # Full stack: backend + frontend + nginx (port 80)
```

## Architecture

### Online Pipeline

```
Input Text → Classifier (bert_v11c_boundary_fix) → {Human, AI}
                                                        ↓
                                      Risk flags + sentence-level analysis
```

- **Classifier**: `BertForSequenceClassification` — binary Human/AI classifier, 98.56% three-set average
- **Span Detector**: historical `BertForTokenClassification` experiment; only loads when `DETECTOR_ENABLE_SPAN=1`
- **Temperature Scaling**: T=0.8165, ECE=0.0034 for calibrated confidence scores

### API Endpoints (api/api.py)

| Endpoint | Purpose |
|----------|---------|
| `POST /api/detect` | Main detection: returns Human/AI label, confidence, risk flags, sentence results |
| `POST /v1/chat/completions` | OpenAI-compatible wrapper for integration |
| `GET /api/health` | Health check |

API env vars: `DETECTOR_CLASSIFIER_MODEL`, `DETECTOR_ENABLE_SPAN=0`, `DETECTOR_SPAN_MODEL`, `DETECTOR_MAX_LENGTH=256`, `DETECTOR_TEMPERATURE=0.8165`, `DETECTOR_DECISION_THRESHOLD=0.8`

### Training Pipeline

1. **Data cleaning** (`scripts/data_cleaning/`): current V11c path focuses on template/unknown removal, weak-domain supplementation, and long-AI sample repair; legacy mixed-text scripts remain for reproducibility
2. **Dataset prep**: `scripts/bert_prep/create_bert_dataset.py` → `AIDetectionDataset` + `dynamic_collate_fn`
3. **Training** (`scripts/training/train_bert_improved.py`): `BERTTrainer` class with warmup, label_smoothing=0.05, length-weighted loss
4. **Evaluation** (`scripts/evaluation/`): ID/OOD split evaluation and hard-case analysis via `datasets/eval/splits/v1/`

### Key Data Format

CSV files with columns: `text`, `label` (0=Human, 1=AI), `category`, `source`.
Mixed text categories (C2/C3/C4) are retained as historical experimental data and
should not be described as the current online output.

## Module Responsibilities

Each module with a `CLAUDE.md` contains detailed guidance for that subsystem.

| Path | Role | Entry Point | Module Docs |
|------|------|-------------|-------------|
| `api/` | FastAPI detection service | `api.py` | [api/CLAUDE.md](api/CLAUDE.md) |
| `scripts/` | Training, evaluation, generation, cleaning | — | [scripts/CLAUDE.md](scripts/CLAUDE.md) |
| `datasets/` | All data; registry at `registry.json` | `registry.json` | [datasets/CLAUDE.md](datasets/CLAUDE.md) |
| `models/` | Trained model checkpoints (read-only) | — | [models/CLAUDE.md](models/CLAUDE.md) |
| `docs/` | Project documentation & plans | — | [docs/CLAUDE.md](docs/CLAUDE.md) |
| `frontend/` | Next.js demo UI | `pnpm dev` | [frontend/CLAUDE.md](frontend/CLAUDE.md) |
| `config/` | API runtime config (contains secrets) | `api.local.json` | — |
| `configs/` | Generation task templates | `scenario_fill_*.json` | — |

## Key Models

| Model | Type | Metric |
|-------|------|--------|
| `models/bert_v11c_boundary_fix/` | Classifier | 98.56% (3-set avg), 98.57% (independent eval) |
| `models/bert_span_detector/` | Historical token classifier | disabled by default |

V11c training config: batch_size=8, accum_steps=4, max_length=256, epochs=4, lr=1e-05, label_smoothing=0.05

## Critical Rules

- **Never delete** `models/` — contains trained weights (781MB)
- **Never modify** `datasets/active/core_v1/` — primary training set
- **Secrets**: `config/api.local.json` and `.env.deploy` contain API keys — never commit
- **Offline mode**: Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` to prevent model downloads
- **Latest status doc**: `docs/project/DEFENSE_CURRENT_STATUS.md`
- **Dataset discovery**: Use `datasets/registry.json` (18 entries with metadata)

## Dependencies

| File | Purpose |
|------|---------|
| `requirements_training.txt` | Training: torch, transformers, pandas, scikit-learn, openai |
| `api/requirements_api.txt` | API only: fastapi, uvicorn, torch, transformers |
| `requirements_dev.txt` | Dev: pytest, pytest-cov, ruff, mypy, httpx |

## Code Style

- Line length: 100 chars (ruff enforced)
- Python 3.12, type hints encouraged
- Import order: stdlib > third-party > local (`scripts`, `api` are first-party)
- `T20` rule active: `print()` statements flagged by ruff — use logging instead
- Config in `pyproject.toml` (pytest, ruff, mypy, coverage — single source of truth)

## Recommended Skills

Use these slash commands for this project:

| Skill | When |
|-------|------|
| `/python-patterns` | Writing or reviewing Python code |
| `/python-testing` | Writing tests, TDD workflow |
| `/security-review` | Before commits touching API/auth/secrets |
| `/verification-loop` | Pre-commit validation (lint + test + build) |

## Hooks (Auto-configured)

PostToolUse hooks fire on every `Edit`/`Write` of `.py` files:

| Hook | What it does |
|------|-------------|
| `ruff-fix.sh` | Auto-formats and fixes lint issues |
| `print-detect.sh` | Warns about `print()` statements (T20) |
| `pytest-run.sh` | Runs related tests (test files or api/ changes) |

Hook scripts live in `.claude/hooks/`, configured in `.claude/settings.local.json`.
