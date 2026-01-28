"""Build dataset registry and classification index.

Creates datasets/README.md and datasets/registry.json to organize current
datasets by category without moving any files.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")


CATEGORY_MAP = {
    "core_v1": "active_train_candidates",
    "final_clean": "legacy_or_archive",
    "combined_v2": "legacy_or_archive",
    "eval": "evaluation_splits",
    "mixed": "mixed_test_sources",
    "analysis": "analysis_metadata",
    "samples": "schema_samples",
    "raw": "raw_sources",
    "logs": "generation_logs",
    "planning": "planning_runs",
    "archive": "legacy_or_archive",
}

CATEGORY_LABELS = {
    "active_train_candidates": "Active training (core)",
    "legacy_or_archive": "Legacy/Archive candidates (not recommended for new training)",
    "evaluation_splits": "Evaluation splits (ID/OOD/Mixed)",
    "mixed_test_candidates": "Mixed-test candidates",
    "mixed_test_sources": "Mixed/hybrid sources",
    "analysis_predictions": "Prediction outputs (pred_probs)",
    "analysis_routed": "Routed pools (core/hard/review/reject)",
    "analysis_classified": "Rule-based classified datasets",
    "analysis_metadata": "Analysis outputs",
    "schema_samples": "Schema conversion samples",
    "raw_sources": "Raw source datasets",
    "generation_logs": "Generation logs",
    "planning_runs": "Planning outputs (data fill runs)",
    "uncategorized": "Uncategorized",
}


def list_dataset_dirs() -> List[str]:
    """List top-level dataset directories."""
    if not os.path.isdir(DATASETS_DIR):
        return []
    return sorted(
        [name for name in os.listdir(DATASETS_DIR) if os.path.isdir(os.path.join(DATASETS_DIR, name))]
    )


def build_registry() -> List[Dict[str, str]]:
    """Build registry entries."""
    entries = []
    for name in list_dataset_dirs():
        category = CATEGORY_MAP.get(name, "uncategorized")
        entries.append(
            {
                "name": name,
                "category": category,
                "category_label": CATEGORY_LABELS.get(category, category),
                "path": f"datasets/{name}",
                "recommended": "true"
                if category == "active_train_candidates"
                else "false",
            }
        )
    return entries


def write_registry(entries: List[Dict[str, str]]) -> None:
    """Write registry JSON and README."""
    os.makedirs(DATASETS_DIR, exist_ok=True)

    registry_path = os.path.join(DATASETS_DIR, "registry.json")
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(
            {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "entries": entries},
            f,
            ensure_ascii=False,
            indent=2,
        )

    readme_path = os.path.join(DATASETS_DIR, "README.md")
    by_category: Dict[str, List[Dict[str, str]]] = {}
    for entry in entries:
        by_category.setdefault(entry["category"], []).append(entry)

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("# Datasets Index\n\n")
        f.write(
            "This index classifies datasets by the current plan. No files were moved.\n\n"
        )
        f.write("## Categories\n\n")
        for category, label in CATEGORY_LABELS.items():
            f.write(f"- {category}: {label}\n")
        f.write("\n## Registry\n\n")
        for category in sorted(by_category.keys()):
            f.write(f"### {CATEGORY_LABELS.get(category, category)}\n\n")
            for entry in by_category[category]:
                f.write(f"- `{entry['path']}`\n")
            f.write("\n")


def main() -> int:
    entries = build_registry()
    write_registry(entries)
    print(os.path.join(DATASETS_DIR, "registry.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
