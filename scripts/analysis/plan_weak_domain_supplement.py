#!/usr/bin/env python3
"""Build a weak-domain supplement plan with minimum sample constraints."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Plan weak-domain data supplement targets.")
    parser.add_argument(
        "--eval-comparison",
        default=str(ROOT / "models/bert_v10_augmented/eval_comparison.json"),
        help="Eval comparison JSON with `independent_data_by_source` metrics.",
    )
    parser.add_argument(
        "--train",
        default=str(ROOT / "datasets/merged_v2/train_v11_candidate.csv"),
        help="Current train set used as counting baseline.",
    )
    parser.add_argument(
        "--weak-threshold",
        type=float,
        default=95.0,
        help="Domain accuracy below this threshold is considered weak.",
    )
    parser.add_argument(
        "--min-per-weak-domain",
        type=int,
        default=300,
        help="Minimum required samples for each weak domain.",
    )
    parser.add_argument(
        "--diversity-target-prefix-uniq-ratio",
        type=float,
        default=0.70,
        help="Target minimum unique normalized prefix ratio for supplements.",
    )
    parser.add_argument(
        "--diversity-max-top-prefix-share",
        type=float,
        default=0.20,
        help="Target maximum share of the most frequent normalized prefix.",
    )
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "docs/plans/weak_domain_supplement_plan_v1.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "docs/plans/weak_domain_supplement_plan_v1.md"),
        help="Output Markdown path.",
    )
    return parser.parse_args()


def normalize_prefix(text: str, limit: int = 80) -> str:
    """Normalize text and return a short prefix key."""
    norm = re.sub(r"\W+", "", str(text).strip().lower())
    return norm[:limit]


def load_eval_metrics(path: Path, weak_threshold: float) -> List[Dict[str, object]]:
    """Load weak domain entries from eval comparison JSON."""
    if not path.exists():
        raise FileNotFoundError(f"Eval comparison not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    model_entry = payload.get("bert_v10_augmented", {})
    by_source = model_entry.get("independent_data_by_source", {})

    weak_domains: List[Dict[str, object]] = []
    for source, metrics in by_source.items():
        accuracy = float(metrics.get("accuracy", 0.0))
        count = int(metrics.get("count", 0))
        if accuracy < weak_threshold:
            weak_domains.append(
                {
                    "source": source,
                    "accuracy": accuracy,
                    "eval_count": count,
                }
            )
    weak_domains.sort(key=lambda item: item["accuracy"])
    return weak_domains


def load_train(path: Path) -> pd.DataFrame:
    """Load train CSV and normalize core columns."""
    if not path.exists():
        raise FileNotFoundError(f"Train CSV not found: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"text", "label", "source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {sorted(missing)} in {path}")

    out = df.copy()
    out["text"] = out["text"].astype(str)
    out["label"] = out["label"].astype(int)
    out["source"] = out["source"].fillna("unknown").astype(str)
    out["normalized_prefix"] = out["text"].map(normalize_prefix)
    out["text_sha1"] = out["text"].map(lambda t: hashlib.sha1(t.encode("utf-8")).hexdigest())
    return out


def calc_diversity_stats(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate lightweight diversity metrics for one source slice."""
    total = len(df)
    if total == 0:
        return {
            "rows": 0,
            "unique_prefix_count": 0,
            "unique_prefix_ratio": 0.0,
            "top_prefix_share": 0.0,
            "duplicate_text_ratio": 0.0,
        }

    prefix_counts = df["normalized_prefix"].value_counts()
    unique_prefix_count = int(prefix_counts.shape[0])
    top_prefix_share = float(prefix_counts.iloc[0] / total) if not prefix_counts.empty else 0.0
    duplicate_text_ratio = float(df.duplicated(subset=["text_sha1"]).sum() / total)
    return {
        "rows": int(total),
        "unique_prefix_count": unique_prefix_count,
        "unique_prefix_ratio": round(unique_prefix_count / total, 4),
        "top_prefix_share": round(top_prefix_share, 4),
        "duplicate_text_ratio": round(duplicate_text_ratio, 4),
    }


def build_plan(
    weak_domains: List[Dict[str, object]],
    train_df: pd.DataFrame,
    min_per_weak_domain: int,
    diversity_target_prefix_uniq_ratio: float,
    diversity_max_top_prefix_share: float,
) -> Dict[str, object]:
    """Build supplement plan payload."""
    source_counts = train_df["source"].value_counts().to_dict()
    plan_rows: List[Dict[str, object]] = []
    total_required = 0

    for item in weak_domains:
        source = item["source"]
        current_count = int(source_counts.get(source, 0))
        required_new = max(0, min_per_weak_domain - current_count)
        total_required += required_new

        source_df = train_df[train_df["source"] == source].copy()
        diversity = calc_diversity_stats(source_df)

        plan_rows.append(
            {
                "source": source,
                "accuracy": item["accuracy"],
                "eval_count": item["eval_count"],
                "current_train_count": current_count,
                "target_train_count": max(current_count, min_per_weak_domain),
                "required_new_samples": required_new,
                "diversity_now": diversity,
                "diversity_targets": {
                    "min_unique_prefix_ratio": diversity_target_prefix_uniq_ratio,
                    "max_top_prefix_share": diversity_max_top_prefix_share,
                },
                "quality_rules": [
                    "at_least_3_prompt_families_if_generated",
                    "avoid_single_template_batching",
                    "manual_spot_check_5_percent",
                ],
            }
        )

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "weak_domain_count": len(weak_domains),
        "min_per_weak_domain": min_per_weak_domain,
        "total_required_new_samples": total_required,
        "domains": plan_rows,
    }


def render_markdown(plan: Dict[str, object], args: argparse.Namespace) -> str:
    """Render Markdown report."""
    lines = [
        "# Weak Domain Supplement Plan v1",
        "",
        f"- Generated at: {plan['generated_at']}",
        f"- Eval comparison: `{Path(args.eval_comparison)}`",
        f"- Train baseline: `{Path(args.train)}`",
        f"- Weak threshold: {args.weak_threshold}",
        f"- Minimum per weak domain: {plan['min_per_weak_domain']}",
        f"- Total required new samples: {plan['total_required_new_samples']}",
        "",
        "| source | acc | eval_count | current_train | target | required_new | "
        "uniq_prefix_ratio | top_prefix_share |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in plan["domains"]:
        diversity = row["diversity_now"]
        lines.append(
            f"| {row['source']} | {row['accuracy']} | {row['eval_count']} | "
            f"{row['current_train_count']} | {row['target_train_count']} | "
            f"{row['required_new_samples']} | {diversity['unique_prefix_ratio']} | "
            f"{diversity['top_prefix_share']} |"
        )

    lines.extend(
        [
            "",
            "## Diversity constraints",
            "",
            f"- min_unique_prefix_ratio >= {args.diversity_target_prefix_uniq_ratio}",
            f"- max_top_prefix_share <= {args.diversity_max_top_prefix_share}",
            "- If generated data is used, use at least 3 prompt families per weak domain.",
        ]
    )
    return "\n".join(lines) + "\n"


def ensure_parent(path: Path) -> None:
    """Ensure parent directory exists."""
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Entry point."""
    args = parse_args()
    eval_comparison_path = Path(args.eval_comparison)
    train_path = Path(args.train)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)

    weak_domains = load_eval_metrics(eval_comparison_path, args.weak_threshold)
    train_df = load_train(train_path)
    plan = build_plan(
        weak_domains=weak_domains,
        train_df=train_df,
        min_per_weak_domain=args.min_per_weak_domain,
        diversity_target_prefix_uniq_ratio=args.diversity_target_prefix_uniq_ratio,
        diversity_max_top_prefix_share=args.diversity_max_top_prefix_share,
    )

    ensure_parent(output_json)
    ensure_parent(output_md)
    output_json.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(render_markdown(plan, args), encoding="utf-8")

    print(f"[OK] JSON report: {output_json}")
    print(f"[OK] Markdown report: {output_md}")
    print(
        "[INFO] weak domains: "
        f"{plan['weak_domain_count']}, required new samples: {plan['total_required_new_samples']}"
    )


if __name__ == "__main__":
    main()

