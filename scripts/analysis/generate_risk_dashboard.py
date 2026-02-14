#!/usr/bin/env python3
"""Generate a risk dashboard for the current training dataset."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.utils.risk_patterns import PATTERNS

EVAL_SET_FILES: Tuple[Tuple[str, str], ...] = (
    ("core_v1_test_clean", "core_v1_test_clean.csv"),
    ("independent_data", "independent_data.csv"),
    ("merged_v2_val_clean", "merged_v2_val_clean.csv"),
)


@dataclass
class OverlapResult:
    """Stores overlap statistics between train and one eval set."""

    dataset: str
    rows: int
    exact_overlap: int
    prefix200_overlap: int
    normalized_prefix200_overlap: int


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate risk dashboard for train_v10.")
    parser.add_argument(
        "--train",
        default=str(ROOT / "datasets/merged_v2/train_v10.csv"),
        help="Training dataset CSV path.",
    )
    parser.add_argument(
        "--eval-dir",
        default=str(ROOT / "datasets/eval/fair_test"),
        help="Directory containing fair test CSV files.",
    )
    parser.add_argument(
        "--eval-comparison",
        default=str(ROOT / "models/bert_v10_augmented/eval_comparison.json"),
        help="Path to eval comparison JSON for weak-domain extraction.",
    )
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "docs/plans/risk_dashboard_v1.json"),
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "docs/plans/risk_dashboard_v1.md"),
        help="Output Markdown report path.",
    )
    parser.add_argument(
        "--weak-domain-threshold",
        type=float,
        default=95.0,
        help="Domain accuracy threshold below which a source is marked weak.",
    )
    parser.add_argument(
        "--exact-duplicate-threshold-percent",
        type=float,
        default=1.0,
        help="Threshold for exact duplicate risk flag.",
    )
    parser.add_argument(
        "--near-duplicate-threshold-percent",
        type=float,
        default=3.0,
        help="Threshold for near duplicate risk flag.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    """Normalize text for near-duplicate checks."""
    lowered = str(text).strip().lower()
    return re.sub(r"\W+", "", lowered)


def load_csv(path: Path) -> pd.DataFrame:
    """Load dataset CSV and derive common audit columns."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {sorted(missing)} in {path}")

    if "source" not in df.columns:
        df["source"] = "unknown"

    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    df["source"] = df["source"].fillna("unknown").astype(str)
    df["char_len"] = df["text"].str.len()
    df["sha1"] = df["text"].map(lambda t: hashlib.sha1(t.encode("utf-8")).hexdigest())
    df["prefix200"] = df["text"].str[:200]
    df["normalized_text"] = df["text"].map(normalize_text)
    df["normalized_sha1"] = df["normalized_text"].map(
        lambda t: hashlib.sha1(t.encode("utf-8")).hexdigest()
    )
    df["normalized_prefix200"] = df["normalized_text"].str[:200]
    return df


def compute_label_stats(df: pd.DataFrame) -> Dict[str, float]:
    """Compute label distribution metrics."""
    label_counts = df["label"].value_counts().to_dict()
    human = int(label_counts.get(0, 0))
    ai = int(label_counts.get(1, 0))
    total = len(df)
    return {
        "rows": total,
        "human_count": human,
        "ai_count": ai,
        "ai_ratio_percent": round(ai * 100.0 / total, 3) if total else 0.0,
        "human_ratio_percent": round(human * 100.0 / total, 3) if total else 0.0,
    }


def compute_source_stats(df: pd.DataFrame) -> Dict[str, object]:
    """Compute source diversity and unknown-source ratios."""
    source_counts = df["source"].value_counts()
    unknown_count = int(source_counts.get("unknown", 0))
    total = len(df)
    return {
        "unique_sources": int(df["source"].nunique()),
        "unknown_count": unknown_count,
        "unknown_ratio_percent": round(unknown_count * 100.0 / total, 3) if total else 0.0,
        "top_sources": source_counts.head(20).to_dict(),
    }


def compute_length_bucket_stats(df: pd.DataFrame) -> List[Dict[str, object]]:
    """Compute label distribution by length bucket."""
    bins = [0, 64, 128, 256, 512, 1024, 2048, 100000]
    labels = ["0-64", "64-128", "128-256", "256-512", "512-1024", "1024-2048", "2048+"]
    bucket = pd.cut(df["char_len"], bins=bins, labels=labels, right=False)
    grouped = df.groupby(bucket, observed=False)["label"].agg(["count", "sum"]).reset_index()

    rows: List[Dict[str, object]] = []
    for _, row in grouped.iterrows():
        count = int(row["count"])
        ai_count = int(row["sum"])
        human_count = count - ai_count
        ai_ratio = round(ai_count * 100.0 / count, 3) if count else 0.0
        risk_level = "balanced"
        if ai_ratio > 80.0 or ai_ratio < 20.0:
            risk_level = "high_bias"
        elif ai_ratio > 70.0 or ai_ratio < 30.0:
            risk_level = "medium_bias"
        rows.append(
            {
                "bucket": str(row["char_len"]),
                "count": count,
                "human_count": human_count,
                "ai_count": ai_count,
                "ai_ratio_percent": ai_ratio,
                "risk_level": risk_level,
            }
        )
    return rows


def _match_pattern_count(texts: pd.Series, patterns: Dict[str, object]) -> Dict[str, int]:
    """Count matches for each compiled pattern."""
    out: Dict[str, int] = {}
    for name, compiled in patterns.items():
        out[name] = int(texts.str.contains(compiled, regex=True, na=False).sum())
    return out


def _match_union_count(texts: pd.Series, patterns: Dict[str, object]) -> int:
    """Count rows matched by at least one pattern in a set."""
    if texts.empty:
        return 0
    mask = pd.Series([False] * len(texts), index=texts.index)
    for compiled in patterns.values():
        mask = mask | texts.str.contains(compiled, regex=True, na=False)
    return int(mask.sum())


def compute_template_noise(df: pd.DataFrame) -> Dict[str, object]:
    """Compute template/instruction leakage metrics."""
    hard_counts = _match_pattern_count(df["text"], PATTERNS.hard_remove)
    soft_counts = _match_pattern_count(df["text"], PATTERNS.soft_flag)

    hard_union = _match_union_count(df["text"], PATTERNS.hard_remove)
    soft_union = _match_union_count(df["text"], PATTERNS.soft_flag)

    ai_df = df[df["label"] == 1]
    human_df = df[df["label"] == 0]

    total = len(df)
    return {
        "hard_pattern_counts": hard_counts,
        "soft_pattern_counts": soft_counts,
        "hard_match_rows": hard_union,
        "soft_match_rows": soft_union,
        "hard_match_ratio_percent": round(hard_union * 100.0 / total, 3) if total else 0.0,
        "soft_match_ratio_percent": round(soft_union * 100.0 / total, 3) if total else 0.0,
        "ai_soft_pattern_counts": _match_pattern_count(ai_df["text"], PATTERNS.soft_flag),
        "human_soft_pattern_counts": _match_pattern_count(human_df["text"], PATTERNS.soft_flag),
    }


def compute_duplicate_stats(df: pd.DataFrame) -> Dict[str, object]:
    """Compute exact and near-duplicate ratios for A1 data health checks."""
    total = len(df)
    if total == 0:
        return {
            "exact_duplicate_rows": 0,
            "exact_duplicate_ratio_percent": 0.0,
            "normalized_duplicate_rows": 0,
            "normalized_duplicate_ratio_percent": 0.0,
            "prefix200_duplicate_rows": 0,
            "prefix200_duplicate_ratio_percent": 0.0,
            "normalized_prefix200_duplicate_rows": 0,
            "normalized_prefix200_duplicate_ratio_percent": 0.0,
            "top_normalized_prefix_clusters": [],
        }

    exact_dup_rows = int(df.duplicated(subset=["sha1"]).sum())
    normalized_dup_rows = int(df.duplicated(subset=["normalized_sha1"]).sum())
    prefix_dup_rows = int(df.duplicated(subset=["prefix200"]).sum())
    normalized_prefix_dup_rows = int(df.duplicated(subset=["normalized_prefix200"]).sum())

    cluster_df = df[["normalized_prefix200", "text"]].copy()
    cluster_df["cluster_key"] = cluster_df["normalized_prefix200"].str[:120]
    cluster_df = cluster_df[cluster_df["cluster_key"].str.len() >= 40]
    grouped = cluster_df.groupby("cluster_key", observed=False)["text"].agg(["size", "first"])
    grouped = grouped[grouped["size"] > 1].sort_values(by="size", ascending=False).head(10)

    top_clusters: List[Dict[str, object]] = []
    for cluster_key, row in grouped.iterrows():
        sample_text = str(row["first"]).replace("\n", " ").strip()
        top_clusters.append(
            {
                "cluster_size": int(row["size"]),
                "sample_text_prefix": sample_text[:160],
                "cluster_key_prefix": cluster_key[:80],
            }
        )

    return {
        "exact_duplicate_rows": exact_dup_rows,
        "exact_duplicate_ratio_percent": round(exact_dup_rows * 100.0 / total, 3),
        "normalized_duplicate_rows": normalized_dup_rows,
        "normalized_duplicate_ratio_percent": round(normalized_dup_rows * 100.0 / total, 3),
        "prefix200_duplicate_rows": prefix_dup_rows,
        "prefix200_duplicate_ratio_percent": round(prefix_dup_rows * 100.0 / total, 3),
        "normalized_prefix200_duplicate_rows": normalized_prefix_dup_rows,
        "normalized_prefix200_duplicate_ratio_percent": round(
            normalized_prefix_dup_rows * 100.0 / total, 3
        ),
        "top_normalized_prefix_clusters": top_clusters,
    }


def compute_overlap(train_df: pd.DataFrame, eval_path: Path, name: str) -> OverlapResult:
    """Check exact and prefix overlaps between train and one eval set."""
    eval_df = load_csv(eval_path)

    train_sha1 = set(train_df["sha1"])
    train_prefix200 = set(train_df["prefix200"])
    train_norm_prefix200 = set(train_df["normalized_prefix200"])

    exact_overlap = int(eval_df["sha1"].isin(train_sha1).sum())
    prefix_overlap = int(eval_df["prefix200"].isin(train_prefix200).sum())
    norm_prefix_overlap = int(eval_df["normalized_prefix200"].isin(train_norm_prefix200).sum())

    return OverlapResult(
        dataset=name,
        rows=len(eval_df),
        exact_overlap=exact_overlap,
        prefix200_overlap=prefix_overlap,
        normalized_prefix200_overlap=norm_prefix_overlap,
    )


def parse_weak_domains(path: Path, threshold: float) -> List[Dict[str, object]]:
    """Extract weak domains from model eval comparison output."""
    if not path.exists():
        return []

    payload = json.loads(path.read_text(encoding="utf-8"))
    v10 = payload.get("bert_v10_augmented", {})
    by_source = v10.get("independent_data_by_source", {})

    weak_domains: List[Dict[str, object]] = []
    for source, metrics in by_source.items():
        accuracy = float(metrics.get("accuracy", 0.0))
        if accuracy < threshold:
            weak_domains.append(
                {
                    "source": source,
                    "accuracy": accuracy,
                    "count": int(metrics.get("count", 0)),
                }
            )
    weak_domains.sort(key=lambda item: item["accuracy"])
    return weak_domains


def build_risk_flags(report: Dict[str, object], args: argparse.Namespace) -> List[str]:
    """Build top-level risk flags for quick triage decisions."""
    flags: List[str] = []

    source_stats = report["source_stats"]
    if source_stats["unknown_ratio_percent"] >= 3.0:
        flags.append("unknown_source_ratio_high")

    if any(row["risk_level"] == "high_bias" for row in report["length_bucket_stats"]):
        flags.append("length_bucket_bias_high")

    template_noise = report["template_noise"]
    if template_noise["hard_match_ratio_percent"] >= 0.1:
        flags.append("template_leakage_present")

    duplicate_stats = report["duplicate_stats"]
    if duplicate_stats["exact_duplicate_ratio_percent"] >= args.exact_duplicate_threshold_percent:
        flags.append("exact_duplicate_ratio_high")
    if (
        duplicate_stats["normalized_prefix200_duplicate_ratio_percent"]
        >= args.near_duplicate_threshold_percent
    ):
        flags.append("near_duplicate_ratio_high")

    overlaps = report["eval_overlap_checks"]
    if any(
        item["exact_overlap"] > 0
        or item["prefix200_overlap"] > 0
        for item in overlaps
    ):
        flags.append("train_eval_overlap_detected")

    if report["weak_domains"]:
        flags.append("weak_domain_present")

    return flags


def render_markdown(report: Dict[str, object]) -> str:
    """Render a concise Markdown summary."""
    label = report["label_stats"]
    source = report["source_stats"]
    duplicate = report["duplicate_stats"]
    template = report["template_noise"]

    lines = [
        "# Risk Dashboard v1",
        "",
        f"- Generated at: {report['generated_at']}",
        f"- Train set: `{report['train_path']}`",
        "",
        "## 1) Basic stats",
        "",
        f"- Rows: {label['rows']:,}",
        f"- Human / AI: {label['human_count']:,} / {label['ai_count']:,}",
        f"- AI ratio: {label['ai_ratio_percent']}%",
        "",
        "## 2) Source risk",
        "",
        f"- Unique sources: {source['unique_sources']}",
        f"- Unknown source rows: {source['unknown_count']:,} ({source['unknown_ratio_percent']}%)",
        "",
        "## 3) Length bucket risk",
        "",
        "| bucket | count | human | ai | ai_ratio | risk |",
        "|---|---:|---:|---:|---:|---|",
    ]

    for row in report["length_bucket_stats"]:
        lines.append(
            f"| {row['bucket']} | {row['count']:,} | {row['human_count']:,} | "
            f"{row['ai_count']:,} | {row['ai_ratio_percent']}% | {row['risk_level']} |"
        )

    lines.extend(
        [
            "",
            "## 4) Duplicate and near-duplicate risk (A1)",
            "",
            f"- Exact duplicates: {duplicate['exact_duplicate_rows']:,} "
            f"({duplicate['exact_duplicate_ratio_percent']}%)",
            f"- Normalized duplicates: {duplicate['normalized_duplicate_rows']:,} "
            f"({duplicate['normalized_duplicate_ratio_percent']}%)",
            f"- Prefix200 duplicates: {duplicate['prefix200_duplicate_rows']:,} "
            f"({duplicate['prefix200_duplicate_ratio_percent']}%)",
            f"- Normalized prefix200 duplicates: "
            f"{duplicate['normalized_prefix200_duplicate_rows']:,} "
            f"({duplicate['normalized_prefix200_duplicate_ratio_percent']}%)",
            "",
            "## 5) Template leakage",
            "",
            f"- Hard pattern rows: {template['hard_match_rows']:,} "
            f"({template['hard_match_ratio_percent']}%)",
            f"- Soft pattern rows: {template['soft_match_rows']:,} "
            f"({template['soft_match_ratio_percent']}%)",
            "",
            "## 6) Train vs fair_test overlap (A1)",
            "",
            "| eval set | rows | exact | prefix200 | normalized_prefix200 |",
            "|---|---:|---:|---:|---:|",
        ]
    )

    for item in report["eval_overlap_checks"]:
        lines.append(
            f"| {item['dataset']} | {item['rows']:,} | {item['exact_overlap']} | "
            f"{item['prefix200_overlap']} | {item['normalized_prefix200_overlap']} |"
        )

    lines.extend(["", "## 7) Weak domains", ""])
    if report["weak_domains"]:
        lines.append("| source | accuracy | count |")
        lines.append("|---|---:|---:|")
        for item in report["weak_domains"]:
            lines.append(f"| {item['source']} | {item['accuracy']} | {item['count']} |")
    else:
        lines.append("- No weak domain below threshold.")

    lines.extend(["", "## 8) Top-level risk flags", ""])
    if report["risk_flags"]:
        for flag in report["risk_flags"]:
            lines.append(f"- `{flag}`")
    else:
        lines.append("- No high-risk flag.")

    return "\n".join(lines) + "\n"


def ensure_parent(path: Path) -> None:
    """Create parent directory when missing."""
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Entry point."""
    args = parse_args()
    train_path = Path(args.train)
    eval_dir = Path(args.eval_dir)
    eval_comparison_path = Path(args.eval_comparison)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)

    train_df = load_csv(train_path)

    overlaps: List[Dict[str, object]] = []
    for eval_name, eval_file in EVAL_SET_FILES:
        eval_path = eval_dir / eval_file
        if eval_path.exists():
            overlaps.append(asdict(compute_overlap(train_df, eval_path, eval_name)))

    report: Dict[str, object] = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "train_path": str(train_path),
        "thresholds": {
            "weak_domain_threshold": args.weak_domain_threshold,
            "exact_duplicate_threshold_percent": args.exact_duplicate_threshold_percent,
            "near_duplicate_threshold_percent": args.near_duplicate_threshold_percent,
        },
        "label_stats": compute_label_stats(train_df),
        "source_stats": compute_source_stats(train_df),
        "length_bucket_stats": compute_length_bucket_stats(train_df),
        "duplicate_stats": compute_duplicate_stats(train_df),
        "template_noise": compute_template_noise(train_df),
        "eval_overlap_checks": overlaps,
        "weak_domains": parse_weak_domains(
            eval_comparison_path, threshold=args.weak_domain_threshold
        ),
    }
    report["risk_flags"] = build_risk_flags(report, args)

    ensure_parent(output_json)
    ensure_parent(output_md)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(render_markdown(report), encoding="utf-8")

    print(f"[OK] JSON report: {output_json}")
    print(f"[OK] Markdown report: {output_md}")
    if report["risk_flags"]:
        print(f"[INFO] risk flags: {', '.join(report['risk_flags'])}")
    else:
        print("[INFO] risk flags: none")


if __name__ == "__main__":
    main()
