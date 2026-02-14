#!/usr/bin/env python3
"""Triage unknown-source samples into keep/review/drop buckets."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.utils.risk_patterns import PATTERNS


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Triage unknown-source rows.")
    parser.add_argument(
        "--input",
        default=str(ROOT / "datasets/merged_v2/train_v10.csv"),
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "datasets/analysis/routed/unknown_source_v1"),
        help="Output directory for triage files.",
    )
    parser.add_argument(
        "--max-text-length",
        type=int,
        default=8000,
        help="Hard cap for text length.",
    )
    parser.add_argument(
        "--min-text-length",
        type=int,
        default=10,
        help="Minimum text length.",
    )
    return parser.parse_args()


def load_df(path: Path) -> pd.DataFrame:
    """Load dataset CSV and normalize required columns."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("input csv must contain `text` and `label`")
    if "source" not in df.columns:
        df["source"] = "unknown"
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    df["source"] = df["source"].fillna("unknown").astype(str)
    return df


def match_patterns(text: str, patterns: Dict[str, object]) -> List[str]:
    """Return pattern names matched in text."""
    hit: List[str] = []
    for name, compiled in patterns.items():
        if compiled.search(text):
            hit.append(name)
    return hit


def classify_row(
    text: str,
    min_len: int,
    max_len: int,
) -> Tuple[str, List[str]]:
    """Classify a row into keep/review/drop and return reasons."""
    reasons: List[str] = []
    text_len = len(text)
    if text_len < min_len:
        reasons.append("too_short")
    if text_len > max_len:
        reasons.append("too_long")

    hard_hits = match_patterns(text, PATTERNS.hard_remove)
    soft_hits = match_patterns(text, PATTERNS.soft_flag)

    if hard_hits:
        reasons.extend([f"hard:{name}" for name in hard_hits])
        return "drop_candidate", reasons
    if reasons:
        return "drop_candidate", reasons
    if soft_hits:
        reasons.extend([f"soft:{name}" for name in soft_hits])
        return "review_needed", reasons
    return "keep_verified", ["clean"]


def add_hash(df: pd.DataFrame) -> pd.DataFrame:
    """Add a stable sha1 hash column for downstream filtering."""
    out = df.copy()
    out["text_sha1"] = out["text"].map(lambda t: hashlib.sha1(t.encode("utf-8")).hexdigest())
    return out


def to_markdown_summary(summary: Dict[str, object]) -> str:
    """Build markdown summary from triage stats."""
    lines = [
        "# unknown source triage v1",
        "",
        f"- generated_at: {summary['generated_at']}",
        f"- input: `{summary['input_path']}`",
        f"- unknown_rows: {summary['unknown_rows']:,}",
        "",
        "## bucket stats",
        "",
        "| bucket | rows | ratio |",
        "|---|---:|---:|",
    ]
    for bucket, row_count in summary["bucket_counts"].items():
        ratio = summary["bucket_ratios"].get(bucket, 0.0)
        lines.append(f"| {bucket} | {row_count:,} | {ratio}% |")

    lines.extend(["", "## top reasons", ""])
    lines.append("| reason | count |")
    lines.append("|---|---:|")
    for reason, count in summary["top_reasons"]:
        lines.append(f"| {reason} | {count:,} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    """Entry point."""
    args = parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_df(input_path)
    unknown_df = df[df["source"].eq("unknown")].copy()

    bucket_values: List[str] = []
    reason_values: List[str] = []
    for text in unknown_df["text"]:
        bucket, reasons = classify_row(
            text=text,
            min_len=args.min_text_length,
            max_len=args.max_text_length,
        )
        bucket_values.append(bucket)
        reason_values.append(";".join(reasons))

    unknown_df["triage_bucket"] = bucket_values
    unknown_df["triage_reason"] = reason_values
    unknown_df = add_hash(unknown_df)

    keep_df = unknown_df[unknown_df["triage_bucket"].eq("keep_verified")].copy()
    review_df = unknown_df[unknown_df["triage_bucket"].eq("review_needed")].copy()
    drop_df = unknown_df[unknown_df["triage_bucket"].eq("drop_candidate")].copy()

    keep_path = out_dir / "keep_verified.csv"
    review_path = out_dir / "review_needed.csv"
    drop_path = out_dir / "drop_candidate.csv"
    keep_df.to_csv(keep_path, index=False, encoding="utf-8-sig")
    review_df.to_csv(review_path, index=False, encoding="utf-8-sig")
    drop_df.to_csv(drop_path, index=False, encoding="utf-8-sig")

    reason_counter = Counter()
    for item in unknown_df["triage_reason"]:
        for reason in str(item).split(";"):
            reason_counter[reason] += 1

    bucket_counts = unknown_df["triage_bucket"].value_counts().to_dict()
    total = len(unknown_df)
    bucket_ratios = {
        key: round(value * 100.0 / total, 3) if total else 0.0
        for key, value in bucket_counts.items()
    }

    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_path": str(input_path),
        "unknown_rows": total,
        "bucket_counts": bucket_counts,
        "bucket_ratios": bucket_ratios,
        "top_reasons": reason_counter.most_common(20),
        "outputs": {
            "keep_verified": str(keep_path),
            "review_needed": str(review_path),
            "drop_candidate": str(drop_path),
        },
    }

    summary_json = out_dir / "summary.json"
    summary_md = out_dir / "summary.md"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_md.write_text(to_markdown_summary(summary), encoding="utf-8")

    print(f"[OK] keep_verified: {len(keep_df):,}")
    print(f"[OK] review_needed: {len(review_df):,}")
    print(f"[OK] drop_candidate: {len(drop_df):,}")
    print(f"[OK] summary: {summary_json}")


if __name__ == "__main__":
    main()

