#!/usr/bin/env python3
"""Prepare Mixed-Test candidates and balanced subsets."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

DEFAULT_PHRASES = [
    "作为一个AI",
    "作为AI",
    "作为一个人工智能",
    "AI语言模型",
    "As an AI",
    "as an AI",
    "I cannot",
    "I can't",
    "无法回答",
    "不能回答",
]

LENGTH_BUCKETS: List[Tuple[int, int, str]] = [
    (0, 80, "<80"),
    (80, 200, "80-200"),
    (200, 500, "200-500"),
    (500, 1000, "500-1000"),
    (1000, 2000, "1000-2000"),
    (2000, 10**9, "2000+"),
]

TARGET_LENGTH_RATIOS = {
    "80-200": 0.20,
    "200-500": 0.30,
    "500-1000": 0.25,
    "1000-2000": 0.20,
    "2000+": 0.05,
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prepare Mixed-Test candidates.")
    parser.add_argument(
        "--hybrid",
        default=str(
            PROJECT_ROOT / "datasets" / "mixed" / "hybrid" / "hybrid_dataset_expanded.csv"
        ),
        help="Input hybrid dataset CSV.",
    )
    parser.add_argument(
        "--final-clean",
        default=str(PROJECT_ROOT / "datasets" / "active" / "core_v1" / "full_dataset.csv"),
        help="Core training full dataset for overlap removal.",
    )
    parser.add_argument(
        "--combined-clean-dir",
        default="",
        help="Optional combined_v2_clean directory for overlap removal.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "datasets" / "mixed" / "candidates"),
        help="Output directory for Mixed-Test candidates.",
    )
    parser.add_argument(
        "--report-dir",
        default=str(PROJECT_ROOT / "docs" / "plans" / "audit_reports"),
        help="Output directory for reports.",
    )
    parser.add_argument(
        "--remove-phrases",
        action="store_true",
        help="Remove explicit AI/refusal phrases.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    return parser.parse_args()


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file with basic error handling."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing file: {path}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to read CSV: {path}") from exc


def hash_series(texts: Iterable[str]) -> set:
    """Hash text series to detect duplicates."""
    return set(pd.util.hash_pandas_object(pd.Series(list(texts)), index=False).astype(str))


def bucket_length(length: int) -> str:
    """Map length to bucket name."""
    for low, high, name in LENGTH_BUCKETS:
        if low <= length < high:
            return name
    return "unknown"


def length_bucket_counts(df: pd.DataFrame) -> Counter:
    """Compute length bucket counts for a dataframe."""
    lengths = df["text"].astype(str).str.len()
    buckets = lengths.apply(lambda x: bucket_length(int(x)))
    return Counter(buckets)


def remove_overlap(df: pd.DataFrame, blocked_hashes: set) -> Tuple[pd.DataFrame, int]:
    """Remove rows whose text hashes are in blocked_hashes."""
    hashes = pd.util.hash_pandas_object(df["text"].astype(str), index=False).astype(str)
    mask = ~hashes.isin(blocked_hashes)
    removed = len(df) - mask.sum()
    return df.loc[mask].copy(), removed


def remove_phrases(df: pd.DataFrame, phrases: List[str]) -> Tuple[pd.DataFrame, int]:
    """Remove rows containing explicit AI/refusal phrases."""
    if not phrases:
        return df.copy(), 0
    pattern = "|".join(phrases)
    mask = ~df["text"].astype(str).str.contains(pattern, regex=True, case=False, na=False)
    removed = len(df) - mask.sum()
    return df.loc[mask].copy(), removed


def write_report(path: Path, lines: List[str]) -> None:
    """Write report lines to markdown."""
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Main entry point."""
    args = parse_args()
    hybrid_path = Path(args.hybrid)
    final_clean_path = Path(args.final_clean)
    combined_dir = Path(args.combined_clean_dir) if args.combined_clean_dir else None

    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    df = load_csv(hybrid_path)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    blocked_hashes = set()
    final_df = load_csv(final_clean_path)
    blocked_hashes |= hash_series(final_df["text"].astype(str))

    if combined_dir and combined_dir.exists():
        for split in ["train.csv", "val.csv", "test.csv"]:
            split_path = combined_dir / split
            if split_path.exists():
                split_df = load_csv(split_path)
                blocked_hashes |= hash_series(split_df["text"].astype(str))

    df, removed_overlap = remove_overlap(df, blocked_hashes)

    sep_removed = int(df["text"].astype(str).str.contains(r"\[SEP\]", regex=True, na=False).sum())
    df = df.loc[~df["text"].astype(str).str.contains(r"\[SEP\]", regex=True, na=False)].copy()

    phrase_removed = 0
    if args.remove_phrases:
        df, phrase_removed = remove_phrases(df, DEFAULT_PHRASES)

    if "category" in df.columns:
        df = df[df["category"].isin(["C2", "C3", "C4"])].copy()

    candidate_path = output_dir / "hybrid_expanded_clean_vs_core_v1.csv"
    df.to_csv(candidate_path, index=False)

    candidate_report = report_dir / f"{datetime.now().strftime('%Y-%m-%d')}_mixed_test_candidate_report.md"
    lines = [
        "# Mixed-Test 候选集生成报告",
        "",
        f"> 生成日期: {datetime.now().strftime('%Y-%m-%d')}",
        "",
        f"- 输入: {hybrid_path}",
        f"- 输出: {candidate_path}",
        f"- 移除 overlap: {removed_overlap}",
        f"- 移除 [SEP]: {sep_removed}",
        f"- 移除口癖: {phrase_removed}",
        f"- 保留样本: {len(df)}",
        "",
        "## 类别分布",
    ]
    if "category" in df.columns:
        for k, v in df["category"].value_counts().to_dict().items():
            pct = round((v / len(df)) * 100, 2) if len(df) else 0
            lines.append(f"- {k}: {v} ({pct}%)")
    else:
        lines.append("- (无 category 字段)")

    lines.append("")
    lines.append("## 长度分桶")
    bucket_counts = length_bucket_counts(df)
    for name in ["<80", "80-200", "200-500", "500-1000", "1000-2000", "2000+"]:
        v = bucket_counts.get(name, 0)
        pct = round((v / len(df)) * 100, 2) if len(df) else 0
        lines.append(f"- {name}: {v} ({pct}%)")
    write_report(candidate_report, lines)

    # Category balance
    balanced_path = output_dir / "mixed_test_balanced_by_category.csv"
    balance_report = report_dir / f"{datetime.now().strftime('%Y-%m-%d')}_mixed_test_category_balance_report.md"
    if "category" in df.columns:
        cat_counts = df["category"].value_counts()
        min_count = int(cat_counts.min()) if not cat_counts.empty else 0
        balanced_frames = []
        for cat in ["C2", "C3", "C4"]:
            subset = df[df["category"] == cat]
            if min_count > 0:
                subset = subset.sample(n=min_count, random_state=args.seed)
            balanced_frames.append(subset)
        balanced_df = pd.concat(balanced_frames, ignore_index=True) if balanced_frames else df
    else:
        balanced_df = df.copy()
        min_count = 0

    balanced_df.to_csv(balanced_path, index=False)
    balance_lines = [
        "# Mixed-Test 类别均衡报告",
        "",
        f"> 生成日期: {datetime.now().strftime('%Y-%m-%d')}",
        "",
        f"- 输入: {candidate_path}",
        f"- 输出: {balanced_path}",
        f"- 采样目标 (min per category): {min_count}",
        f"- 输出样本: {len(balanced_df)}",
        "",
        "## 类别分布",
    ]
    if "category" in balanced_df.columns:
        for k, v in balanced_df["category"].value_counts().to_dict().items():
            pct = round((v / len(balanced_df)) * 100, 2) if len(balanced_df) else 0
            balance_lines.append(f"- {k}: {v} ({pct}%)")
    else:
        balance_lines.append("- (无 category 字段)")
    write_report(balance_report, balance_lines)

    # Length balance (cap only, no upsampling)
    length_balanced_path = output_dir / "mixed_test_balanced_by_category_length.csv"
    length_report = report_dir / f"{datetime.now().strftime('%Y-%m-%d')}_mixed_test_length_balance_report.md"

    balanced_df = balanced_df.copy()
    balanced_df["length_bucket"] = balanced_df["text"].astype(str).str.len().apply(bucket_length)
    total_before_short = len(balanced_df)
    balanced_df = balanced_df[balanced_df["length_bucket"] != "<80"].copy()
    short_removed = total_before_short - len(balanced_df)

    total = len(balanced_df)
    target_counts = {
        bucket: int(round(total * ratio)) for bucket, ratio in TARGET_LENGTH_RATIOS.items()
    }

    sampled_frames = []
    length_removed = 0
    for bucket, target in target_counts.items():
        subset = balanced_df[balanced_df["length_bucket"] == bucket]
        if len(subset) > target:
            subset = subset.sample(n=target, random_state=args.seed)
            length_removed += len(balanced_df[balanced_df["length_bucket"] == bucket]) - len(subset)
        sampled_frames.append(subset)

    length_balanced_df = pd.concat(sampled_frames, ignore_index=True) if sampled_frames else balanced_df
    length_balanced_df.drop(columns=["length_bucket"], inplace=True, errors="ignore")
    length_balanced_df.to_csv(length_balanced_path, index=False)

    length_lines = [
        "# Mixed-Test 长度均衡报告",
        "",
        f"> 生成日期: {datetime.now().strftime('%Y-%m-%d')}",
        "",
        f"- 输入: {balanced_path}",
        f"- 输出: {length_balanced_path}",
        f"- 原始样本(去除<80前): {total_before_short}",
        f"- <80 移除: {short_removed}",
        f"- 原始样本(去除<80后): {total}",
        f"- 下采样移除: {length_removed}",
        f"- 输出样本: {len(length_balanced_df)}",
        "",
        "## 长度分桶",
    ]
    post_counts = length_bucket_counts(length_balanced_df)
    for name in ["<80", "80-200", "200-500", "500-1000", "1000-2000", "2000+"]:
        v = post_counts.get(name, 0)
        pct = round((v / len(length_balanced_df)) * 100, 2) if len(length_balanced_df) else 0
        length_lines.append(f"- {name}: {v} ({pct}%)")
    write_report(length_report, length_lines)

    print(str(candidate_path))
    print(str(balanced_path))
    print(str(length_balanced_path))


if __name__ == "__main__":
    main()
