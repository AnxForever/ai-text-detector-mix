#!/usr/bin/env python3
"""Build v11 candidate training set from v10 with risk-aware filtering."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Dict, List, Set

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.utils.risk_patterns import PATTERNS


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Build risk-filtered train_v11 candidate.")
    parser.add_argument(
        "--input",
        default=str(ROOT / "datasets/merged_v2/train_v10.csv"),
        help="Input v10 dataset csv path.",
    )
    parser.add_argument(
        "--unknown-triage-dir",
        default=str(ROOT / "datasets/analysis/routed/unknown_source_v1"),
        help="Directory from triage_unknown_source.py.",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "datasets/merged_v2/train_v11_candidate.csv"),
        help="Output candidate csv path.",
    )
    parser.add_argument(
        "--summary-json",
        default=str(ROOT / "datasets/merged_v2/train_v11_candidate_summary.json"),
        help="Summary json output path.",
    )
    parser.add_argument(
        "--allow-review-unknown",
        action="store_true",
        help="Keep review_needed unknown rows in addition to keep_verified.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=8000,
        help="Drop rows longer than this threshold.",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=10,
        help="Drop rows shorter than this threshold.",
    )
    return parser.parse_args()


def load_df(path: Path) -> pd.DataFrame:
    """Load input dataframe."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    for col in ("text", "label"):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    if "source" not in df.columns:
        df["source"] = "unknown"
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    df["source"] = df["source"].fillna("unknown").astype(str)
    return df


def add_sha1(df: pd.DataFrame) -> pd.DataFrame:
    """Add sha1 column."""
    out = df.copy()
    out["text_sha1"] = out["text"].map(lambda t: hashlib.sha1(t.encode("utf-8")).hexdigest())
    return out


def collect_allowed_unknown_hashes(
    triage_dir: Path,
    include_review: bool,
) -> Set[str]:
    """Load allowed unknown hashes from triage outputs."""
    keep_path = triage_dir / "keep_verified.csv"
    review_path = triage_dir / "review_needed.csv"
    hashes: Set[str] = set()

    if keep_path.exists():
        keep_df = pd.read_csv(keep_path, encoding="utf-8-sig")
        if "text_sha1" in keep_df.columns:
            hashes.update(keep_df["text_sha1"].dropna().astype(str).tolist())

    if include_review and review_path.exists():
        review_df = pd.read_csv(review_path, encoding="utf-8-sig")
        if "text_sha1" in review_df.columns:
            hashes.update(review_df["text_sha1"].dropna().astype(str).tolist())
    return hashes


def hard_pattern_mask(text_series: pd.Series) -> pd.Series:
    """Return boolean mask for hard-remove pattern matches."""
    mask = pd.Series([False] * len(text_series), index=text_series.index)
    for compiled in PATTERNS.hard_remove.values():
        mask = mask | text_series.str.contains(compiled, regex=True, na=False)
    return mask


def filter_rows(
    df: pd.DataFrame,
    allowed_unknown_hashes: Set[str],
    min_length: int,
    max_length: int,
) -> tuple[pd.DataFrame, Dict[str, int]]:
    """Apply risk-aware filtering and return filtered df + counters."""
    counters: Dict[str, int] = Counter()
    out = df.copy()

    out["char_len"] = out["text"].str.len()
    out = out[~out["text"].str.fullmatch(r"\s*", na=False)]
    counters["drop_empty"] = len(df) - len(out)

    before_len = len(out)
    out = out[(out["char_len"] >= min_length) & (out["char_len"] <= max_length)]
    counters["drop_length"] = before_len - len(out)

    before_hard = len(out)
    hard_mask = hard_pattern_mask(out["text"])
    out = out[~hard_mask]
    counters["drop_hard_pattern"] = before_hard - len(out)

    before_unknown = len(out)
    unknown_mask = out["source"].eq("unknown")
    keep_unknown_mask = unknown_mask & out["text_sha1"].isin(allowed_unknown_hashes)
    out = out[~unknown_mask | keep_unknown_mask]
    counters["drop_unknown_unapproved"] = before_unknown - len(out)

    before_dedup = len(out)
    out = out.drop_duplicates(subset=["text"], keep="first")
    counters["drop_duplicates"] = before_dedup - len(out)

    out = out.drop(columns=["char_len"], errors="ignore").reset_index(drop=True)
    return out, counters


def summarize(df: pd.DataFrame) -> Dict[str, object]:
    """Return summary stats for output dataframe."""
    counts = df["label"].value_counts().to_dict()
    out = {
        "rows": len(df),
        "human_count": int(counts.get(0, 0)),
        "ai_count": int(counts.get(1, 0)),
        "unknown_count": int((df["source"] == "unknown").sum()),
        "unique_sources": int(df["source"].nunique()),
    }
    if len(df):
        out["ai_ratio_percent"] = round(out["ai_count"] * 100.0 / len(df), 3)
    else:
        out["ai_ratio_percent"] = 0.0
    return out


def main() -> None:
    """Entry point."""
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary_json)
    triage_dir = Path(args.unknown_triage_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    base_df = add_sha1(load_df(input_path))
    allowed_unknown_hashes = collect_allowed_unknown_hashes(
        triage_dir=triage_dir,
        include_review=args.allow_review_unknown,
    )

    filtered_df, counters = filter_rows(
        df=base_df,
        allowed_unknown_hashes=allowed_unknown_hashes,
        min_length=args.min_length,
        max_length=args.max_length,
    )

    filtered_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "allow_review_unknown": args.allow_review_unknown,
        "filter_counters": counters,
        "before": summarize(base_df),
        "after": summarize(filtered_df),
        "approved_unknown_hashes": len(allowed_unknown_hashes),
        "top_sources_after": filtered_df["source"].value_counts().head(20).to_dict(),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] output: {output_path}")
    print(f"[OK] summary: {summary_path}")
    print(f"[INFO] rows: {len(base_df):,} -> {len(filtered_df):,}")
    print(f"[INFO] unknown after: {(filtered_df['source'] == 'unknown').sum():,}")


if __name__ == "__main__":
    main()

