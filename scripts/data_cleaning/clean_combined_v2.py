#!/usr/bin/env python3
"""Clean combined_v2 dataset by removing [SEP], explicit AI phrases, and duplicates.

This script:
1) Loads train/val/test from datasets/archive/combined_v2
2) Removes samples containing [SEP]
3) Optionally removes explicit AI/拒绝口癖 phrases
4) Drops exact-duplicate texts across all splits
5) Re-splits into train/val/test with stratified labels
6) Writes cleaned splits and a mixed_sep file for review
7) Saves a JSON audit log with counts
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


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


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Clean combined_v2 dataset.")
    parser.add_argument(
        "--input-dir",
        default=str(PROJECT_ROOT / "datasets" / "combined_v2"),
        help="Input directory containing train/val/test CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "datasets" / "combined_v2_clean"),
        help="Output directory for cleaned dataset.",
    )
    parser.add_argument(
        "--remove-phrases",
        action="store_true",
        help="Remove explicit AI/refusal phrases (recommended).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for re-splitting.",
    )
    return parser.parse_args()


def load_split(path: Path) -> pd.DataFrame:
    """Load CSV split with basic error handling."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing dataset file: {path}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to load CSV: {path}") from exc


def split_sep(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows by presence of [SEP] in text."""
    text_series = df["text"].astype(str)
    mask = text_series.str.contains(r"\[SEP\]", regex=True, na=False)
    return df[~mask].copy(), df[mask].copy()


def split_phrases(df: pd.DataFrame, phrases: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows by explicit AI/refusal phrases."""
    if not phrases:
        return df.copy(), df.iloc[0:0].copy()

    pattern = "|".join(re.escape(p) for p in phrases)
    text_series = df["text"].astype(str)
    mask = text_series.str.contains(pattern, regex=True, case=False, na=False)
    return df[~mask].copy(), df[mask].copy()


def ensure_length_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure length column exists and matches text length."""
    if "length" not in df.columns:
        df["length"] = df["text"].astype(str).str.len()
    return df


def stratified_split(
    df: pd.DataFrame, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataframe into train/val/test with stratified labels when possible."""
    try:
        train, temp = train_test_split(
            df,
            test_size=0.2,
            random_state=seed,
            stratify=df["label"],
        )
        val, test = train_test_split(
            temp,
            test_size=0.5,
            random_state=seed,
            stratify=temp["label"],
        )
        return train, val, test
    except ValueError:
        train, temp = train_test_split(df, test_size=0.2, random_state=seed)
        val, test = train_test_split(temp, test_size=0.5, random_state=seed)
        return train, val, test


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write dataframe to CSV with UTF-8 encoding."""
    try:
        df.to_csv(path, index=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to write CSV: {path}") from exc


def write_json(data: Dict, path: Path) -> None:
    """Write JSON with UTF-8 encoding."""
    try:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        raise RuntimeError(f"Failed to write JSON: {path}") from exc


def main() -> None:
    """Main entry point."""
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = input_dir / "train.csv"
    val_path = input_dir / "val.csv"
    test_path = input_dir / "test.csv"

    train_df = load_split(train_path)
    val_df = load_split(val_path)
    test_df = load_split(test_path)

    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    log: Dict[str, Dict] = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "rows_before": len(combined_df),
    }

    clean_df, sep_df = split_sep(combined_df)
    log["sep_removed"] = len(sep_df)

    if args.remove_phrases:
        clean_df, phrase_df = split_phrases(clean_df, DEFAULT_PHRASES)
        log["phrase_removed"] = len(phrase_df)
    else:
        phrase_df = clean_df.iloc[0:0].copy()
        log["phrase_removed"] = 0

    clean_df = clean_df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    log["duplicates_removed"] = log["rows_before"] - len(sep_df) - len(phrase_df) - len(
        clean_df
    )

    clean_df = ensure_length_column(clean_df)

    train_clean, val_clean, test_clean = stratified_split(clean_df, args.seed)
    log["rows_after"] = len(clean_df)
    log["split_counts"] = {
        "train": len(train_clean),
        "val": len(val_clean),
        "test": len(test_clean),
    }

    write_csv(train_clean, output_dir / "train.csv")
    write_csv(val_clean, output_dir / "val.csv")
    write_csv(test_clean, output_dir / "test.csv")

    if len(sep_df) > 0:
        write_csv(sep_df, output_dir / "mixed_sep.csv")
    if len(phrase_df) > 0:
        write_csv(phrase_df, output_dir / "phrase_removed.csv")

    write_json(log, output_dir / "cleaning_log.json")

    print(f"Cleaned dataset saved to: {output_dir}")
    print(json.dumps(log, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
