#!/usr/bin/env python3
"""
Build source-aware hard/easy subsets from existing train and evaluation CSV files.

This script annotates rows into:
    - human_hard
    - human_easy
    - human_other
    - ai_hard
    - ai_easy
    - ai_other

It is designed for two downstream uses:
1. Construct a reusable hard evaluation set without collecting new data.
2. Extract hard training candidates for reweighting / resampling experiments.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

os.environ["PYTHONIOENCODING"] = "utf-8"

ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_INPUTS: Sequence[Tuple[str, str, str]] = (
    ("train_v11c", "datasets/merged_v2/train_v11c_candidate.csv", "train"),
    ("core_v1_test_clean", "datasets/eval/fair_test/core_v1_test_clean.csv", "eval"),
    ("independent_data", "datasets/eval/fair_test/independent_data.csv", "eval"),
    ("merged_v2_val_clean", "datasets/eval/fair_test/merged_v2_val_clean.csv", "eval"),
)

HUMAN_HARD_EXACT = {
    "formal_collected",
    "Wikipedia_CN",
    "external_m4_qazh",
    "external_vcsum",
    "Review_ChnSentiCorp_htl_all",
    "Review_waimai_10k",
    "hc3_human_casual",
    "hc3_human_hesitant",
    "hc3_human_student",
}

HUMAN_HARD_PREFIXES = (
    "学术摘要-",
    "新华社新闻-",
    "实验室管理处通知-",
    "学生资助中心通知-",
)

HUMAN_EASY_EXACT = {
    "hc3_human",
    "thucnews",
    "Toutiao_News",
    "Toutiao_news_tech",
    "Toutiao_news_finance",
    "Toutiao_news_edu",
    "Toutiao_news_military",
    "Toutiao_news_car",
    "Toutiao_news_agriculture",
    "Toutiao_news_game",
}

AI_HARD_PREFIXES = (
    "p0_",
    "parallel_",
    "auto_",
    "real_ai_",
)

AI_HARD_EXACT = {
    "parallel_claude-4-sonnet",
    "parallel_claude-4.5-sonnet",
}

AI_EASY_EXACT = {
    "hc3_chatgpt",
    "Senior_qwen",
    "Senior_deepseek",
}

AI_EASY_PREFIXES = (
    "v11c_long_ai_",
    "v10_edu_ai_",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build source-aware hard/easy subsets from current train/eval CSV files."
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "datasets" / "eval" / "hard_test"),
        help="Directory to save annotated CSVs and summary JSON.",
    )
    parser.add_argument(
        "--max-per-source",
        type=int,
        default=0,
        help="Optional cap per source when exporting hard eval/train sets (0 means no cap).",
    )
    parser.add_argument(
        "--balance-hard-eval",
        action="store_true",
        help="Export an additional balanced hard evaluation set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for balanced sampling.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    """Create a directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def load_input_csv(name: str, rel_path: str, role: str) -> pd.DataFrame:
    """Load one CSV file and normalize required columns."""
    csv_path = ROOT / rel_path
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except Exception as exc:
        raise RuntimeError(f"Failed to read {csv_path}: {exc}") from exc

    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {sorted(missing)}")

    out = df.copy()
    out["text"] = out["text"].astype(str)
    out["label"] = out["label"].astype(int)
    if "source" not in out.columns:
        out["source"] = "unknown"
    out["source"] = out["source"].fillna("unknown").astype(str)
    out["dataset_name"] = name
    out["dataset_role"] = role
    out["char_len"] = out["text"].str.len()
    keep_cols = [
        "text",
        "label",
        "source",
        "dataset_name",
        "dataset_role",
        "char_len",
    ]
    for optional_col in ("category", "length", "origin_dataset", "origin_split"):
        if optional_col in out.columns:
            keep_cols.append(optional_col)
    return out[keep_cols]


def starts_with_any(value: str, prefixes: Sequence[str]) -> bool:
    """Return True if a value starts with any prefix."""
    return any(value.startswith(prefix) for prefix in prefixes)


def assign_subset(label: int, source: str) -> Tuple[str, str]:
    """Assign a row to hard/easy/other subset and reason."""
    src = source or "unknown"

    if label == 0:
        if src in HUMAN_HARD_EXACT:
            return "human_hard", "human_hard_exact"
        if starts_with_any(src, HUMAN_HARD_PREFIXES):
            return "human_hard", "human_hard_prefix"
        if src in HUMAN_EASY_EXACT:
            return "human_easy", "human_easy_exact"
        return "human_other", "human_unmapped"

    if starts_with_any(src, AI_HARD_PREFIXES) or src in AI_HARD_EXACT:
        return "ai_hard", "ai_hard_source_family"
    if src in AI_EASY_EXACT or starts_with_any(src, AI_EASY_PREFIXES):
        return "ai_easy", "ai_easy_source_family"
    return "ai_other", "ai_unmapped"


def annotate_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Annotate subset and difficulty columns."""
    out = df.copy()
    subset_reason = out.apply(
        lambda row: assign_subset(int(row["label"]), str(row["source"])),
        axis=1,
    )
    out["subset"] = subset_reason.map(lambda item: item[0])
    out["subset_reason"] = subset_reason.map(lambda item: item[1])
    out["difficulty"] = out["subset"].map(
        lambda value: "hard" if value.endswith("hard") else "easy" if value.endswith("easy") else "other"
    )
    return out


def cap_per_source(df: pd.DataFrame, max_per_source: int, seed: int) -> pd.DataFrame:
    """Optionally cap rows per source."""
    if max_per_source <= 0:
        return df.reset_index(drop=True)

    parts: List[pd.DataFrame] = []
    for _, group in df.groupby("source", sort=True):
        if len(group) <= max_per_source:
            parts.append(group)
        else:
            parts.append(group.sample(n=max_per_source, random_state=seed))
    return pd.concat(parts, ignore_index=True).reset_index(drop=True)


def build_balanced_hard_eval(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Build a balanced hard evaluation set across human_hard and ai_hard."""
    hard = df[df["subset"].isin({"human_hard", "ai_hard"})].copy()
    if hard.empty:
        return hard

    counts = hard["subset"].value_counts()
    if "human_hard" not in counts or "ai_hard" not in counts:
        return hard

    target = int(min(counts["human_hard"], counts["ai_hard"]))
    parts = []
    for subset in ("human_hard", "ai_hard"):
        part = hard[hard["subset"] == subset]
        if len(part) > target:
            part = part.sample(n=target, random_state=seed)
        parts.append(part)
    return pd.concat(parts, ignore_index=True).sort_values(
        ["subset", "source", "dataset_name"]
    ).reset_index(drop=True)


def top_sources(df: pd.DataFrame, top_n: int = 20) -> List[Dict[str, object]]:
    """Summarize top sources inside a dataframe."""
    if df.empty:
        return []
    summary = (
        df.groupby(["subset", "source"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["subset", "count", "source"], ascending=[True, False, True])
    )
    return summary.head(top_n).to_dict(orient="records")


def make_summary(all_df: pd.DataFrame, train_df: pd.DataFrame, eval_df: pd.DataFrame) -> Dict[str, object]:
    """Build a JSON-serializable summary."""
    summary: Dict[str, object] = {
        "inputs": [
            {"name": name, "path": rel_path, "role": role}
            for name, rel_path, role in DEFAULT_INPUTS
        ],
        "all_rows": int(len(all_df)),
        "train_rows": int(len(train_df)),
        "eval_rows": int(len(eval_df)),
        "all_subset_counts": all_df["subset"].value_counts().sort_index().to_dict(),
        "train_subset_counts": train_df["subset"].value_counts().sort_index().to_dict(),
        "eval_subset_counts": eval_df["subset"].value_counts().sort_index().to_dict(),
        "all_difficulty_counts": all_df["difficulty"].value_counts().sort_index().to_dict(),
        "eval_hard_top_sources": top_sources(eval_df[eval_df["difficulty"] == "hard"]),
        "train_hard_top_sources": top_sources(train_df[train_df["difficulty"] == "hard"]),
        "unmapped_sources": (
            all_df[all_df["subset_reason"].str.endswith("unmapped")]
            .groupby(["label", "source"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(30)
            .to_dict(orient="records")
        ),
    }
    return summary


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """Save a dataframe to CSV with consistent encoding."""
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
    except Exception as exc:
        raise RuntimeError(f"Failed to write CSV {path}: {exc}") from exc


def save_json(data: Dict[str, object], path: Path) -> None:
    """Save JSON summary."""
    try:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to write JSON {path}: {exc}") from exc


def main() -> None:
    """Entry point."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    frames: List[pd.DataFrame] = []
    for name, rel_path, role in DEFAULT_INPUTS:
        frame = load_input_csv(name=name, rel_path=rel_path, role=role)
        frames.append(frame)

    all_df = annotate_frame(pd.concat(frames, ignore_index=True))
    train_df = all_df[all_df["dataset_role"] == "train"].copy()
    eval_df = all_df[all_df["dataset_role"] == "eval"].copy()

    hard_train = train_df[train_df["subset"].isin({"human_hard", "ai_hard"})].copy()
    hard_eval = eval_df[eval_df["subset"].isin({"human_hard", "ai_hard"})].copy()

    hard_train = cap_per_source(hard_train, max_per_source=args.max_per_source, seed=args.seed)
    hard_eval = cap_per_source(hard_eval, max_per_source=args.max_per_source, seed=args.seed)

    balanced_hard_eval = (
        build_balanced_hard_eval(hard_eval, seed=args.seed)
        if args.balance_hard_eval
        else pd.DataFrame(columns=hard_eval.columns)
    )

    save_csv(all_df, output_dir / "annotated_all.csv")
    save_csv(train_df, output_dir / "annotated_train.csv")
    save_csv(eval_df, output_dir / "annotated_eval.csv")
    save_csv(hard_train, output_dir / "hard_train_set.csv")
    save_csv(hard_eval, output_dir / "hard_eval_set.csv")
    save_csv(eval_df[eval_df["subset"] == "human_hard"], output_dir / "human_hard_eval.csv")
    save_csv(eval_df[eval_df["subset"] == "ai_hard"], output_dir / "ai_hard_eval.csv")
    if args.balance_hard_eval:
        save_csv(balanced_hard_eval, output_dir / "hard_eval_balanced.csv")

    summary = make_summary(all_df=all_df, train_df=train_df, eval_df=eval_df)
    if args.balance_hard_eval:
        summary["balanced_hard_eval_rows"] = int(len(balanced_hard_eval))
    save_json(summary, output_dir / "summary.json")

    print("=" * 70)
    print("Hard-set build complete")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Annotated rows: {len(all_df)}")
    print(f"Train subset counts: {summary['train_subset_counts']}")
    print(f"Eval subset counts: {summary['eval_subset_counts']}")
    print(f"Hard train rows: {len(hard_train)}")
    print(f"Hard eval rows: {len(hard_eval)}")
    if args.balance_hard_eval:
        print(f"Balanced hard eval rows: {len(balanced_hard_eval)}")


if __name__ == "__main__":
    main()
