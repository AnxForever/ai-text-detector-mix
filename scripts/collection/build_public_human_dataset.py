#!/usr/bin/env python3
"""
Build a combined public human dataset from THUCNews and Wikipedia.
"""
import argparse
import csv
import random
from pathlib import Path
from typing import List, Dict

import pandas as pd


def find_latest_wikipedia_file(directory: Path) -> Path:
    candidates = sorted(directory.glob("wikipedia_*_texts.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else Path()


def normalize_wikipedia(df: pd.DataFrame) -> pd.DataFrame:
    if "text" not in df.columns and "text_content" in df.columns:
        df = df.rename(columns={"text_content": "text"})

    df["source"] = df.get("source", "wikipedia")
    if "title" in df.columns:
        df["topic"] = df["title"].fillna("百科")
    else:
        df["topic"] = "百科"
    df["attribute"] = "说明"

    if "length" not in df.columns:
        df["length"] = df["text"].astype(str).str.len()

    if "timestamp" not in df.columns:
        df["timestamp"] = ""

    return df[["text", "source", "attribute", "topic", "length", "timestamp"]]


def normalize_thucnews(df: pd.DataFrame) -> pd.DataFrame:
    if "text" not in df.columns and "text_content" in df.columns:
        df = df.rename(columns={"text_content": "text"})
    if "attribute" not in df.columns:
        df["attribute"] = "说明"
    if "topic" not in df.columns:
        df["topic"] = "未知"
    if "length" not in df.columns:
        df["length"] = df["text"].astype(str).str.len()
    if "timestamp" not in df.columns:
        df["timestamp"] = ""
    if "source" not in df.columns:
        df["source"] = "thucnews"
    return df[["text", "source", "attribute", "topic", "length", "timestamp"]]


def sample_df(df: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    if count <= 0:
        return df.head(0)
    if len(df) <= count:
        return df
    return df.sample(n=count, random_state=seed)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build combined public human dataset.")
    parser.add_argument("--ai-file", default="datasets/final/parallel_dataset_cleaned.csv", help="AI dataset path")
    parser.add_argument("--thucnews-file", default="datasets/human_texts/thucnews_real_human_9000.csv",
                        help="THUCNews human dataset path")
    parser.add_argument("--wikipedia-file", default="", help="Wikipedia dataset path (optional)")
    parser.add_argument("--wiki-ratio", type=float, default=0.2, help="Target Wikipedia ratio (0-1)")
    parser.add_argument("--target-count", type=int, default=0, help="Target total human count")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default="datasets/human_texts/human_public_combined.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    ai_path = Path(args.ai_file)
    if not ai_path.exists():
        print(f"AI file not found: {ai_path}")
        return 1

    ai_count = sum(1 for _ in csv.DictReader(ai_path.open("r", encoding="utf-8-sig", newline="")))
    target_count = args.target_count or ai_count

    thucnews_path = Path(args.thucnews_file)
    if not thucnews_path.exists():
        print(f"THUCNews file not found: {thucnews_path}")
        return 1

    wiki_path = Path(args.wikipedia_file) if args.wikipedia_file else find_latest_wikipedia_file(thucnews_path.parent)
    if wiki_path and wiki_path.exists():
        df_wiki = normalize_wikipedia(pd.read_csv(wiki_path, encoding="utf-8-sig"))
    else:
        df_wiki = pd.DataFrame(columns=["text", "source", "attribute", "topic", "length", "timestamp"])
        wiki_path = Path()

    df_thuc = normalize_thucnews(pd.read_csv(thucnews_path, encoding="utf-8-sig"))

    wiki_target = int(target_count * args.wiki_ratio)
    wiki_target = min(wiki_target, len(df_wiki))
    thuc_target = target_count - wiki_target
    if len(df_thuc) < thuc_target:
        thuc_target = len(df_thuc)
        wiki_target = min(target_count - thuc_target, len(df_wiki))

    random.seed(args.seed)
    df_thuc_sampled = sample_df(df_thuc, thuc_target, args.seed)
    df_wiki_sampled = sample_df(df_wiki, wiki_target, args.seed + 1)

    combined = pd.concat([df_thuc_sampled, df_wiki_sampled], ignore_index=True)
    combined = combined.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"AI count: {ai_count}")
    print(f"Target human count: {target_count}")
    print(f"THUCNews used: {len(df_thuc_sampled)}")
    print(f"Wikipedia used: {len(df_wiki_sampled)}")
    print(f"Output: {output_path}")
    if wiki_path:
        print(f"Wikipedia source: {wiki_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
