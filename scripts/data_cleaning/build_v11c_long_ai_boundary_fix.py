#!/usr/bin/env python3
"""Build V11c by restoring long-AI bucket coverage to protect decision boundary."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
from typing import Dict, Iterable, List, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.utils.risk_patterns import PATTERNS

DEFAULT_POOLS = (
    "datasets/my_generated_ai/all_generated.jsonl",
    "datasets/generated/merged_2026-01-29/merged_ai_generated.jsonl",
    "datasets/generated/scenario_fill/2026-01-27_10h_multi_proxies/final_deduplicated.jsonl",
)

BUCKET_ORDER = ["0-64", "64-128", "128-256", "256-512", "512-1024", "1024-2048", "2048+"]
BUCKET_BINS = [0, 64, 128, 256, 512, 1024, 2048, 100000]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Restore long-AI bucket coverage by adding trusted AI supplements."
    )
    parser.add_argument(
        "--base-train",
        default=str(ROOT / "datasets/merged_v2/train_v11b_candidate.csv"),
        help="Base train CSV path (typically V11b).",
    )
    parser.add_argument(
        "--reference-train",
        default=str(ROOT / "datasets/merged_v2/train_v10.csv"),
        help="Reference train CSV path used for target AI bucket counts.",
    )
    parser.add_argument(
        "--fair-test-dir",
        default=str(ROOT / "datasets/eval/fair_test"),
        help="Fair-test CSV directory used for leakage exclusion.",
    )
    parser.add_argument(
        "--pool-jsonl",
        nargs="+",
        default=[str(ROOT / p) for p in DEFAULT_POOLS],
        help="JSONL AI pools to sample supplements from.",
    )
    parser.add_argument(
        "--target-retention-ratio",
        type=float,
        default=1.0,
        help="Target AI bucket ratio against reference train (1.0 means match reference).",
    )
    parser.add_argument(
        "--target-buckets",
        nargs="+",
        default=["256-512", "512-1024", "1024-2048", "2048+"],
        choices=BUCKET_ORDER,
        help="Length buckets to restore.",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=10,
        help="Minimum text length.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=8000,
        help="Maximum text length.",
    )
    parser.add_argument(
        "--max-per-model-share",
        type=float,
        default=0.45,
        help="Soft cap per model family within each bucket sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Sampling seed.",
    )
    parser.add_argument(
        "--supplement-output",
        default=str(ROOT / "datasets/merged_v2/long_ai_boundary_supplement_v11c.csv"),
        help="Boundary supplement CSV path.",
    )
    parser.add_argument(
        "--train-output",
        default=str(ROOT / "datasets/merged_v2/train_v11c_candidate.csv"),
        help="Output merged V11c train CSV path.",
    )
    parser.add_argument(
        "--summary-json",
        default=str(ROOT / "datasets/merged_v2/train_v11c_boundary_fix_summary.json"),
        help="Summary JSON path.",
    )
    parser.add_argument(
        "--summary-md",
        default=str(ROOT / "docs/plans/v11c_long_ai_boundary_fix.md"),
        help="Summary markdown path.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    """Ensure parent directory exists."""
    path.parent.mkdir(parents=True, exist_ok=True)


def text_sha1(text: str) -> str:
    """Hash text."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def normalize_prefix(text: str) -> str:
    """Normalize text for simple diversity stats."""
    return re.sub(r"\W+", "", text.strip().lower())[:80]


def has_hard_pattern(text: str) -> bool:
    """Check hard-risk pattern."""
    return any(pattern.search(text) for pattern in PATTERNS.hard_remove.values())


def model_family(model_name: str) -> str:
    """Map raw model name into a coarse family."""
    name = (model_name or "").lower()
    if "llama" in name:
        return "llama"
    if "deepseek" in name:
        return "deepseek"
    if "gpt-5" in name:
        return "gpt5"
    if "gpt-4" in name:
        return "gpt4"
    if "gpt-oss" in name:
        return "gpt-oss"
    if "glm" in name:
        return "glm"
    if "gemini" in name:
        return "gemini"
    if "qwen" in name:
        return "qwen"
    return "other"


def add_length_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Add char_len and bucket columns."""
    out = df.copy()
    out["char_len"] = out["text"].astype(str).str.len()
    out["length_bucket"] = pd.cut(
        out["char_len"],
        bins=BUCKET_BINS,
        labels=BUCKET_ORDER,
        right=False,
    )
    return out


def collect_exclude_hashes(base_train: Path, fair_test_dir: Path) -> Dict[str, set[str]]:
    """Collect hash exclusions from base train and fair_test."""
    base_df = pd.read_csv(base_train, encoding="utf-8-sig", usecols=["text"])
    base_hash = set(base_df["text"].astype(str).map(text_sha1).tolist())

    fair_hash: set[str] = set()
    for csv_path in sorted(fair_test_dir.glob("*.csv")):
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        if "text" in df.columns:
            fair_hash.update(df["text"].astype(str).map(text_sha1).tolist())

    return {"base": base_hash, "fair": fair_hash}


def iter_jsonl(path: Path) -> Iterable[dict]:
    """Yield JSON objects from JSONL file."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def collect_pool(
    pool_paths: Sequence[Path],
    exclude_hashes: set[str],
    min_length: int,
    max_length: int,
    target_buckets: set[str],
) -> pd.DataFrame:
    """Collect eligible AI candidates from JSONL pools."""
    rows: List[Dict[str, object]] = []
    seen_hash: set[str] = set()

    for pool_path in pool_paths:
        if not pool_path.exists():
            continue
        for item in iter_jsonl(pool_path):
            text = str(item.get("text", "")).strip()
            if not text:
                continue

            label_raw = item.get("label", 1)
            try:
                label = int(label_raw)
            except (TypeError, ValueError):
                label = 1
            if label != 1:
                continue

            text_len = len(text)
            if text_len < min_length or text_len > max_length:
                continue
            if has_hard_pattern(text):
                continue

            h = text_sha1(text)
            if h in exclude_hashes or h in seen_hash:
                continue

            bucket = pd.cut(
                pd.Series([text_len]),
                bins=BUCKET_BINS,
                labels=BUCKET_ORDER,
                right=False,
            ).iloc[0]
            if str(bucket) not in target_buckets:
                continue

            raw_model = str(item.get("model", "")).strip()
            fam = model_family(raw_model)
            rows.append(
                {
                    "text": text,
                    "label": 1,
                    "source": f"v11c_long_ai_{fam}",
                    "origin_source": str(item.get("source", "unknown")).strip(),
                    "origin_pool": str(pool_path),
                    "model": raw_model,
                    "model_family": fam,
                    "char_len": text_len,
                    "length_bucket": str(bucket),
                    "text_sha1": h,
                }
            )
            seen_hash.add(h)

    return pd.DataFrame(rows)


def sample_bucket(
    df: pd.DataFrame,
    n: int,
    max_per_model_share: float,
    seed: int,
) -> pd.DataFrame:
    """Sample one bucket with model-family diversity."""
    if n <= 0:
        return df.iloc[0:0].copy()
    if len(df) < n:
        raise ValueError(f"Not enough candidates in bucket: need {n}, got {len(df)}")

    rng = random.Random(seed)
    by_family: Dict[str, List[dict]] = defaultdict(list)
    for _, row in df.iterrows():
        by_family[str(row["model_family"])].append(row.to_dict())

    for family in by_family:
        rng.shuffle(by_family[family])

    family_quota = {
        family: max(1, math.ceil(n * max_per_model_share)) for family in by_family.keys()
    }
    family_selected = Counter()
    selected: List[dict] = []
    family_order = sorted(by_family.keys())
    idx_by_family = {family: 0 for family in family_order}

    progress = True
    while len(selected) < n and progress:
        progress = False
        for family in family_order:
            idx = idx_by_family[family]
            if idx >= len(by_family[family]):
                continue
            if family_selected[family] >= family_quota[family]:
                continue
            selected.append(by_family[family][idx])
            idx_by_family[family] += 1
            family_selected[family] += 1
            progress = True
            if len(selected) >= n:
                break

    if len(selected) < n:
        for family in family_order:
            while idx_by_family[family] < len(by_family[family]) and len(selected) < n:
                selected.append(by_family[family][idx_by_family[family]])
                idx_by_family[family] += 1
            if len(selected) >= n:
                break

    return pd.DataFrame(selected)


def bucket_counts_ai(df: pd.DataFrame) -> Dict[str, int]:
    """Count AI rows by bucket."""
    work = add_length_bucket(df)
    ai = work[work["label"].astype(int) == 1]
    counts = ai["length_bucket"].value_counts().to_dict()
    return {bucket: int(counts.get(bucket, 0)) for bucket in BUCKET_ORDER}


def render_markdown(summary: Dict[str, object]) -> str:
    """Render markdown summary."""
    lines = [
        "# V11c Long-AI Boundary Fix",
        "",
        f"- Generated at: {summary['generated_at']}",
        f"- Base train: `{summary['base_train_path']}`",
        f"- Reference train: `{summary['reference_train_path']}`",
        f"- Supplement rows: {summary['supplement_rows']}",
        f"- Train rows: {summary['base_rows']} -> {summary['train_rows_after_merge']}",
        "",
        "## Bucket restoration",
        "",
        "| bucket | reference_ai | base_ai | target_ai | added | after_ai |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for row in summary["bucket_plan"]:
        lines.append(
            f"| {row['bucket']} | {row['reference_ai']} | {row['base_ai']} | {row['target_ai']} | "
            f"{row['added']} | {row['after_ai']} |"
        )

    lines.extend(
        [
            "",
            "## Leakage checks",
            "",
            f"- overlap with base train: {summary['leakage']['overlap_with_base_train']}",
            f"- overlap with fair_test: {summary['leakage']['overlap_with_fair_test']}",
            "",
            "## Model-family distribution (supplement)",
            "",
        ]
    )
    for family, count in summary["supplement_model_family_top"].items():
        lines.append(f"- {family}: {count}")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Entry point."""
    args = parse_args()

    base_train_path = Path(args.base_train)
    reference_train_path = Path(args.reference_train)
    fair_test_dir = Path(args.fair_test_dir)
    pool_paths = [Path(path) for path in args.pool_jsonl]
    target_buckets = set(args.target_buckets)

    supplement_output = Path(args.supplement_output)
    train_output = Path(args.train_output)
    summary_json = Path(args.summary_json)
    summary_md = Path(args.summary_md)
    for path in (supplement_output, train_output, summary_json, summary_md):
        ensure_parent(path)

    base_df = pd.read_csv(base_train_path, encoding="utf-8-sig")
    ref_df = pd.read_csv(reference_train_path, encoding="utf-8-sig")
    for df_name, df in (("base", base_df), ("reference", ref_df)):
        if not {"text", "label"}.issubset(df.columns):
            raise ValueError(f"{df_name} train must contain text/label columns.")
        if "source" not in df.columns:
            df["source"] = "unknown"

    base_ai_bucket = bucket_counts_ai(base_df)
    ref_ai_bucket = bucket_counts_ai(ref_df)

    bucket_plan: List[Dict[str, int]] = []
    total_needed = 0
    for bucket in BUCKET_ORDER:
        reference_ai = ref_ai_bucket.get(bucket, 0)
        base_ai = base_ai_bucket.get(bucket, 0)
        target_ai = int(round(reference_ai * args.target_retention_ratio))
        need = max(0, target_ai - base_ai) if bucket in target_buckets else 0
        total_needed += need
        bucket_plan.append(
            {
                "bucket": bucket,
                "reference_ai": reference_ai,
                "base_ai": base_ai,
                "target_ai": target_ai,
                "need": need,
            }
        )

    exclude = collect_exclude_hashes(base_train_path, fair_test_dir)
    pool_df = collect_pool(
        pool_paths=pool_paths,
        exclude_hashes=exclude["base"] | exclude["fair"],
        min_length=args.min_length,
        max_length=args.max_length,
        target_buckets=target_buckets,
    )
    if pool_df.empty and total_needed > 0:
        raise ValueError("Candidate pool is empty after filters.")

    selected_parts: List[pd.DataFrame] = []
    for plan_row in bucket_plan:
        bucket = plan_row["bucket"]
        need = plan_row["need"]
        if need <= 0:
            plan_row["pool_usable"] = int((pool_df["length_bucket"] == bucket).sum())
            plan_row["added"] = 0
            plan_row["after_ai"] = plan_row["base_ai"]
            continue

        bucket_pool = pool_df[pool_df["length_bucket"] == bucket].copy()
        plan_row["pool_usable"] = len(bucket_pool)
        if len(bucket_pool) < need:
            raise ValueError(
                f"Bucket {bucket} needs {need} rows, but only {len(bucket_pool)} available."
            )

        sampled = sample_bucket(
            df=bucket_pool,
            n=need,
            max_per_model_share=args.max_per_model_share,
            seed=args.seed + BUCKET_ORDER.index(bucket) * 17,
        )
        selected_parts.append(sampled)
        plan_row["added"] = len(sampled)
        plan_row["after_ai"] = plan_row["base_ai"] + len(sampled)

    supplement_df = (
        pd.concat(selected_parts, ignore_index=True)
        if selected_parts
        else pd.DataFrame(columns=pool_df.columns)
    )
    supplement_df = supplement_df.drop_duplicates(subset=["text"], keep="first").reset_index(
        drop=True
    )
    supplement_df.to_csv(supplement_output, index=False, encoding="utf-8-sig")

    merge_cols = ["text", "label", "source"]
    merged_df = pd.concat([base_df[merge_cols], supplement_df[merge_cols]], ignore_index=True)
    before_dedup = len(merged_df)
    merged_df = merged_df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    dedup_drop = before_dedup - len(merged_df)
    merged_df.to_csv(train_output, index=False, encoding="utf-8-sig")

    supplement_hashes = set(supplement_df["text"].astype(str).map(text_sha1).tolist())
    overlap_with_base = len(supplement_hashes & exclude["base"])
    overlap_with_fair = len(supplement_hashes & exclude["fair"])

    after_ai_bucket = bucket_counts_ai(merged_df)

    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "base_train_path": str(base_train_path),
        "reference_train_path": str(reference_train_path),
        "supplement_output": str(supplement_output),
        "train_output": str(train_output),
        "target_retention_ratio": args.target_retention_ratio,
        "target_buckets": sorted(target_buckets),
        "base_rows": len(base_df),
        "supplement_rows": len(supplement_df),
        "train_rows_after_merge": len(merged_df),
        "train_dedup_drop_after_merge": dedup_drop,
        "bucket_plan": [
            {
                **row,
                "after_ai": int(after_ai_bucket.get(row["bucket"], row.get("after_ai", 0))),
            }
            for row in bucket_plan
        ],
        "supplement_model_family_top": supplement_df["model_family"].value_counts()
        .head(20)
        .to_dict(),
        "supplement_bucket_counts": supplement_df["length_bucket"].value_counts().to_dict(),
        "supplement_diversity": {
            "unique_prefix_ratio": round(
                (
                    len({normalize_prefix(t) for t in supplement_df["text"].astype(str)})
                    / max(len(supplement_df), 1)
                ),
                4,
            ),
            "top_prefix_share": round(
                (
                    Counter(normalize_prefix(t) for t in supplement_df["text"].astype(str))
                    .most_common(1)[0][1]
                    / max(len(supplement_df), 1)
                )
                if len(supplement_df)
                else 0.0,
                4,
            ),
        },
        "leakage": {
            "overlap_with_base_train": overlap_with_base,
            "overlap_with_fair_test": overlap_with_fair,
        },
    }

    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_md.write_text(render_markdown(summary), encoding="utf-8")

    print(f"[OK] supplement: {supplement_output}")
    print(f"[OK] train_v11c: {train_output}")
    print(f"[OK] summary_json: {summary_json}")
    print(f"[OK] summary_md: {summary_md}")
    print(
        "[INFO] supplement rows / leakage: "
        f"{len(supplement_df)} / base_overlap={overlap_with_base}, fair_overlap={overlap_with_fair}"
    )


if __name__ == "__main__":
    main()

