#!/usr/bin/env python3
"""Build V11b train set by adding weak-domain supplements to V11a."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Dict, Iterable, List, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.utils.risk_patterns import PATTERNS

DEFAULT_FORMAL_POOLS = (
    "datasets/human_formal_samples.jsonl",
    "datasets/human_formal_manual.jsonl",
)
DEFAULT_LLAMA_POOLS = (
    "datasets/generated/merged_2026-01-29/merged_ai_generated.jsonl",
    "datasets/my_generated_ai/by_model/meta_llama-3.1-405b-instruct.jsonl",
)


@dataclass(frozen=True)
class CandidateRecord:
    """Stores candidate text with lightweight metadata."""

    text: str
    label: int
    source: str
    origin_file: str
    origin_source: str
    group_key: str


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build train_v11b_candidate.csv with weak-domain supplements."
    )
    parser.add_argument(
        "--base-train",
        default=str(ROOT / "datasets/merged_v2/train_v11_candidate.csv"),
        help="Base train CSV path (V11a).",
    )
    parser.add_argument(
        "--fair-test-dir",
        default=str(ROOT / "datasets/eval/fair_test"),
        help="Directory containing fair_test CSV files for leakage exclusion.",
    )
    parser.add_argument(
        "--formal-pools",
        nargs="+",
        default=[str(ROOT / p) for p in DEFAULT_FORMAL_POOLS],
        help="JSONL pools for formal human supplement.",
    )
    parser.add_argument(
        "--llama-pools",
        nargs="+",
        default=[str(ROOT / p) for p in DEFAULT_LLAMA_POOLS],
        help="JSONL pools for LLaMA AI supplement.",
    )
    parser.add_argument(
        "--target-formal",
        type=int,
        default=300,
        help="Target rows for formal_collected supplement.",
    )
    parser.add_argument(
        "--target-llama",
        type=int,
        default=300,
        help="Target rows for real_ai_llama-3.1-405b-instruct supplement.",
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
        "--seed",
        type=int,
        default=42,
        help="Sampling seed.",
    )
    parser.add_argument(
        "--supplement-output",
        default=str(ROOT / "datasets/merged_v2/weak_domain_supplement_v11b.csv"),
        help="Supplement CSV output path.",
    )
    parser.add_argument(
        "--supplement-summary-json",
        default=str(ROOT / "datasets/merged_v2/weak_domain_supplement_v11b_summary.json"),
        help="Supplement summary JSON path.",
    )
    parser.add_argument(
        "--train-output",
        default=str(ROOT / "datasets/merged_v2/train_v11b_candidate.csv"),
        help="Merged train_v11b CSV path.",
    )
    parser.add_argument(
        "--train-summary-json",
        default=str(ROOT / "datasets/merged_v2/train_v11b_candidate_summary.json"),
        help="Merged train summary JSON path.",
    )
    parser.add_argument(
        "--summary-md",
        default=str(ROOT / "docs/plans/weak_domain_supplement_build_v11b.md"),
        help="Human-readable summary markdown output path.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    """Ensure parent directory exists."""
    path.parent.mkdir(parents=True, exist_ok=True)


def text_sha1(text: str) -> str:
    """Return SHA1 hash for a text."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def normalize_prefix(text: str, length: int = 80) -> str:
    """Normalize text to a compact prefix for diversity metrics."""
    norm = re.sub(r"\W+", "", text.strip().lower())
    return norm[:length]


def has_hard_pattern(text: str) -> bool:
    """Check if text hits any hard-removal risk pattern."""
    return any(pattern.search(text) for pattern in PATTERNS.hard_remove.values())


def collect_exclude_hashes(base_train: Path, fair_test_dir: Path) -> Dict[str, set[str]]:
    """Build hash sets for train/fair leakage exclusion."""
    if not base_train.exists():
        raise FileNotFoundError(f"Base train not found: {base_train}")
    if not fair_test_dir.exists():
        raise FileNotFoundError(f"Fair test dir not found: {fair_test_dir}")

    base_df = pd.read_csv(base_train, encoding="utf-8-sig", usecols=["text"])
    base_hashes = set(base_df["text"].astype(str).map(text_sha1).tolist())

    fair_hashes: set[str] = set()
    for csv_path in sorted(fair_test_dir.glob("*.csv")):
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        if "text" not in df.columns:
            continue
        fair_hashes.update(df["text"].astype(str).map(text_sha1).tolist())

    return {"base": base_hashes, "fair": fair_hashes}


def iterate_jsonl(path: Path) -> Iterable[dict]:
    """Yield parsed JSON objects from a JSONL file."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def collect_formal_candidates(
    pool_paths: Sequence[Path],
    exclude_hashes: set[str],
    min_length: int,
    max_length: int,
) -> List[CandidateRecord]:
    """Collect formal human candidates from JSONL pools."""
    records: List[CandidateRecord] = []
    seen_hashes: set[str] = set()

    for pool_path in pool_paths:
        if not pool_path.exists():
            continue
        for item in iterate_jsonl(pool_path):
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            if not (min_length <= len(text) <= max_length):
                continue
            if has_hard_pattern(text):
                continue
            h = text_sha1(text)
            if h in exclude_hashes or h in seen_hashes:
                continue

            origin_source = str(item.get("source", "unknown")).strip()
            category = str(item.get("category", "")).strip() or "formal_misc"
            records.append(
                CandidateRecord(
                    text=text,
                    label=0,
                    source="formal_collected",
                    origin_file=str(pool_path),
                    origin_source=origin_source,
                    group_key=category,
                )
            )
            seen_hashes.add(h)
    return records


def collect_llama_candidates(
    pool_paths: Sequence[Path],
    exclude_hashes: set[str],
    min_length: int,
    max_length: int,
) -> List[CandidateRecord]:
    """Collect LLaMA AI candidates from JSONL pools."""
    records: List[CandidateRecord] = []
    seen_hashes: set[str] = set()
    model_key = "llama-3.1-405b-instruct"

    for pool_path in pool_paths:
        if not pool_path.exists():
            continue
        for item in iterate_jsonl(pool_path):
            model = str(item.get("model", "")).strip().lower()
            if model_key not in model:
                continue
            if int(item.get("label", -1)) != 1:
                continue

            text = str(item.get("text", "")).strip()
            if not text:
                continue
            if not (min_length <= len(text) <= max_length):
                continue
            if has_hard_pattern(text):
                continue
            h = text_sha1(text)
            if h in exclude_hashes or h in seen_hashes:
                continue

            category = str(item.get("category", "")).strip() or "llama_misc"
            scenario = str(item.get("scenario_id", "")).strip()
            group_key = f"{category}::{scenario}" if scenario else category
            origin_source = str(item.get("source", "unknown")).strip()
            records.append(
                CandidateRecord(
                    text=text,
                    label=1,
                    source="real_ai_llama-3.1-405b-instruct",
                    origin_file=str(pool_path),
                    origin_source=origin_source,
                    group_key=group_key,
                )
            )
            seen_hashes.add(h)
    return records


def stratified_sample(
    records: Sequence[CandidateRecord],
    target: int,
    seed: int,
) -> List[CandidateRecord]:
    """Sample records with coarse stratification by group key."""
    if len(records) < target:
        raise ValueError(f"Not enough candidates: need {target}, got {len(records)}")

    by_group: Dict[str, List[CandidateRecord]] = defaultdict(list)
    for record in records:
        by_group[record.group_key].append(record)

    rng = pd.Series(range(max(1, target))).sample(frac=1, random_state=seed)
    _ = rng  # quiet lint for deterministic seed setup

    selected: List[CandidateRecord] = []
    target_per_group = max(1, target // max(1, len(by_group)))

    for group_key, group_records in sorted(by_group.items()):
        group_seed = seed + int(hashlib.sha1(group_key.encode("utf-8")).hexdigest()[:8], 16) % 10000
        sampled = (
            pd.Series(group_records)
            .sample(
                n=min(target_per_group, len(group_records)),
                random_state=group_seed,
                replace=False,
            )
            .tolist()
        )
        selected.extend(sampled)

    if len(selected) < target:
        selected_hashes = {text_sha1(record.text) for record in selected}
        remaining = [record for record in records if text_sha1(record.text) not in selected_hashes]
        needed = target - len(selected)
        top_up = pd.Series(remaining).sample(n=needed, random_state=seed, replace=False).tolist()
        selected.extend(top_up)

    if len(selected) > target:
        selected = pd.Series(selected).sample(n=target, random_state=seed, replace=False).tolist()
    return selected


def diversity_metrics(records: Sequence[CandidateRecord]) -> Dict[str, float]:
    """Compute diversity metrics for a selected supplement slice."""
    if not records:
        return {
            "rows": 0,
            "unique_prefix_ratio": 0.0,
            "top_prefix_share": 0.0,
        }

    prefixes = [normalize_prefix(record.text) for record in records]
    prefix_counts = Counter(prefixes)
    rows = len(records)
    unique_prefix_ratio = len(prefix_counts) / rows
    top_prefix_share = max(prefix_counts.values()) / rows
    return {
        "rows": rows,
        "unique_prefix_ratio": round(unique_prefix_ratio, 4),
        "top_prefix_share": round(top_prefix_share, 4),
    }


def to_df(records: Sequence[CandidateRecord]) -> pd.DataFrame:
    """Convert records to the train CSV schema."""
    return pd.DataFrame(
        {
            "text": [record.text for record in records],
            "label": [record.label for record in records],
            "source": [record.source for record in records],
        }
    )


def build_markdown(summary: Dict[str, object]) -> str:
    """Render a compact markdown report."""
    lines = [
        "# V11b Weak Domain Supplement Build",
        "",
        f"- Generated at: {summary['generated_at']}",
        f"- Base train: `{summary['base_train_path']}`",
        f"- Supplement rows: {summary['supplement_rows']}",
        f"- Train rows: {summary['base_rows']} -> {summary['train_rows_after_merge']}",
        "",
        "## Domain targets",
        "",
        "| source | target | selected | pool_usable | unique_prefix_ratio | top_prefix_share |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for row in summary["domains"]:
        lines.append(
            f"| {row['source']} | {row['target']} | {row['selected']} | {row['pool_usable']} | "
            f"{row['diversity']['unique_prefix_ratio']} | {row['diversity']['top_prefix_share']} |"
        )

    lines.extend(
        [
            "",
            "## Leakage checks",
            "",
            f"- Supplement overlap with base train: {summary['leakage']['overlap_with_base_train']}",
            f"- Supplement overlap with fair_test: {summary['leakage']['overlap_with_fair_test']}",
            "",
            "## Notes",
            "",
            "- formal_collected supplement is sourced from non-fair pools and tagged for weak-domain "
            "coverage.",
            "- LLaMA supplement uses `model` filter `llama-3.1-405b-instruct` and excludes fair/test "
            "overlap.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Entry point."""
    args = parse_args()

    base_train_path = Path(args.base_train)
    fair_test_dir = Path(args.fair_test_dir)
    formal_pools = [Path(path) for path in args.formal_pools]
    llama_pools = [Path(path) for path in args.llama_pools]

    supplement_output = Path(args.supplement_output)
    supplement_summary_json = Path(args.supplement_summary_json)
    train_output = Path(args.train_output)
    train_summary_json = Path(args.train_summary_json)
    summary_md = Path(args.summary_md)

    for path in (
        supplement_output,
        supplement_summary_json,
        train_output,
        train_summary_json,
        summary_md,
    ):
        ensure_parent(path)

    base_df = pd.read_csv(base_train_path, encoding="utf-8-sig")
    if "text" not in base_df.columns or "label" not in base_df.columns:
        raise ValueError("Base train must contain `text` and `label` columns.")
    if "source" not in base_df.columns:
        base_df["source"] = "unknown"

    exclude = collect_exclude_hashes(base_train_path, fair_test_dir)
    all_exclude = exclude["base"] | exclude["fair"]

    formal_pool = collect_formal_candidates(
        pool_paths=formal_pools,
        exclude_hashes=all_exclude,
        min_length=args.min_length,
        max_length=args.max_length,
    )
    llama_pool = collect_llama_candidates(
        pool_paths=llama_pools,
        exclude_hashes=all_exclude,
        min_length=args.min_length,
        max_length=args.max_length,
    )

    formal_selected = stratified_sample(formal_pool, target=args.target_formal, seed=args.seed)
    llama_selected = stratified_sample(llama_pool, target=args.target_llama, seed=args.seed + 7)

    supplement_records = formal_selected + llama_selected
    supplement_df = to_df(supplement_records).drop_duplicates(subset=["text"], keep="first")
    supplement_df.to_csv(supplement_output, index=False, encoding="utf-8-sig")

    merged_df = pd.concat([base_df[["text", "label", "source"]], supplement_df], ignore_index=True)
    before_merge = len(merged_df)
    merged_df = merged_df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    dedup_drop = before_merge - len(merged_df)
    merged_df.to_csv(train_output, index=False, encoding="utf-8-sig")

    supplement_hashes = set(supplement_df["text"].astype(str).map(text_sha1).tolist())
    overlap_with_base = len(supplement_hashes & exclude["base"])
    overlap_with_fair = len(supplement_hashes & exclude["fair"])

    domain_rows = [
        {
            "source": "formal_collected",
            "target": args.target_formal,
            "selected": int((supplement_df["source"] == "formal_collected").sum()),
            "pool_usable": len(formal_pool),
            "diversity": diversity_metrics(formal_selected),
            "origin_source_top": Counter(record.origin_source for record in formal_selected).most_common(
                10
            ),
            "origin_file_top": Counter(record.origin_file for record in formal_selected).most_common(10),
            "group_key_top": Counter(record.group_key for record in formal_selected).most_common(10),
        },
        {
            "source": "real_ai_llama-3.1-405b-instruct",
            "target": args.target_llama,
            "selected": int(
                (supplement_df["source"] == "real_ai_llama-3.1-405b-instruct").sum()
            ),
            "pool_usable": len(llama_pool),
            "diversity": diversity_metrics(llama_selected),
            "origin_source_top": Counter(record.origin_source for record in llama_selected).most_common(
                10
            ),
            "origin_file_top": Counter(record.origin_file for record in llama_selected).most_common(10),
            "group_key_top": Counter(record.group_key for record in llama_selected).most_common(10),
        },
    ]

    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "base_train_path": str(base_train_path),
        "supplement_output": str(supplement_output),
        "train_output": str(train_output),
        "targets": {
            "formal_collected": args.target_formal,
            "real_ai_llama-3.1-405b-instruct": args.target_llama,
        },
        "base_rows": len(base_df),
        "supplement_rows": len(supplement_df),
        "train_rows_after_merge": len(merged_df),
        "train_dedup_drop_after_merge": dedup_drop,
        "domains": domain_rows,
        "leakage": {
            "overlap_with_base_train": overlap_with_base,
            "overlap_with_fair_test": overlap_with_fair,
        },
        "label_distribution_after_merge": merged_df["label"].value_counts().to_dict(),
        "source_distribution_after_merge_top20": merged_df["source"].value_counts()
        .head(20)
        .to_dict(),
    }

    supplement_summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    train_summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_md.write_text(build_markdown(summary), encoding="utf-8")

    print(f"[OK] supplement: {supplement_output}")
    print(f"[OK] train_v11b: {train_output}")
    print(f"[OK] summary_json: {supplement_summary_json}")
    print(f"[OK] summary_md: {summary_md}")
    print(
        "[INFO] domain selected: "
        f"formal_collected={domain_rows[0]['selected']}, "
        f"real_ai_llama-3.1-405b-instruct={domain_rows[1]['selected']}"
    )
    print(
        "[INFO] leakage check: "
        f"base_overlap={overlap_with_base}, fair_overlap={overlap_with_fair}"
    )


if __name__ == "__main__":
    main()
