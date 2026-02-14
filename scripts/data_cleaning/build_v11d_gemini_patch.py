#!/usr/bin/env python3
"""Build train_v11d_candidate.csv with a targeted Gemini-search style patch.

This script is intended for a narrow V11d patch:
1) Add short Gemini-style AI samples (64-256 chars), prioritizing search-like outputs.
2) Add matched human samples in the same length range to stabilize the boundary.
3) Enforce de-dup and no overlap with base train / fair_test.
"""

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

DEFAULT_AI_INPUTS = (
    "datasets/my_generated_ai/all_generated.jsonl",
    "datasets/generated/scenario_fill/2026-01-27_10h_multi_proxies",
)
DEFAULT_HUMAN_INPUTS = (
    "datasets/human_formal_samples.jsonl",
    "datasets/human_formal_manual.jsonl",
)
PRESET_V11D2_HUMAN_INPUTS = (
    "datasets/human_formal_samples.jsonl",
    "datasets/defense_patch/extracted/all_extracted.jsonl",
    "datasets/human_consolidated/all_human_consolidated.jsonl",
    "datasets/human_supplement/diverse_human_samples.jsonl",
)
PRESET_V11D2_HUMAN_SOURCE_QUOTA = (
    "LCSTS-news=70",
    "external_m4_qazh=70",
    "\u6a21\u677f-\u901a\u77e5=20",
)

ASCII_STYLE_CUES = (
    "pdf",
    "api",
    "sla",
    "p99",
    "kpi",
    "bug",
    "http",
    "json",
    "token",
    "sql",
    "query",
    "report",
)

LENGTH_BUCKETS = ("64-128", "128-256")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build V11d patch set focused on Gemini search-style regression."
    )
    parser.add_argument(
        "--preset",
        choices=["none", "v11d2"],
        default="none",
        help="Optional preset profile.",
    )
    parser.add_argument(
        "--base-train",
        default=str(ROOT / "datasets/merged_v2/train_v11c_candidate.csv"),
        help="Base train CSV path (typically V11c).",
    )
    parser.add_argument(
        "--fair-test-dir",
        default=str(ROOT / "datasets/eval/fair_test"),
        help="Directory containing fair_test CSV files for leakage exclusion.",
    )
    parser.add_argument(
        "--ai-inputs",
        nargs="+",
        default=[str(ROOT / item) for item in DEFAULT_AI_INPUTS],
        help="AI JSONL file(s) or directory(ies) to scan.",
    )
    parser.add_argument(
        "--human-inputs",
        nargs="+",
        default=[str(ROOT / item) for item in DEFAULT_HUMAN_INPUTS],
        help="Human JSONL file(s) or directory(ies) to scan.",
    )
    parser.add_argument(
        "--include-rejected-jsonl",
        action="store_true",
        help="Include *rejected*.jsonl when scanning directories.",
    )
    parser.add_argument(
        "--include-legacy-jsonl",
        action="store_true",
        help="Include JSONL under legacy folders such as _old.",
    )
    parser.add_argument(
        "--target-ai",
        type=int,
        default=120,
        help="Target AI supplement size.",
    )
    parser.add_argument(
        "--target-human",
        type=int,
        default=120,
        help="Target human supplement size.",
    )
    parser.add_argument(
        "--min-search-share",
        type=float,
        default=0.2,
        help="Minimum share of search-model rows in AI supplement.",
    )
    parser.add_argument(
        "--target-search-ai",
        type=int,
        default=None,
        help="Optional exact count of search-model AI rows.",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=64,
        help="Minimum text length (inclusive).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum text length (exclusive).",
    )
    parser.add_argument(
        "--min-style-ai",
        type=int,
        default=1,
        help="Minimum style score for AI candidates.",
    )
    parser.add_argument(
        "--min-style-human",
        type=int,
        default=1,
        help="Minimum style score for human candidates.",
    )
    parser.add_argument(
        "--max-group-share-ai",
        type=float,
        default=0.55,
        help="Soft cap per group during AI sampling.",
    )
    parser.add_argument(
        "--max-group-share-human",
        type=float,
        default=0.85,
        help="Soft cap per group during human sampling.",
    )
    parser.add_argument(
        "--human-source-quota",
        action="append",
        default=[],
        help="Optional source quota in form source=count (can repeat).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Sampling seed.",
    )
    parser.add_argument(
        "--supplement-output",
        default=str(ROOT / "datasets/merged_v2/v11d_gemini_style_patch.csv"),
        help="Supplement CSV output path.",
    )
    parser.add_argument(
        "--train-output",
        default=str(ROOT / "datasets/merged_v2/train_v11d_candidate.csv"),
        help="Merged V11d train CSV path.",
    )
    parser.add_argument(
        "--summary-json",
        default=str(ROOT / "datasets/merged_v2/train_v11d_candidate_summary.json"),
        help="Summary JSON output path.",
    )
    parser.add_argument(
        "--summary-md",
        default=str(ROOT / "docs/plans/v11d_gemini_patch_build.md"),
        help="Summary markdown output path.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    """Create parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)


def text_sha1(text: str) -> str:
    """Return SHA1 hash for text."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def normalize_prefix(text: str, length: int = 80) -> str:
    """Normalize prefix for diversity checks."""
    compact = re.sub(r"\W+", "", text.strip().lower())
    return compact[:length]


def style_score(text: str) -> int:
    """Score short policy/search-like style cues."""
    score = 0
    lower = text.lower()
    if any(ch.isdigit() for ch in text):
        score += 1
    if "%" in text or "％" in text:
        score += 1
    if any(cue in lower for cue in ASCII_STYLE_CUES):
        score += 1
    if 96 <= len(text) <= 220:
        score += 1
    return score


def has_hard_pattern(text: str) -> bool:
    """Check hard-risk pattern hit."""
    return any(pattern.search(text) for pattern in PATTERNS.hard_remove.values())


def length_bucket(text_len: int) -> str:
    """Convert length to bucket label."""
    if 64 <= text_len < 128:
        return "64-128"
    return "128-256"


def expand_jsonl_inputs(
    inputs: Sequence[Path],
    include_rejected_jsonl: bool,
    include_legacy_jsonl: bool,
) -> List[Path]:
    """Expand input files/directories into sorted JSONL file list."""
    files: List[Path] = []
    for entry in inputs:
        if entry.is_file():
            if entry.suffix.lower() == ".jsonl":
                if include_rejected_jsonl or "rejected" not in entry.name.lower():
                    files.append(entry)
            continue
        if entry.is_dir():
            for path in entry.rglob("*.jsonl"):
                if not include_legacy_jsonl and any(part.lower() == "_old" for part in path.parts):
                    continue
                if include_rejected_jsonl or "rejected" not in path.name.lower():
                    files.append(path)
    unique = sorted(set(files))
    return unique


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


def collect_exclude_hashes(base_train: Path, fair_test_dir: Path) -> Dict[str, set[str]]:
    """Collect exclusion hashes from base train and fair_test."""
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


def collect_ai_candidates(
    jsonl_files: Sequence[Path],
    exclude_hashes: set[str],
    min_length: int,
    max_length: int,
    min_style_score: int,
) -> pd.DataFrame:
    """Collect eligible AI candidates from Gemini-family outputs."""
    rows: List[Dict[str, object]] = []
    seen_hashes: set[str] = set()

    for jsonl_path in jsonl_files:
        for item in iter_jsonl(jsonl_path):
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            text_len = len(text)
            if not (min_length <= text_len < max_length):
                continue
            if has_hard_pattern(text):
                continue

            label_raw = item.get("label", 1)
            try:
                label = int(label_raw)
            except (TypeError, ValueError):
                label = 1
            if label != 1:
                continue

            model = str(item.get("model", "")).strip().lower()
            if "gemini" not in model:
                continue

            score = style_score(text)
            if score < min_style_score:
                continue

            h = text_sha1(text)
            if h in exclude_hashes or h in seen_hashes:
                continue
            seen_hashes.add(h)

            scenario_id = str(item.get("scenario_id", "")).strip() or "unknown"
            origin_source = str(item.get("source", "")).strip() or "unknown"
            is_search_model = "search" in model
            rows.append(
                {
                    "text": text,
                    "label": 1,
                    "source": (
                        "v11d_ai_gemini_search"
                        if is_search_model
                        else "v11d_ai_gemini_related"
                    ),
                    "origin_pool": str(jsonl_path),
                    "origin_source": origin_source,
                    "model": model,
                    "scenario_id": scenario_id,
                    "char_len": text_len,
                    "length_bucket": length_bucket(text_len),
                    "style_score": score,
                    "is_search_model": is_search_model,
                    "text_sha1": h,
                }
            )
    return pd.DataFrame(rows)


def collect_human_candidates(
    jsonl_files: Sequence[Path],
    exclude_hashes: set[str],
    min_length: int,
    max_length: int,
    min_style_score: int,
) -> pd.DataFrame:
    """Collect eligible human candidates for boundary stabilization."""
    rows: List[Dict[str, object]] = []
    seen_hashes: set[str] = set()

    for jsonl_path in jsonl_files:
        for item in iter_jsonl(jsonl_path):
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            text_len = len(text)
            if not (min_length <= text_len < max_length):
                continue
            if has_hard_pattern(text):
                continue

            label_raw = item.get("label", 0)
            try:
                label = int(label_raw)
            except (TypeError, ValueError):
                label = 0
            if label != 0:
                continue

            score = style_score(text)
            if score < min_style_score:
                continue

            h = text_sha1(text)
            if h in exclude_hashes or h in seen_hashes:
                continue
            seen_hashes.add(h)

            origin_source = str(item.get("source", "")).strip() or "unknown"
            origin_type = str(item.get("type", "")).strip() or "unknown"
            rows.append(
                {
                    "text": text,
                    "label": 0,
                    "source": "v11d_human_match",
                    "origin_pool": str(jsonl_path),
                    "origin_source": origin_source,
                    "origin_type": origin_type,
                    "char_len": text_len,
                    "length_bucket": length_bucket(text_len),
                    "style_score": score,
                    "text_sha1": h,
                }
            )
    return pd.DataFrame(rows)


def sample_diverse(
    df: pd.DataFrame,
    target: int,
    group_cols: Sequence[str],
    score_col: str,
    seed: int,
    max_group_share: float,
) -> pd.DataFrame:
    """Sample rows with coarse diversity constraints."""
    if target <= 0:
        return df.iloc[0:0].copy()
    if len(df) < target:
        raise ValueError(f"Not enough rows for sampling: need {target}, got {len(df)}")

    work = df.copy()
    rng = random.Random(seed)
    work["_rand"] = [rng.random() for _ in range(len(work))]
    work["_group_key"] = (
        work[list(group_cols)]
        .fillna("unknown")
        .astype(str)
        .agg("::".join, axis=1)
    )
    work = work.sort_values([score_col, "_rand"], ascending=[False, True]).reset_index(drop=True)

    grouped: Dict[str, List[int]] = defaultdict(list)
    for idx, row in work.iterrows():
        grouped[str(row["_group_key"])].append(idx)

    cap = max(1, math.ceil(target * max_group_share))
    selected_idx: List[int] = []
    selected_set: set[int] = set()
    group_counts: Counter[str] = Counter()

    progress = True
    while len(selected_idx) < target and progress:
        progress = False
        for group_key in sorted(grouped.keys()):
            if len(selected_idx) >= target:
                break
            if group_counts[group_key] >= cap:
                continue
            rows = grouped[group_key]
            while rows and rows[0] in selected_set:
                rows.pop(0)
            if not rows:
                continue
            take_idx = rows.pop(0)
            selected_idx.append(take_idx)
            selected_set.add(take_idx)
            group_counts[group_key] += 1
            progress = True

    if len(selected_idx) < target:
        for idx in work.index:
            if len(selected_idx) >= target:
                break
            if idx in selected_set:
                continue
            selected_idx.append(idx)
            selected_set.add(idx)

    selected = work.loc[selected_idx].copy()
    selected = selected.drop(columns=["_rand", "_group_key"]).reset_index(drop=True)
    return selected


def sample_ai_patch(
    ai_pool: pd.DataFrame,
    target_ai: int,
    min_search_share: float,
    target_search_rows: int | None,
    seed: int,
    max_group_share_ai: float,
) -> pd.DataFrame:
    """Sample AI patch rows, prioritizing Gemini search model outputs."""
    if ai_pool.empty:
        raise ValueError("AI candidate pool is empty.")

    search_pool = ai_pool[ai_pool["is_search_model"]].copy()
    non_search_pool = ai_pool[~ai_pool["is_search_model"]].copy()

    min_search_rows = int(round(target_ai * min_search_share))
    min_search_rows = min(min_search_rows, target_ai)
    required_search_rows = min_search_rows
    if target_search_rows is not None:
        if target_search_rows < 0 or target_search_rows > target_ai:
            raise ValueError(
                f"target_search_rows out of range: {target_search_rows} (target_ai={target_ai})"
            )
        required_search_rows = target_search_rows

    if len(search_pool) < required_search_rows:
        raise ValueError(
            "Search-model candidates not enough: "
            f"need {required_search_rows}, got {len(search_pool)}"
        )

    take_search = required_search_rows
    search_selected = sample_diverse(
        search_pool,
        target=take_search,
        group_cols=["scenario_id", "length_bucket"],
        score_col="style_score",
        seed=seed + 101,
        max_group_share=max_group_share_ai,
    )

    remain_target = target_ai - len(search_selected)
    if remain_target <= 0:
        return search_selected.iloc[:target_ai].copy().reset_index(drop=True)

    if target_search_rows is None:
        remain_pool = pd.concat([non_search_pool, search_pool], ignore_index=True)
    else:
        remain_pool = non_search_pool.copy()
    remain_pool = remain_pool[
        ~remain_pool["text_sha1"].isin(search_selected["text_sha1"])
    ].copy()
    remain_selected = sample_diverse(
        remain_pool,
        target=remain_target,
        group_cols=["model", "scenario_id", "length_bucket"],
        score_col="style_score",
        seed=seed + 202,
        max_group_share=max_group_share_ai,
    )
    selected = pd.concat([search_selected, remain_selected], ignore_index=True)
    selected = selected.drop_duplicates(subset=["text_sha1"], keep="first").reset_index(drop=True)

    if len(selected) < target_ai:
        raise ValueError(f"AI patch rows after dedup < target: {len(selected)} < {target_ai}")
    return selected.iloc[:target_ai].copy().reset_index(drop=True)


def allocate_human_bucket_quota(
    ai_selected: pd.DataFrame,
    target_human: int,
) -> Dict[str, int]:
    """Allocate human quota by AI length bucket ratio."""
    bucket_counts = ai_selected["length_bucket"].value_counts().to_dict()
    quotas = {bucket: 0 for bucket in LENGTH_BUCKETS}
    if target_human <= 0:
        return quotas

    running = 0
    for bucket in LENGTH_BUCKETS:
        raw = target_human * bucket_counts.get(bucket, 0) / max(len(ai_selected), 1)
        quotas[bucket] = int(round(raw))
        running += quotas[bucket]

    while running < target_human:
        best_bucket = max(LENGTH_BUCKETS, key=lambda b: bucket_counts.get(b, 0))
        quotas[best_bucket] += 1
        running += 1
    while running > target_human:
        for bucket in sorted(LENGTH_BUCKETS, key=lambda b: quotas[b], reverse=True):
            if quotas[bucket] > 0 and running > target_human:
                quotas[bucket] -= 1
                running -= 1
    return quotas


def sample_human_patch(
    human_pool: pd.DataFrame,
    ai_selected: pd.DataFrame,
    target_human: int,
    seed: int,
    max_group_share_human: float,
) -> pd.DataFrame:
    """Sample human patch rows with bucket-aware balancing."""
    if target_human <= 0:
        return human_pool.iloc[0:0].copy()
    if human_pool.empty:
        raise ValueError("Human candidate pool is empty.")

    quotas = allocate_human_bucket_quota(ai_selected, target_human)
    selected_parts: List[pd.DataFrame] = []
    deficits = 0

    for offset, bucket in enumerate(LENGTH_BUCKETS):
        bucket_pool = human_pool[human_pool["length_bucket"] == bucket].copy()
        need = quotas[bucket]
        if need <= 0:
            continue
        take = min(len(bucket_pool), need)
        if take > 0:
            part = sample_diverse(
                bucket_pool,
                target=take,
                group_cols=["origin_source", "origin_type", "length_bucket"],
                score_col="style_score",
                seed=seed + 300 + offset * 17,
                max_group_share=max_group_share_human,
            )
            selected_parts.append(part)
        deficits += need - take

    selected = (
        pd.concat(selected_parts, ignore_index=True)
        if selected_parts
        else human_pool.iloc[0:0].copy()
    )
    selected = selected.drop_duplicates(subset=["text_sha1"], keep="first").reset_index(drop=True)

    if deficits > 0:
        remain_pool = human_pool[~human_pool["text_sha1"].isin(selected["text_sha1"])].copy()
        top_up = sample_diverse(
            remain_pool,
            target=deficits,
            group_cols=["origin_source", "origin_type", "length_bucket"],
            score_col="style_score",
            seed=seed + 399,
            max_group_share=max_group_share_human,
        )
        selected = pd.concat([selected, top_up], ignore_index=True)
        selected = selected.drop_duplicates(subset=["text_sha1"], keep="first").reset_index(drop=True)

    if len(selected) < target_human:
        raise ValueError(f"Human patch rows after dedup < target: {len(selected)} < {target_human}")
    return selected.iloc[:target_human].copy().reset_index(drop=True)


def sample_human_patch_with_source_quota(
    human_pool: pd.DataFrame,
    ai_selected: pd.DataFrame,
    target_human: int,
    source_quota: Dict[str, int],
    seed: int,
    max_group_share_human: float,
) -> pd.DataFrame:
    """Sample human patch rows with explicit per-source quotas first."""
    if target_human <= 0:
        return human_pool.iloc[0:0].copy()
    if not source_quota:
        return sample_human_patch(
            human_pool=human_pool,
            ai_selected=ai_selected,
            target_human=target_human,
            seed=seed,
            max_group_share_human=max_group_share_human,
        )

    quota_total = sum(source_quota.values())
    if quota_total > target_human:
        raise ValueError(
            f"Sum of source quotas {quota_total} exceeds target_human {target_human}"
        )

    selected_parts: List[pd.DataFrame] = []
    used_hashes: set[str] = set()
    for idx, source in enumerate(sorted(source_quota.keys())):
        need = int(source_quota[source])
        if need <= 0:
            continue
        source_pool = human_pool[
            (human_pool["origin_source"] == source)
            & (~human_pool["text_sha1"].isin(used_hashes))
        ].copy()
        if len(source_pool) < need:
            raise ValueError(
                f"Human source quota cannot be met for '{source}': need {need}, got {len(source_pool)}"
            )
        part = sample_diverse(
            source_pool,
            target=need,
            group_cols=["origin_type", "length_bucket"],
            score_col="style_score",
            seed=seed + 700 + idx * 17,
            max_group_share=max_group_share_human,
        )
        selected_parts.append(part)
        used_hashes.update(part["text_sha1"].tolist())

    selected = (
        pd.concat(selected_parts, ignore_index=True)
        if selected_parts
        else human_pool.iloc[0:0].copy()
    )
    selected = selected.drop_duplicates(subset=["text_sha1"], keep="first").reset_index(drop=True)

    remain_target = target_human - len(selected)
    if remain_target > 0:
        remain_pool = human_pool[~human_pool["text_sha1"].isin(selected["text_sha1"])].copy()
        remain_selected = sample_human_patch(
            human_pool=remain_pool,
            ai_selected=ai_selected,
            target_human=remain_target,
            seed=seed + 899,
            max_group_share_human=max_group_share_human,
        )
        selected = pd.concat([selected, remain_selected], ignore_index=True)
        selected = selected.drop_duplicates(subset=["text_sha1"], keep="first").reset_index(drop=True)

    if len(selected) < target_human:
        raise ValueError(
            f"Human patch rows after source quota + top-up < target: {len(selected)} < {target_human}"
        )
    return selected.iloc[:target_human].copy().reset_index(drop=True)


def diversity_metrics(texts: Sequence[str]) -> Dict[str, float]:
    """Compute simple diversity metrics."""
    if not texts:
        return {
            "rows": 0,
            "unique_prefix_ratio": 0.0,
            "top_prefix_share": 0.0,
        }
    prefixes = [normalize_prefix(text) for text in texts]
    prefix_counter = Counter(prefixes)
    rows = len(prefixes)
    top_share = prefix_counter.most_common(1)[0][1] / rows
    return {
        "rows": rows,
        "unique_prefix_ratio": round(len(prefix_counter) / rows, 4),
        "top_prefix_share": round(top_share, 4),
    }


def safe_int_dict(raw: Dict[object, object]) -> Dict[str, int]:
    """Convert dict values to int with string keys."""
    out: Dict[str, int] = {}
    for key, value in raw.items():
        out[str(key)] = int(value)
    return out


def parse_source_quota(quota_items: Sequence[str]) -> Dict[str, int]:
    """Parse source quotas from source=count strings."""
    quotas: Dict[str, int] = {}
    for raw in quota_items:
        item = str(raw).strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid quota format (expected source=count): {item}")
        source, value = item.split("=", 1)
        source = source.strip()
        if not source:
            raise ValueError(f"Empty source in quota: {item}")
        try:
            count = int(value.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid quota count in: {item}") from exc
        if count < 0:
            raise ValueError(f"Quota must be non-negative: {item}")
        quotas[source] = quotas.get(source, 0) + count
    return quotas


def apply_preset(args: argparse.Namespace) -> None:
    """Apply preset values in-place."""
    if args.preset != "v11d2":
        return

    args.target_ai = 80
    args.target_human = 160
    args.min_search_share = 0.3
    args.target_search_ai = 24
    args.min_style_human = 0
    args.ai_inputs = [str(ROOT / item) for item in DEFAULT_AI_INPUTS]
    args.human_inputs = [str(ROOT / item) for item in PRESET_V11D2_HUMAN_INPUTS]
    args.human_source_quota = list(PRESET_V11D2_HUMAN_SOURCE_QUOTA)
    args.supplement_output = str(ROOT / "datasets/merged_v2/v11d2_gemini_patch.csv")
    args.train_output = str(ROOT / "datasets/merged_v2/train_v11d2_candidate.csv")
    args.summary_json = str(ROOT / "datasets/merged_v2/train_v11d2_candidate_summary.json")
    args.summary_md = str(ROOT / "docs/plans/v11d2_gemini_patch_build.md")


def build_markdown(summary: Dict[str, object]) -> str:
    """Render summary markdown."""
    lines = [
        "# V11d Gemini Patch Build",
        "",
        f"- Generated at: {summary['generated_at']}",
        f"- Preset: {summary['preset']}",
        f"- Base train: `{summary['base_train_path']}`",
        f"- Train rows: {summary['base_rows']} -> {summary['train_rows_after_merge']}",
        f"- Supplement rows: {summary['supplement_rows']}",
        f"  - AI rows: {summary['ai_selected_rows']}",
        f"  - Human rows: {summary['human_selected_rows']}",
        "",
        "## AI supplement",
        "",
        f"- target search rows: {summary['target_search_ai']}",
        f"- search rows selected: {summary['ai_search_rows_selected']}",
        f"- length buckets: {summary['ai_length_bucket_counts']}",
        f"- style score dist: {summary['ai_style_score_counts']}",
        "",
        "## Human supplement",
        "",
        f"- source quota requested: {summary['human_source_quota_requested']}",
        f"- length buckets: {summary['human_length_bucket_counts']}",
        f"- style score dist: {summary['human_style_score_counts']}",
        "",
        "## Leakage checks",
        "",
        f"- overlap with base train: {summary['leakage']['overlap_with_base_train']}",
        f"- overlap with fair_test: {summary['leakage']['overlap_with_fair_test']}",
        "",
        "## Diversity",
        "",
        f"- AI: {summary['ai_diversity']}",
        f"- Human: {summary['human_diversity']}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """Entry point."""
    args = parse_args()
    apply_preset(args)
    human_source_quota = parse_source_quota(args.human_source_quota)

    base_train_path = Path(args.base_train)
    fair_test_dir = Path(args.fair_test_dir)
    ai_inputs = [Path(item) for item in args.ai_inputs]
    human_inputs = [Path(item) for item in args.human_inputs]

    supplement_output = Path(args.supplement_output)
    train_output = Path(args.train_output)
    summary_json_path = Path(args.summary_json)
    summary_md_path = Path(args.summary_md)
    for path in (supplement_output, train_output, summary_json_path, summary_md_path):
        ensure_parent(path)

    base_df = pd.read_csv(base_train_path, encoding="utf-8-sig")
    for col in ("text", "label"):
        if col not in base_df.columns:
            raise ValueError(f"Base train missing required column: {col}")
    if "source" not in base_df.columns:
        base_df["source"] = "unknown"

    exclude = collect_exclude_hashes(base_train_path, fair_test_dir)
    exclude_hashes = exclude["base"] | exclude["fair"]

    ai_files = expand_jsonl_inputs(
        ai_inputs,
        include_rejected_jsonl=args.include_rejected_jsonl,
        include_legacy_jsonl=args.include_legacy_jsonl,
    )
    human_files = expand_jsonl_inputs(
        human_inputs,
        include_rejected_jsonl=args.include_rejected_jsonl,
        include_legacy_jsonl=args.include_legacy_jsonl,
    )
    if not ai_files:
        raise ValueError("No valid AI JSONL files found.")
    if not human_files:
        raise ValueError("No valid human JSONL files found.")

    ai_pool = collect_ai_candidates(
        jsonl_files=ai_files,
        exclude_hashes=exclude_hashes,
        min_length=args.min_length,
        max_length=args.max_length,
        min_style_score=args.min_style_ai,
    )
    human_pool = collect_human_candidates(
        jsonl_files=human_files,
        exclude_hashes=exclude_hashes,
        min_length=args.min_length,
        max_length=args.max_length,
        min_style_score=args.min_style_human,
    )

    ai_selected = sample_ai_patch(
        ai_pool=ai_pool,
        target_ai=args.target_ai,
        min_search_share=args.min_search_share,
        target_search_rows=args.target_search_ai,
        seed=args.seed,
        max_group_share_ai=args.max_group_share_ai,
    )
    human_selected = sample_human_patch_with_source_quota(
        human_pool=human_pool,
        ai_selected=ai_selected,
        target_human=args.target_human,
        source_quota=human_source_quota,
        seed=args.seed,
        max_group_share_human=args.max_group_share_human,
    )

    supplement_df = pd.concat([ai_selected, human_selected], ignore_index=True)
    supplement_df = supplement_df.drop_duplicates(subset=["text_sha1"], keep="first").reset_index(
        drop=True
    )
    supplement_df.to_csv(supplement_output, index=False, encoding="utf-8-sig")

    merge_cols = ["text", "label", "source"]
    merged_df = pd.concat(
        [base_df[merge_cols], supplement_df[merge_cols]],
        ignore_index=True,
    )
    before_dedup = len(merged_df)
    merged_df = merged_df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    dedup_drop = before_dedup - len(merged_df)
    merged_df.to_csv(train_output, index=False, encoding="utf-8-sig")

    supplement_hashes = set(supplement_df["text"].astype(str).map(text_sha1).tolist())
    overlap_with_base = len(supplement_hashes & exclude["base"])
    overlap_with_fair = len(supplement_hashes & exclude["fair"])

    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "base_train_path": str(base_train_path),
        "fair_test_dir": str(fair_test_dir),
        "ai_inputs": [str(item) for item in ai_inputs],
        "human_inputs": [str(item) for item in human_inputs],
        "ai_jsonl_files_count": int(len(ai_files)),
        "human_jsonl_files_count": int(len(human_files)),
        "ai_jsonl_files_sample": [str(path) for path in ai_files[:20]],
        "human_jsonl_files_sample": [str(path) for path in human_files[:20]],
        "preset": args.preset,
        "target_ai": int(args.target_ai),
        "target_human": int(args.target_human),
        "target_search_ai": (
            int(args.target_search_ai) if args.target_search_ai is not None else None
        ),
        "base_rows": int(len(base_df)),
        "supplement_rows": int(len(supplement_df)),
        "ai_pool_rows": int(len(ai_pool)),
        "human_pool_rows": int(len(human_pool)),
        "ai_selected_rows": int(len(ai_selected)),
        "human_selected_rows": int(len(human_selected)),
        "ai_search_rows_selected": int(ai_selected["is_search_model"].sum()),
        "train_rows_after_merge": int(len(merged_df)),
        "train_dedup_drop_after_merge": int(dedup_drop),
        "ai_source_counts": safe_int_dict(ai_selected["source"].value_counts().to_dict()),
        "ai_model_top": safe_int_dict(ai_selected["model"].value_counts().head(20).to_dict()),
        "ai_scenario_top": safe_int_dict(
            ai_selected["scenario_id"].value_counts().head(20).to_dict()
        ),
        "ai_length_bucket_counts": safe_int_dict(
            ai_selected["length_bucket"].value_counts().to_dict()
        ),
        "ai_style_score_counts": safe_int_dict(ai_selected["style_score"].value_counts().to_dict()),
        "human_source_counts": safe_int_dict(
            human_selected["origin_source"].value_counts().head(20).to_dict()
        ),
        "human_type_counts": safe_int_dict(
            human_selected["origin_type"].value_counts().head(20).to_dict()
        ),
        "human_length_bucket_counts": safe_int_dict(
            human_selected["length_bucket"].value_counts().to_dict()
        ),
        "human_style_score_counts": safe_int_dict(
            human_selected["style_score"].value_counts().to_dict()
        ),
        "human_source_quota_requested": safe_int_dict(human_source_quota),
        "ai_diversity": diversity_metrics(ai_selected["text"].astype(str).tolist()),
        "human_diversity": diversity_metrics(human_selected["text"].astype(str).tolist()),
        "leakage": {
            "overlap_with_base_train": int(overlap_with_base),
            "overlap_with_fair_test": int(overlap_with_fair),
        },
        "artifacts": {
            "supplement_output": str(supplement_output),
            "train_output": str(train_output),
            "summary_json": str(summary_json_path),
            "summary_md": str(summary_md_path),
        },
    }

    summary_json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_md_path.write_text(build_markdown(summary), encoding="utf-8")

    print(f"[OK] supplement: {supplement_output}")
    print(f"[OK] train_v11d: {train_output}")
    print(f"[OK] summary_json: {summary_json_path}")
    print(f"[OK] summary_md: {summary_md_path}")
    print(
        "[INFO] patch rows / leakage: "
        f"{len(supplement_df)} / base_overlap={overlap_with_base}, fair_overlap={overlap_with_fair}"
    )


if __name__ == "__main__":
    main()
