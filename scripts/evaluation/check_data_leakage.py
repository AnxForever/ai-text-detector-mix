#!/usr/bin/env python3
"""
数据泄露与重复检测脚本
Data Leakage & Duplication Checker

检查训练集、验证集、测试集之间的数据泄露问题：
1. 精确重复 (SHA1 hash)
2. 近似重复 (SimHash, hamming distance <= 3)
3. 集内重复 (internal duplicates)

作者：AI Detection Team
日期：2026-02-10
"""

import hashlib
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

# ──────────────────────────────────────────────────────────────
# 项目根目录
# ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ──────────────────────────────────────────────────────────────
# 数据集路径配置
# ──────────────────────────────────────────────────────────────
TRAIN_SETS = {
    "core_v1/train": "datasets/active/core_v1/train.csv",
    "core_v2/train": "datasets/active/core_v2/train.csv",
    "core_v3/train": "datasets/active/core_v3/train.csv",
    "merged_v2/train": "datasets/merged_v2/train.csv",
}

VAL_SETS = {
    "core_v1/val": "datasets/active/core_v1/val.csv",
    "core_v2/val": "datasets/active/core_v2/val.csv",
    "core_v3/val": "datasets/active/core_v3/val.csv",
    "merged_v2/val": "datasets/merged_v2/val.csv",
}

TEST_SETS = {
    "core_v1/test": "datasets/active/core_v1/test.csv",
    "core_v2/test": "datasets/active/core_v2/test.csv",
}

FAIR_TEST_SETS = {
    "fair/core_v1_clean": "datasets/eval/fair_test/core_v1_test_clean.csv",
    "fair/independent": "datasets/eval/fair_test/independent_data.csv",
    "fair/merged_v2_val_clean": "datasets/eval/fair_test/merged_v2_val_clean.csv",
}


# ──────────────────────────────────────────────────────────────
# 文本规范化
# ──────────────────────────────────────────────────────────────
_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Strip and collapse all whitespace to single space."""
    if not isinstance(text, str):
        return ""
    return _WS_RE.sub(" ", text.strip())


# ──────────────────────────────────────────────────────────────
# SHA1 精确哈希
# ──────────────────────────────────────────────────────────────
def sha1_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# ──────────────────────────────────────────────────────────────
# SimHash 近似哈希 (64-bit, 纯 Python 实现)
# ──────────────────────────────────────────────────────────────
def _tokenize(text: str) -> List[str]:
    """Simple character n-gram tokenizer (3-grams) for Chinese text.

    Character n-grams work well for Chinese because words are typically
    1-4 characters, so 3-grams capture meaningful sub-word units without
    needing a segmentation library.
    """
    tokens = []
    n = 3
    for i in range(max(1, len(text) - n + 1)):
        tokens.append(text[i : i + n])
    return tokens


def _feature_hash_64(token: str) -> int:
    """Hash a single token to a 64-bit integer using MD5 truncation."""
    digest = hashlib.md5(token.encode("utf-8")).digest()
    # Take first 8 bytes as a 64-bit int
    return int.from_bytes(digest[:8], byteorder="little")


def simhash_64(text: str) -> int:
    """Compute a 64-bit SimHash for the given text.

    Algorithm:
    1. Tokenize text into character n-grams.
    2. For each token, compute a 64-bit hash.
    3. Maintain a 64-element vector of weights.
       For each bit position: if hash bit is 1, add 1; else subtract 1.
    4. Final hash: bit i = 1 if vector[i] > 0 else 0.
    """
    tokens = _tokenize(text)
    if not tokens:
        return 0

    v = [0] * 64
    for token in tokens:
        h = _feature_hash_64(token)
        for i in range(64):
            if h & (1 << i):
                v[i] += 1
            else:
                v[i] -= 1

    fingerprint = 0
    for i in range(64):
        if v[i] > 0:
            fingerprint |= 1 << i
    return fingerprint


def hamming_distance(a: int, b: int) -> int:
    """Count differing bits between two 64-bit integers."""
    return bin(a ^ b).count("1")


# ──────────────────────────────────────────────────────────────
# 数据集加载
# ──────────────────────────────────────────────────────────────
def load_dataset(
    name: str, rel_path: str
) -> Optional[pd.DataFrame]:
    """Load a single CSV dataset. Returns None if file missing."""
    full_path = PROJECT_ROOT / rel_path
    if not full_path.exists():
        print(f"  [SKIP] {name}: file not found ({rel_path})")
        return None
    try:
        df = pd.read_csv(
            full_path,
            encoding="utf-8-sig",
            usecols=["text", "label"],
            dtype={"text": str, "label": int},
        )
        # Normalize text in-place
        df["text"] = df["text"].apply(normalize_text)
        # Drop rows with empty text after normalization
        df = df[df["text"].str.len() > 0].reset_index(drop=True)
        print(f"  [OK]   {name}: {len(df):,} rows")
        return df
    except Exception as e:
        print(f"  [ERR]  {name}: {e}")
        return None


def load_group(
    group: Dict[str, str],
) -> Dict[str, pd.DataFrame]:
    """Load a group of datasets, skipping missing ones."""
    result = {}
    for name, path in group.items():
        df = load_dataset(name, path)
        if df is not None:
            result[name] = df
    return result


# ──────────────────────────────────────────────────────────────
# SHA1 哈希集构建
# ──────────────────────────────────────────────────────────────
def build_hash_set(df: pd.DataFrame) -> Set[str]:
    """Build a set of SHA1 hashes from a DataFrame's text column."""
    return set(df["text"].apply(sha1_hash))


def build_hash_map(df: pd.DataFrame) -> Dict[str, List[int]]:
    """Build hash -> [row indices] mapping for duplicate detection."""
    hmap: Dict[str, List[int]] = defaultdict(list)
    for idx, text in enumerate(df["text"]):
        h = sha1_hash(text)
        hmap[h].append(idx)
    return hmap


# ──────────────────────────────────────────────────────────────
# 泄露检测
# ──────────────────────────────────────────────────────────────
def check_exact_overlap(
    name_a: str,
    df_a: pd.DataFrame,
    hashes_a: Set[str],
    name_b: str,
    df_b: pd.DataFrame,
    hashes_b: Set[str],
    max_samples: int = 3,
) -> Dict:
    """Check exact overlap between two datasets using SHA1 hashes.

    Returns a dict with overlap statistics and sample overlapping texts.
    """
    overlap_hashes = hashes_a & hashes_b
    overlap_count = len(overlap_hashes)

    # Compute overlap rate relative to the smaller dataset
    size_a = len(hashes_a)
    size_b = len(hashes_b)
    smaller = min(size_a, size_b)
    overlap_rate = overlap_count / smaller if smaller > 0 else 0.0

    # Collect sample overlapping texts from df_b (the "leaked-into" set)
    samples = []
    if overlap_count > 0 and max_samples > 0:
        # Build reverse lookup from hash -> text for df_b
        sample_count = 0
        for text in df_b["text"]:
            h = sha1_hash(text)
            if h in overlap_hashes:
                truncated = text[:120] + ("..." if len(text) > 120 else "")
                samples.append(truncated)
                sample_count += 1
                if sample_count >= max_samples:
                    break

    return {
        "pair": f"{name_a} vs {name_b}",
        "size_a": size_a,
        "size_b": size_b,
        "overlap": overlap_count,
        "rate": overlap_rate,
        "samples": samples,
    }


def check_internal_duplicates(
    name: str, df: pd.DataFrame
) -> Dict:
    """Check for internal duplicates within a single dataset."""
    hmap = build_hash_map(df)
    dup_groups = {h: idxs for h, idxs in hmap.items() if len(idxs) > 1}
    total_dup_rows = sum(len(v) - 1 for v in dup_groups.values())
    dup_rate = total_dup_rows / len(df) if len(df) > 0 else 0.0

    samples = []
    for h, idxs in list(dup_groups.items())[:3]:
        text = df.iloc[idxs[0]]["text"]
        truncated = text[:120] + ("..." if len(text) > 120 else "")
        samples.append(f"[x{len(idxs)}] {truncated}")

    return {
        "dataset": name,
        "total_rows": len(df),
        "unique_hashes": len(hmap),
        "dup_groups": len(dup_groups),
        "dup_rows": total_dup_rows,
        "dup_rate": dup_rate,
        "samples": samples,
    }


# ──────────────────────────────────────────────────────────────
# SimHash 近似重复检测
# ──────────────────────────────────────────────────────────────
def check_near_duplicates(
    name_a: str,
    df_a: pd.DataFrame,
    name_b: str,
    df_b: pd.DataFrame,
    threshold: int = 3,
    max_samples: int = 5,
    max_check: int = 50000,
) -> Dict:
    """Check near-duplicates between two datasets using SimHash.

    Uses a bucket-based approach for efficiency:
    1. Compute SimHash for all texts in both datasets.
    2. Split 64-bit hash into 4 x 16-bit bands.
    3. Candidate pairs share at least one band.
    4. Verify candidates with full hamming distance check.

    Args:
        threshold: Maximum hamming distance to consider near-duplicate.
        max_check: Maximum rows to process from each dataset (memory/time).
        max_samples: Maximum sample pairs to collect.
    """
    print(f"\n  SimHash near-duplicate: {name_a} ({len(df_a):,}) vs {name_b} ({len(df_b):,})")

    # Subsample if datasets are too large
    texts_a = df_a["text"].tolist()[:max_check]
    texts_b = df_b["text"].tolist()[:max_check]

    print(f"    Computing SimHash for {len(texts_a):,} + {len(texts_b):,} texts...")
    t0 = time.time()

    hashes_a = [simhash_64(t) for t in texts_a]
    hashes_b = [simhash_64(t) for t in texts_b]

    elapsed_hash = time.time() - t0
    print(f"    Hashing done in {elapsed_hash:.1f}s")

    # Band-based candidate generation (4 bands x 16 bits)
    NUM_BANDS = 4
    BAND_BITS = 16
    BAND_MASK = (1 << BAND_BITS) - 1

    def get_bands(h: int) -> Tuple[int, ...]:
        return tuple(
            (h >> (i * BAND_BITS)) & BAND_MASK for i in range(NUM_BANDS)
        )

    # Index dataset B into band buckets
    buckets_b: List[Dict[int, List[int]]] = [
        defaultdict(list) for _ in range(NUM_BANDS)
    ]
    for idx, h in enumerate(hashes_b):
        bands = get_bands(h)
        for band_idx, band_val in enumerate(bands):
            buckets_b[band_idx][band_val].append(idx)

    # Find candidate pairs from A
    print("    Searching candidates via band buckets...")
    t1 = time.time()

    near_dup_count = 0
    sample_pairs: List[Tuple[str, str, int]] = []
    checked_pairs: Set[Tuple[int, int]] = set()

    for idx_a, h_a in enumerate(hashes_a):
        bands_a = get_bands(h_a)
        candidate_idxs_b: Set[int] = set()
        for band_idx, band_val in enumerate(bands_a):
            candidate_idxs_b.update(buckets_b[band_idx].get(band_val, []))

        for idx_b in candidate_idxs_b:
            pair_key = (idx_a, idx_b)
            if pair_key in checked_pairs:
                continue
            checked_pairs.add(pair_key)

            dist = hamming_distance(h_a, hashes_b[idx_b])
            if dist <= threshold and dist > 0:
                # dist == 0 means exact match at SimHash level, skip those
                # (caught by SHA1 already)
                near_dup_count += 1
                if len(sample_pairs) < max_samples:
                    ta = texts_a[idx_a][:80]
                    tb = texts_b[idx_b][:80]
                    sample_pairs.append((ta, tb, dist))

    elapsed_search = time.time() - t1
    smaller = min(len(texts_a), len(texts_b))
    near_rate = near_dup_count / smaller if smaller > 0 else 0.0

    print(
        f"    Found {near_dup_count:,} near-duplicates "
        f"(hamming <= {threshold}) in {elapsed_search:.1f}s"
    )
    print(
        f"    Checked {len(checked_pairs):,} candidate pairs "
        f"(rate: {near_rate:.4%})"
    )

    return {
        "pair": f"{name_a} vs {name_b}",
        "checked_a": len(texts_a),
        "checked_b": len(texts_b),
        "candidates_checked": len(checked_pairs),
        "near_duplicates": near_dup_count,
        "near_dup_rate": near_rate,
        "threshold": threshold,
        "sample_pairs": sample_pairs,
        "time_hash": elapsed_hash,
        "time_search": elapsed_search,
    }


# ──────────────────────────────────────────────────────────────
# 报告输出
# ──────────────────────────────────────────────────────────────
SEP_LINE = "=" * 90
THIN_LINE = "-" * 90


def print_header(title: str) -> None:
    print(f"\n{SEP_LINE}")
    print(f"  {title}")
    print(SEP_LINE)


def print_overlap_table(results: List[Dict]) -> None:
    """Print a formatted table of overlap results."""
    # Table header
    print(
        f"{'Pair':<45} {'|':>1} {'Size A':>8} {'|':>1} {'Size B':>8} "
        f"{'|':>1} {'Overlap':>8} {'|':>1} {'Rate':>10}"
    )
    print(THIN_LINE)

    for r in results:
        rate_str = f"{r['rate']:.4%}" if r["rate"] > 0 else "0.0000%"
        marker = " *** LEAK" if r["overlap"] > 0 else ""
        print(
            f"{r['pair']:<45} {'|':>1} {r['size_a']:>8,} {'|':>1} "
            f"{r['size_b']:>8,} {'|':>1} {r['overlap']:>8,} {'|':>1} "
            f"{rate_str:>10}{marker}"
        )

    # Print samples for any overlapping pair
    has_overlap = [r for r in results if r["overlap"] > 0]
    if has_overlap:
        print(f"\n  Sample overlapping texts:")
        for r in has_overlap:
            if r["samples"]:
                print(f"\n  [{r['pair']}]")
                for s in r["samples"]:
                    print(f"    - {s}")


def print_internal_dup_table(results: List[Dict]) -> None:
    """Print internal duplicate results table."""
    print(
        f"{'Dataset':<30} {'|':>1} {'Total':>8} {'|':>1} {'Unique':>8} "
        f"{'|':>1} {'Dup Groups':>10} {'|':>1} {'Dup Rows':>9} "
        f"{'|':>1} {'Rate':>10}"
    )
    print(THIN_LINE)

    for r in results:
        rate_str = f"{r['dup_rate']:.4%}"
        marker = " *" if r["dup_rate"] > 0.01 else ""
        print(
            f"{r['dataset']:<30} {'|':>1} {r['total_rows']:>8,} {'|':>1} "
            f"{r['unique_hashes']:>8,} {'|':>1} {r['dup_groups']:>10,} "
            f"{'|':>1} {r['dup_rows']:>9,} {'|':>1} {rate_str:>10}{marker}"
        )

    # Print samples
    has_dups = [r for r in results if r["dup_rows"] > 0]
    if has_dups:
        print(f"\n  Sample internal duplicates:")
        for r in has_dups:
            if r["samples"]:
                print(f"\n  [{r['dataset']}]")
                for s in r["samples"]:
                    print(f"    - {s}")


def print_near_dup_results(results: List[Dict]) -> None:
    """Print near-duplicate detection results."""
    for r in results:
        print(f"\n  {r['pair']}:")
        print(f"    Texts checked: {r['checked_a']:,} x {r['checked_b']:,}")
        print(f"    Candidate pairs: {r['candidates_checked']:,}")
        print(f"    Near-duplicates (hamming <= {r['threshold']}): {r['near_duplicates']:,}")
        print(f"    Near-dup rate: {r['near_dup_rate']:.4%}")
        print(
            f"    Time: hash={r['time_hash']:.1f}s, search={r['time_search']:.1f}s"
        )
        if r["sample_pairs"]:
            print(f"    Sample near-duplicate pairs:")
            for ta, tb, dist in r["sample_pairs"]:
                print(f"      [dist={dist}]")
                print(f"        A: {ta}...")
                print(f"        B: {tb}...")


# ──────────────────────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────────────────────
def main() -> None:
    t_start = time.time()

    print_header("Data Leakage & Duplication Check")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Script: {__file__}")
    print()

    # ── Step 1: Load all datasets ──
    print_header("Step 1: Loading Datasets")

    print("\n  Training sets:")
    train_data = load_group(TRAIN_SETS)

    print("\n  Validation sets:")
    val_data = load_group(VAL_SETS)

    print("\n  Test sets:")
    test_data = load_group(TEST_SETS)

    print("\n  Fair test sets:")
    fair_data = load_group(FAIR_TEST_SETS)

    # Combine test + fair_test as "evaluation" group
    eval_data = {**test_data, **fair_data}

    # ── Step 2: Build hash sets ──
    print_header("Step 2: Building SHA1 Hash Sets")

    train_hashes = {name: build_hash_set(df) for name, df in train_data.items()}
    val_hashes = {name: build_hash_set(df) for name, df in val_data.items()}
    test_hashes = {name: build_hash_set(df) for name, df in test_data.items()}
    fair_hashes = {name: build_hash_set(df) for name, df in fair_data.items()}
    eval_hashes = {**test_hashes, **fair_hashes}

    # Union of all training hashes for aggregate check
    all_train_hashes: Set[str] = set()
    for h in train_hashes.values():
        all_train_hashes.update(h)
    print(f"\n  Total unique training hashes (union): {len(all_train_hashes):,}")

    all_val_hashes: Set[str] = set()
    for h in val_hashes.values():
        all_val_hashes.update(h)
    print(f"  Total unique validation hashes (union): {len(all_val_hashes):,}")

    all_eval_hashes: Set[str] = set()
    for h in eval_hashes.values():
        all_eval_hashes.update(h)
    print(f"  Total unique eval/test hashes (union): {len(all_eval_hashes):,}")

    # ── Step 3: Train vs Test/Eval overlaps (pairwise) ──
    print_header("Step 3: Train vs Test/Eval Exact Overlap (Pairwise)")

    train_vs_eval_results = []
    for t_name in sorted(train_data.keys()):
        for e_name in sorted(eval_data.keys()):
            result = check_exact_overlap(
                t_name,
                train_data[t_name],
                train_hashes[t_name],
                e_name,
                eval_data[e_name],
                eval_hashes[e_name],
            )
            train_vs_eval_results.append(result)

    print_overlap_table(train_vs_eval_results)

    # Aggregate: union of all train vs each eval set
    print_header("Step 3b: ALL Training (Union) vs Each Eval Set")

    aggregate_results = []
    for e_name in sorted(eval_data.keys()):
        overlap = all_train_hashes & eval_hashes[e_name]
        smaller = min(len(all_train_hashes), len(eval_hashes[e_name]))
        rate = len(overlap) / smaller if smaller > 0 else 0.0

        samples = []
        if len(overlap) > 0:
            count = 0
            for text in eval_data[e_name]["text"]:
                if sha1_hash(text) in overlap:
                    truncated = text[:120] + ("..." if len(text) > 120 else "")
                    samples.append(truncated)
                    count += 1
                    if count >= 3:
                        break

        aggregate_results.append(
            {
                "pair": f"ALL_TRAIN vs {e_name}",
                "size_a": len(all_train_hashes),
                "size_b": len(eval_hashes[e_name]),
                "overlap": len(overlap),
                "rate": rate,
                "samples": samples,
            }
        )

    print_overlap_table(aggregate_results)

    # ── Step 4: Train vs Fair Test overlaps ──
    print_header("Step 4: Train vs Fair Test Exact Overlap")

    train_vs_fair_results = []
    for t_name in sorted(train_data.keys()):
        for f_name in sorted(fair_data.keys()):
            result = check_exact_overlap(
                t_name,
                train_data[t_name],
                train_hashes[t_name],
                f_name,
                fair_data[f_name],
                fair_hashes[f_name],
            )
            train_vs_fair_results.append(result)

    print_overlap_table(train_vs_fair_results)

    # ── Step 5: Val vs Test overlaps ──
    print_header("Step 5: Validation vs Test Exact Overlap")

    val_vs_test_results = []
    for v_name in sorted(val_data.keys()):
        for e_name in sorted(eval_data.keys()):
            result = check_exact_overlap(
                v_name,
                val_data[v_name],
                val_hashes[v_name],
                e_name,
                eval_data[e_name],
                eval_hashes[e_name],
            )
            val_vs_test_results.append(result)

    print_overlap_table(val_vs_test_results)

    # ── Step 6: Internal duplicates ──
    print_header("Step 6: Internal Duplicates Within Each Dataset")

    all_datasets = {**train_data, **val_data, **test_data, **fair_data}
    internal_results = []
    for name in sorted(all_datasets.keys()):
        result = check_internal_duplicates(name, all_datasets[name])
        internal_results.append(result)

    print_internal_dup_table(internal_results)

    # ── Step 7: Near-duplicate detection (SimHash) ──
    print_header("Step 7: Near-Duplicate Detection (SimHash, hamming <= 3)")

    near_dup_results = []

    # merged_v2/train vs each fair_test set
    if "merged_v2/train" in train_data:
        for f_name in sorted(fair_data.keys()):
            result = check_near_duplicates(
                "merged_v2/train",
                train_data["merged_v2/train"],
                f_name,
                fair_data[f_name],
                threshold=3,
                max_samples=5,
                max_check=50000,
            )
            near_dup_results.append(result)

    # Also check core_v3/train vs fair_test if available
    if "core_v3/train" in train_data:
        for f_name in sorted(fair_data.keys()):
            result = check_near_duplicates(
                "core_v3/train",
                train_data["core_v3/train"],
                f_name,
                fair_data[f_name],
                threshold=3,
                max_samples=5,
                max_check=50000,
            )
            near_dup_results.append(result)

    if near_dup_results:
        print_near_dup_results(near_dup_results)
    else:
        print("\n  No near-duplicate checks performed (datasets not available).")

    # ── Summary ──
    elapsed = time.time() - t_start
    print_header("Summary")

    total_leaks_train_eval = sum(r["overlap"] for r in train_vs_eval_results)
    total_leaks_train_fair = sum(r["overlap"] for r in train_vs_fair_results)
    total_leaks_val_test = sum(r["overlap"] for r in val_vs_test_results)
    total_internal_dups = sum(r["dup_rows"] for r in internal_results)
    total_near_dups = sum(r["near_duplicates"] for r in near_dup_results)

    print(f"\n  Train vs Eval (pairwise sum):     {total_leaks_train_eval:,} exact overlaps")
    print(f"  Train vs Fair Test (pairwise sum): {total_leaks_train_fair:,} exact overlaps")
    print(f"  Val vs Test (pairwise sum):        {total_leaks_val_test:,} exact overlaps")
    print(f"  Internal duplicates (all sets):    {total_internal_dups:,} duplicate rows")
    print(f"  Near-duplicates (SimHash):         {total_near_dups:,} pairs (hamming <= 3)")
    print(f"\n  Total elapsed time: {elapsed:.1f}s")

    # Final verdict
    if total_leaks_train_eval == 0 and total_leaks_train_fair == 0:
        print("\n  VERDICT: No data leakage detected between training and evaluation sets.")
    else:
        print(
            "\n  VERDICT: DATA LEAKAGE DETECTED! "
            "Review overlapping pairs above and clean affected datasets."
        )

    if total_leaks_val_test > 0:
        print(
            "  WARNING: Validation-test overlap detected. "
            "This may inflate validation-based model selection."
        )

    if total_internal_dups > 0:
        print(
            f"  NOTE: {total_internal_dups:,} internal duplicates found. "
            "Consider deduplication for cleaner training."
        )

    print(f"\n{SEP_LINE}\n")


if __name__ == "__main__":
    main()
