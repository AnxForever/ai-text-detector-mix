#!/usr/bin/env python3
"""
Deduplicate generated AI text samples.

Removes near-duplicate samples based on:
1. Exact text duplicates
2. Near-duplicate (first N characters match)
3. High similarity (optional, using simple hashing)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


def compute_text_signature(text: str, prefix_len: int = 100) -> str:
    """Compute a signature for near-duplicate detection."""
    # Use first N characters after normalization
    normalized = "".join(text.split())[:prefix_len]
    return normalized


def deduplicate_records(
    records: List[Dict[str, Any]],
    prefix_len: int = 100,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Deduplicate records based on text similarity.

    Returns:
        Tuple of (unique_records, duplicate_records)
    """
    seen_signatures: Set[str] = set()
    seen_exact: Set[str] = set()
    unique: List[Dict[str, Any]] = []
    duplicates: List[Dict[str, Any]] = []

    for record in records:
        text = record.get("text", "")
        if not text:
            duplicates.append(record)
            continue

        # Check exact duplicate
        if text in seen_exact:
            record["duplicate_reason"] = "exact"
            duplicates.append(record)
            continue

        # Check near-duplicate
        signature = compute_text_signature(text, prefix_len)
        if signature in seen_signatures:
            record["duplicate_reason"] = "near_duplicate"
            duplicates.append(record)
            continue

        # New unique record
        seen_exact.add(text)
        seen_signatures.add(signature)
        unique.append(record)

    return unique, duplicates


def process_files(
    input_paths: List[Path],
    output_path: Path,
    duplicates_path: Path | None = None,
    prefix_len: int = 100,
) -> Dict[str, int]:
    """Process multiple JSONL files and deduplicate."""
    # Load all records
    all_records: List[Dict[str, Any]] = []
    for path in input_paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        record["_source_file"] = path.name
                        all_records.append(record)
                    except json.JSONDecodeError:
                        pass

    print(f"Loaded {len(all_records)} records from {len(input_paths)} files")

    # Deduplicate
    unique, duplicates = deduplicate_records(all_records, prefix_len)

    # Write unique records
    with output_path.open("w", encoding="utf-8") as f:
        for record in unique:
            # Remove internal tracking field
            record.pop("_source_file", None)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Write duplicates if requested
    if duplicates_path and duplicates:
        with duplicates_path.open("w", encoding="utf-8") as f:
            for record in duplicates:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    stats = {
        "total": len(all_records),
        "unique": len(unique),
        "duplicates": len(duplicates),
        "exact_duplicates": sum(1 for d in duplicates if d.get("duplicate_reason") == "exact"),
        "near_duplicates": sum(1 for d in duplicates if d.get("duplicate_reason") == "near_duplicate"),
    }
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Deduplicate AI text samples")
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input JSONL files or directory",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output JSONL file for unique records",
    )
    parser.add_argument(
        "--duplicates",
        help="Output file for duplicate records",
    )
    parser.add_argument(
        "--prefix-len",
        type=int,
        default=100,
        help="Prefix length for near-duplicate detection (default: 100)",
    )
    args = parser.parse_args()

    # Collect input files
    input_paths: List[Path] = []
    for inp in args.inputs:
        path = Path(inp)
        if path.is_dir():
            input_paths.extend(sorted(path.glob("*_cleaned.jsonl")))
        elif path.exists():
            input_paths.append(path)

    if not input_paths:
        print("No input files found", file=sys.stderr)
        return 1

    output_path = Path(args.output)
    duplicates_path = Path(args.duplicates) if args.duplicates else None

    stats = process_files(input_paths, output_path, duplicates_path, args.prefix_len)

    print()
    print("=== Deduplication Results ===")
    print(f"Total records: {stats['total']}")
    print(f"Unique records: {stats['unique']}")
    print(f"Duplicates removed: {stats['duplicates']}")
    print(f"  - Exact duplicates: {stats['exact_duplicates']}")
    print(f"  - Near duplicates: {stats['near_duplicates']}")
    print(f"Deduplication rate: {100 * stats['duplicates'] / max(stats['total'], 1):.1f}%")
    print()
    print(f"Output: {output_path}")
    if duplicates_path:
        print(f"Duplicates: {duplicates_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
