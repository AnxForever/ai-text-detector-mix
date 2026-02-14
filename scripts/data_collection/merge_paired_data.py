#!/usr/bin/env python3
"""
Merge human text data with senior's AI-generated data into a unified training dataset.

Pipeline:
1. Load senior's AI data from 新建文件夹/
2. Load collected human data from datasets/external/processed/
3. Clean and deduplicate
4. Balance categories
5. Output unified training set
"""

import csv
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Directories
SENIOR_DATA_DIR = PROJECT_ROOT / "新建文件夹"
HUMAN_DATA_DIR = PROJECT_ROOT / "datasets" / "external" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "paired"


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'https?://\S+', '', text)
    text = text.replace('\u3000', ' ')
    # Remove BOM
    text = text.replace('\ufeff', '')
    return text.strip()


def get_text_hash(text: str) -> str:
    """Generate hash for deduplication."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def is_valid_sample(text: str, min_length: int = 50, max_length: int = 5000) -> bool:
    """Check if a text sample is valid."""
    if not text:
        return False
    length = len(text)
    if length < min_length or length > max_length:
        return False
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    if chinese_chars < length * 0.2:
        return False
    return True


# =============================================================================
# Load Data Functions
# =============================================================================

def load_senior_ai_data() -> list[dict]:
    """Load AI data from senior's directories."""
    print("\n[AI DATA] Loading senior's AI-generated data...")

    samples = []
    datasets_dirs = [
        SENIOR_DATA_DIR / "auto_generated_datasets4000",
        SENIOR_DATA_DIR / "auto_generated_datasets5000",
        SENIOR_DATA_DIR / "auto_generated_datasets_dt2000",
    ]

    for ds_dir in datasets_dirs:
        if not ds_dir.exists():
            print(f"  [SKIP] {ds_dir.name} not found")
            continue

        # Find the combined dataset CSV
        csv_files = list(ds_dir.glob("AUTO_COMBINED_DATASET_*.csv"))
        if not csv_files:
            print(f"  [SKIP] No combined CSV in {ds_dir.name}")
            continue

        # Use the latest one
        csv_file = sorted(csv_files)[-1]
        print(f"  Loading {csv_file.name}...")

        with open(csv_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                text = clean_text(row.get('text_content', ''))
                if not is_valid_sample(text):
                    continue

                samples.append({
                    'text': text,
                    'label': 1,  # AI
                    'category': 'AI',
                    'source': f"Senior_{row.get('source_model', 'unknown')}",
                    'genre': row.get('genre', '未知')[:50],
                })
                count += 1

            print(f"    Loaded {count} samples from {csv_file.name}")

    print(f"  Total AI samples: {len(samples)}")
    return samples


def load_human_data() -> list[dict]:
    """Load human data from processed external datasets."""
    print("\n[HUMAN DATA] Loading collected human text data...")

    samples = []
    csv_files = list(HUMAN_DATA_DIR.glob("*.csv"))

    for csv_file in csv_files:
        print(f"  Loading {csv_file.name}...")

        with open(csv_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                text = clean_text(row.get('text', ''))
                if not is_valid_sample(text, min_length=30):
                    continue

                samples.append({
                    'text': text,
                    'label': 0,  # Human
                    'category': 'Human',
                    'source': row.get('source', 'Unknown'),
                    'genre': row.get('genre', '未知'),
                })
                count += 1

            print(f"    Loaded {count} samples")

    print(f"  Total human samples: {len(samples)}")
    return samples


# =============================================================================
# Data Processing
# =============================================================================

def deduplicate_samples(samples: list[dict]) -> list[dict]:
    """Remove duplicate samples based on text hash."""
    print("\n[DEDUP] Removing duplicates...")

    seen_hashes = set()
    unique_samples = []

    for sample in samples:
        text_hash = get_text_hash(sample['text'])
        if text_hash not in seen_hashes:
            seen_hashes.add(text_hash)
            unique_samples.append(sample)

    removed = len(samples) - len(unique_samples)
    print(f"  Removed {removed} duplicates, {len(unique_samples)} unique samples remaining")

    return unique_samples


def balance_dataset(
    samples: list[dict],
    max_ai: Optional[int] = None,
    max_human: Optional[int] = None,
    target_ratio: float = 1.0  # AI:Human ratio
) -> list[dict]:
    """Balance the dataset between AI and human samples."""
    print("\n[BALANCE] Balancing dataset...")

    ai_samples = [s for s in samples if s['label'] == 1]
    human_samples = [s for s in samples if s['label'] == 0]

    print(f"  Before: AI={len(ai_samples)}, Human={len(human_samples)}")

    # Apply max limits if specified
    if max_ai and len(ai_samples) > max_ai:
        random.seed(42)
        ai_samples = random.sample(ai_samples, max_ai)

    if max_human and len(human_samples) > max_human:
        random.seed(42)
        human_samples = random.sample(human_samples, max_human)

    # Balance to target ratio
    if target_ratio > 0:
        desired_ai = int(len(human_samples) * target_ratio)
        if len(ai_samples) > desired_ai:
            random.seed(42)
            ai_samples = random.sample(ai_samples, desired_ai)

    print(f"  After: AI={len(ai_samples)}, Human={len(human_samples)}")

    balanced = ai_samples + human_samples
    random.seed(42)
    random.shuffle(balanced)

    return balanced


# =============================================================================
# Output Functions
# =============================================================================

def save_unified_dataset(samples: list[dict], output_name: str = "paired_v1") -> Path:
    """Save the unified dataset."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Split into train/val/test (80/10/10)
    random.seed(42)
    random.shuffle(samples)

    n = len(samples)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]

    # Save each split
    fieldnames = ['text', 'label', 'category', 'source', 'genre']

    for split_name, split_samples in [
        ('train', train_samples),
        ('val', val_samples),
        ('test', test_samples)
    ]:
        output_path = OUTPUT_DIR / f"{output_name}_{split_name}.csv"
        with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for sample in split_samples:
                row = {k: sample.get(k, '') for k in fieldnames}
                writer.writerow(row)

        print(f"  [SAVED] {split_name}: {len(split_samples)} samples → {output_path}")

    # Save combined as well
    combined_path = OUTPUT_DIR / f"{output_name}_combined.csv"
    with open(combined_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            row = {k: sample.get(k, '') for k in fieldnames}
            writer.writerow(row)

    print(f"  [SAVED] combined: {len(samples)} samples → {combined_path}")

    # Save metadata
    meta = {
        'total_samples': len(samples),
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'test_samples': len(test_samples),
        'ai_samples': sum(1 for s in samples if s['label'] == 1),
        'human_samples': sum(1 for s in samples if s['label'] == 0),
        'sources': dict(Counter(s['source'] for s in samples)),
        'genres': dict(Counter(s['genre'] for s in samples)),
    }

    meta_path = OUTPUT_DIR / f"{output_name}_meta.json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return combined_path


def print_statistics(samples: list[dict]):
    """Print dataset statistics."""
    print("\n" + "=" * 60)
    print("FINAL DATASET STATISTICS")
    print("=" * 60)

    print(f"\nTotal samples: {len(samples)}")

    # By label
    labels = Counter(s['label'] for s in samples)
    print(f"\nBy label:")
    print(f"  Human (0): {labels.get(0, 0)}")
    print(f"  AI (1): {labels.get(1, 0)}")

    # By source (top 10)
    sources = Counter(s['source'] for s in samples)
    print(f"\nBy source (top 10):")
    for src, count in sources.most_common(10):
        print(f"  {src}: {count}")

    # By genre (top 10)
    genres = Counter(s['genre'] for s in samples)
    print(f"\nBy genre (top 10):")
    for genre, count in genres.most_common(10):
        print(f"  {genre[:40]}: {count}")

    # Text length
    lengths = [len(s['text']) for s in samples]
    print(f"\nText length:")
    print(f"  Min: {min(lengths)}")
    print(f"  Max: {max(lengths)}")
    print(f"  Avg: {sum(lengths) // len(lengths)}")


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Merge human and AI data into unified training set")
    parser.add_argument('--max-ai', type=int, default=None, help='Max AI samples to use')
    parser.add_argument('--max-human', type=int, default=None, help='Max human samples to use')
    parser.add_argument('--ratio', type=float, default=1.0, help='Target AI:Human ratio')
    parser.add_argument('--output', type=str, default='paired_v1', help='Output name prefix')

    args = parser.parse_args()

    print("=" * 60)
    print("Human + AI Data Merge Pipeline")
    print("=" * 60)

    # Load data
    ai_samples = load_senior_ai_data()
    human_samples = load_human_data()

    if not ai_samples:
        print("\n[ERROR] No AI samples loaded!")
        return 1

    if not human_samples:
        print("\n[ERROR] No human samples loaded!")
        return 1

    # Combine
    all_samples = ai_samples + human_samples
    print(f"\n[TOTAL] Combined: {len(all_samples)} samples")

    # Deduplicate
    all_samples = deduplicate_samples(all_samples)

    # Balance
    all_samples = balance_dataset(
        all_samples,
        max_ai=args.max_ai,
        max_human=args.max_human,
        target_ratio=args.ratio
    )

    # Save
    print("\n[OUTPUT] Saving unified dataset...")
    save_unified_dataset(all_samples, args.output)

    # Stats
    print_statistics(all_samples)

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
