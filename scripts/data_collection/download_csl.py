#!/usr/bin/env python3
"""
Download CSL (Chinese Scientific Literature) academic abstracts.
Uses HuggingFace datasets library for reliable access.
"""

import json
import os
import random
import re
import sys
import csv
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DIR = PROJECT_ROOT / "datasets" / "external" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'https?://\S+', '', text)
    text = text.replace('\u3000', ' ')
    return text.strip()


def is_valid_sample(text: str, min_length: int = 100, max_length: int = 3000) -> bool:
    """Check if a text sample is valid for training."""
    if not text:
        return False
    length = len(text)
    if length < min_length or length > max_length:
        return False
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    if chinese_chars < length * 0.3:
        return False
    return True


def download_csl_from_huggingface(target_count: int = 1000) -> list[dict]:
    """Download CSL dataset from HuggingFace."""
    print("[CSL] Downloading from HuggingFace...")

    try:
        from datasets import load_dataset

        # Try the CSL paper classification dataset
        dataset = load_dataset("ydli-ai/csl", split="train", trust_remote_code=True)

        samples = []
        for item in dataset:
            abstract = item.get('abst') or item.get('abstract', '')
            title = item.get('title', '')

            text = clean_text(abstract)
            if is_valid_sample(text):
                samples.append({
                    'text': text,
                    'label': 0,  # Human
                    'category': 'Human',
                    'source': 'CSL_Academic',
                    'genre': '学术论文摘要',
                    'title': title
                })

        print(f"  Found {len(samples)} valid academic abstracts")

        if len(samples) > target_count:
            random.seed(42)
            samples = random.sample(samples, target_count)
            print(f"  Sampled {target_count} abstracts")

        return samples

    except Exception as e:
        print(f"  [ERROR] HuggingFace CSL failed: {e}")
        return try_alternative_sources(target_count)


def try_alternative_sources(target_count: int) -> list[dict]:
    """Try alternative academic text sources."""
    print("\n[ALT] Trying alternative academic sources...")

    samples = []

    try:
        from datasets import load_dataset

        # Try Chinese Wikipedia as alternative
        print("  Loading Chinese Wikipedia subset...")
        wiki = load_dataset("pleisto/wikipedia-cn-20230720-filtered", split="train", trust_remote_code=True)

        for item in wiki:
            text = clean_text(item.get('completion', ''))
            if is_valid_sample(text, min_length=100, max_length=2000):
                # Filter for academic-like content
                if any(kw in text for kw in ['研究', '分析', '方法', '结果', '实验', '理论', '科学', '学术', '论文']):
                    samples.append({
                        'text': text,
                        'label': 0,
                        'category': 'Human',
                        'source': 'Wikipedia_CN',
                        'genre': '百科知识',
                    })
                    if len(samples) >= target_count:
                        break

        print(f"  Found {len(samples)} valid Wikipedia entries")

    except Exception as e:
        print(f"  [ERROR] Wikipedia failed: {e}")

    # Try CLUE benchmark datasets
    if len(samples) < target_count:
        try:
            print("  Loading CLUE TNEWS dataset...")
            tnews = load_dataset("clue", "tnews", split="train", trust_remote_code=True)

            for item in tnews:
                text = clean_text(item.get('sentence', ''))
                if is_valid_sample(text, min_length=50, max_length=500):
                    samples.append({
                        'text': text,
                        'label': 0,
                        'category': 'Human',
                        'source': 'CLUE_TNEWS',
                        'genre': '新闻短文',
                    })
                    if len(samples) >= target_count:
                        break

            print(f"  Now have {len(samples)} samples total")

        except Exception as e:
            print(f"  [WARN] CLUE failed: {e}")

    return samples[:target_count]


def save_data(samples: list[dict], filename: str = "csl_academic") -> Path:
    """Save collected samples to unified format."""
    output_path = PROCESSED_DIR / f"{filename}.csv"

    fieldnames = ['text', 'label', 'category', 'source', 'genre']

    with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            row = {k: sample.get(k, '') for k in fieldnames}
            writer.writerow(row)

    print(f"\n[SAVED] {len(samples)} samples to {output_path}")

    jsonl_path = PROCESSED_DIR / f"{filename}.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    return output_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download CSL academic abstracts")
    parser.add_argument('--count', type=int, default=1000, help='Number of samples to collect')
    parser.add_argument('--output', type=str, default='csl_academic', help='Output filename')
    args = parser.parse_args()

    print("=" * 60)
    print("CSL Academic Abstracts Collection")
    print("=" * 60)
    print(f"Target: {args.count} samples")

    samples = download_csl_from_huggingface(args.count)

    if not samples:
        print("\n[ERROR] No samples collected!")
        return 1

    save_data(samples, args.output)

    # Stats
    print(f"\nTotal samples: {len(samples)}")
    sources = {}
    for s in samples:
        src = s.get('source', 'Unknown')
        sources[src] = sources.get(src, 0) + 1
    print("By source:")
    for src, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {src}: {count}")

    lengths = [len(s['text']) for s in samples]
    print(f"\nText length: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)//len(lengths)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
