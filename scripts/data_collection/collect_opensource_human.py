#!/usr/bin/env python3
"""
Download and process open-source Chinese human text datasets.

Datasets:
- CSL: Chinese Scientific Literature (academic abstracts) - 400K samples
- Toutiao: 今日头条新闻分类 (news articles) - 380K samples
- Douban: 豆瓣短评 (book/movie reviews) - various sources

Target: ~2000 high-quality human samples to pair with ~10K AI samples
"""

import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve
import zipfile
import gzip
import csv

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Output directories
RAW_DIR = PROJECT_ROOT / "datasets" / "external" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "external" / "processed"

# Ensure directories exist
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Download a file with progress indication."""
    if dest.exists():
        print(f"  [SKIP] {desc or dest.name} already exists")
        return True

    print(f"  [DOWNLOAD] {desc or url}")
    try:
        def progress_hook(count, block_size, total_size):
            if total_size > 0:
                percent = min(100, count * block_size * 100 // total_size)
                print(f"\r    Progress: {percent}%", end="", flush=True)

        urlretrieve(url, dest, reporthook=progress_hook)
        print()  # newline after progress
        return True
    except Exception as e:
        print(f"\n    [ERROR] Download failed: {e}")
        return False


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())

    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+\.\S+', '', text)

    # Normalize punctuation
    text = text.replace('\u3000', ' ')  # Full-width space

    return text.strip()


def is_valid_sample(text: str, min_length: int = 50, max_length: int = 5000) -> bool:
    """Check if a text sample is valid for training."""
    if not text:
        return False

    length = len(text)
    if length < min_length or length > max_length:
        return False

    # Check for reasonable Chinese character ratio
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    if chinese_chars < length * 0.3:  # At least 30% Chinese
        return False

    return True


# =============================================================================
# CSL Dataset (Academic Abstracts)
# =============================================================================

def download_csl_dataset() -> Optional[Path]:
    """Download CSL dataset from GitHub."""
    print("\n[CSL] Downloading Chinese Scientific Literature dataset...")

    # Try GitHub release first
    csl_url = "https://github.com/ydli-ai/CSL/raw/main/fulltext/data/csl_fulltext.json.gz"
    csl_dest = RAW_DIR / "csl_fulltext.json.gz"

    if not download_file(csl_url, csl_dest, "CSL fulltext"):
        # Fallback to abstracts only
        csl_url = "https://github.com/ydli-ai/CSL/raw/main/csl_data/train.json"
        csl_dest = RAW_DIR / "csl_train.json"
        if not download_file(csl_url, csl_dest, "CSL train.json"):
            print("  [WARN] Could not download CSL dataset")
            return None

    return csl_dest


def process_csl_dataset(file_path: Path, target_count: int = 1000) -> list[dict]:
    """Process CSL dataset and extract academic abstracts."""
    print(f"\n[CSL] Processing {file_path.name}...")

    samples = []

    try:
        if file_path.suffix == ".gz":
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                data = json.load(f)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

        # CSL format: list of dicts with 'abstract' or 'abst' field
        for item in data:
            abstract = item.get('abstract') or item.get('abst', '')
            title = item.get('title', '')

            text = clean_text(abstract)
            if is_valid_sample(text, min_length=100, max_length=3000):
                samples.append({
                    'text': text,
                    'label': 0,  # Human
                    'category': 'Human',
                    'source': 'CSL_Academic',
                    'genre': '学术论文摘要',
                    'title': title
                })

        print(f"  Found {len(samples)} valid academic abstracts")

        # Randomly sample if we have more than target
        if len(samples) > target_count:
            random.seed(42)
            samples = random.sample(samples, target_count)
            print(f"  Sampled {target_count} abstracts")

    except Exception as e:
        print(f"  [ERROR] Failed to process CSL: {e}")

    return samples


# =============================================================================
# Toutiao Dataset (News Articles)
# =============================================================================

def download_toutiao_dataset() -> Optional[Path]:
    """Download Toutiao news classification dataset."""
    print("\n[TOUTIAO] Downloading 今日头条新闻分类 dataset...")

    # Dataset from GitHub
    url = "https://github.com/fateleak/toutiao-text-classfication-dataset/raw/master/toutiao_cat_data.txt.zip"
    dest = RAW_DIR / "toutiao_cat_data.txt.zip"

    if not download_file(url, dest, "Toutiao news"):
        # Try alternative URL
        url2 = "https://raw.githubusercontent.com/fateleak/toutiao-text-classfication-dataset/master/toutiao_cat_data.txt.zip"
        if not download_file(url2, dest, "Toutiao news (alt)"):
            print("  [WARN] Could not download Toutiao dataset")
            return None

    # Extract if zip
    if dest.suffix == ".zip":
        extract_dir = RAW_DIR / "toutiao"
        extract_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(dest, 'r') as z:
            z.extractall(extract_dir)

        # Find the txt file
        txt_files = list(extract_dir.glob("*.txt"))
        if txt_files:
            return txt_files[0]

    return dest


def process_toutiao_dataset(file_path: Path, target_count: int = 500) -> list[dict]:
    """Process Toutiao dataset and extract news articles."""
    print(f"\n[TOUTIAO] Processing {file_path.name}...")

    samples = []

    # Category mapping
    categories = {
        '100': '民生故事',
        '101': '文化文学',
        '102': '娱乐',
        '103': '体育',
        '104': '财经',
        '105': '房产',
        '106': '汽车',
        '107': '教育',
        '108': '科技',
        '109': '军事',
        '110': '旅游',
        '111': '国际',
        '112': '股票',
        '113': '农业',
        '114': '游戏',
    }

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('_!_')
                if len(parts) >= 4:
                    news_id = parts[0]
                    cat_code = parts[1]
                    cat_name = parts[2]
                    title = parts[3]
                    keywords = parts[4] if len(parts) > 4 else ''

                    text = clean_text(title)
                    if is_valid_sample(text, min_length=20, max_length=200):
                        samples.append({
                            'text': text,
                            'label': 0,  # Human
                            'category': 'Human',
                            'source': 'Toutiao_News',
                            'genre': f'新闻标题-{cat_name}',
                            'keywords': keywords
                        })

        print(f"  Found {len(samples)} valid news titles")

        # Sample diverse categories
        if len(samples) > target_count:
            random.seed(42)
            samples = random.sample(samples, target_count)
            print(f"  Sampled {target_count} news items")

    except Exception as e:
        print(f"  [ERROR] Failed to process Toutiao: {e}")

    return samples


# =============================================================================
# Douban Dataset (Reviews)
# =============================================================================

def download_douban_dataset() -> Optional[Path]:
    """Download Douban short reviews dataset."""
    print("\n[DOUBAN] Downloading 豆瓣短评 dataset...")

    # Try the AI Studio dataset (public version)
    # Note: Large datasets may require manual download

    # Try a smaller public subset from HuggingFace or GitHub
    urls_to_try = [
        ("https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv",
         RAW_DIR / "ChnSentiCorp_htl_all.csv", "ChnSentiCorp (酒店评论)"),
        ("https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/waimai_10k/waimai_10k.csv",
         RAW_DIR / "waimai_10k.csv", "外卖评论10K"),
        ("https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/online_shopping_10_cats/online_shopping_10_cats.csv",
         RAW_DIR / "online_shopping_10_cats.csv", "电商评论"),
    ]

    downloaded = []
    for url, dest, desc in urls_to_try:
        if download_file(url, dest, desc):
            downloaded.append(dest)

    if not downloaded:
        print("  [WARN] Could not download any review datasets")
        return None

    return downloaded[0] if len(downloaded) == 1 else downloaded


def process_douban_dataset(file_paths, target_count: int = 500) -> list[dict]:
    """Process Douban/review datasets."""
    print(f"\n[DOUBAN] Processing review datasets...")

    samples = []

    if not isinstance(file_paths, list):
        file_paths = [file_paths]

    for file_path in file_paths:
        if not file_path or not Path(file_path).exists():
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Try different column names
                    text = row.get('review') or row.get('comment') or row.get('text', '')
                    text = clean_text(text)

                    if is_valid_sample(text, min_length=30, max_length=1000):
                        source_name = Path(file_path).stem
                        samples.append({
                            'text': text,
                            'label': 0,  # Human
                            'category': 'Human',
                            'source': f'Review_{source_name}',
                            'genre': '用户评论'
                        })

            print(f"  Found {len(samples)} valid reviews from {file_path.name}")

        except Exception as e:
            print(f"  [ERROR] Failed to process {file_path}: {e}")

    # Sample if needed
    if len(samples) > target_count:
        random.seed(42)
        samples = random.sample(samples, target_count)
        print(f"  Sampled {target_count} reviews total")

    return samples


# =============================================================================
# Main Collection Pipeline
# =============================================================================

def collect_all_human_data(
    csl_count: int = 1000,
    toutiao_count: int = 500,
    douban_count: int = 500
) -> list[dict]:
    """Collect human data from all sources."""

    all_samples = []

    # 1. CSL Academic Abstracts
    csl_path = download_csl_dataset()
    if csl_path:
        csl_samples = process_csl_dataset(csl_path, target_count=csl_count)
        all_samples.extend(csl_samples)

    # 2. Toutiao News
    toutiao_path = download_toutiao_dataset()
    if toutiao_path:
        toutiao_samples = process_toutiao_dataset(toutiao_path, target_count=toutiao_count)
        all_samples.extend(toutiao_samples)

    # 3. Douban Reviews
    douban_paths = download_douban_dataset()
    if douban_paths:
        douban_samples = process_douban_dataset(douban_paths, target_count=douban_count)
        all_samples.extend(douban_samples)

    return all_samples


def save_collected_data(samples: list[dict], output_name: str = "human_opensource") -> Path:
    """Save collected samples to unified format."""

    output_path = PROCESSED_DIR / f"{output_name}.csv"

    fieldnames = ['text', 'label', 'category', 'source', 'genre']

    with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            # Only write the standard fields
            row = {k: sample.get(k, '') for k in fieldnames}
            writer.writerow(row)

    print(f"\n[SAVED] {len(samples)} samples to {output_path}")

    # Also save as JSONL
    jsonl_path = PROCESSED_DIR / f"{output_name}.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"[SAVED] {len(samples)} samples to {jsonl_path}")

    return output_path


def print_statistics(samples: list[dict]):
    """Print statistics about collected data."""

    print("\n" + "="*60)
    print("COLLECTION STATISTICS")
    print("="*60)

    print(f"\nTotal samples: {len(samples)}")

    # By source
    sources = {}
    for s in samples:
        src = s.get('source', 'Unknown')
        sources[src] = sources.get(src, 0) + 1

    print("\nBy source:")
    for src, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {src}: {count}")

    # By genre
    genres = {}
    for s in samples:
        genre = s.get('genre', 'Unknown')
        genres[genre] = genres.get(genre, 0) + 1

    print("\nBy genre (top 10):")
    for genre, count in sorted(genres.items(), key=lambda x: -x[1])[:10]:
        print(f"  {genre}: {count}")

    # Text length distribution
    lengths = [len(s['text']) for s in samples]
    print(f"\nText length statistics:")
    print(f"  Min: {min(lengths)}")
    print(f"  Max: {max(lengths)}")
    print(f"  Avg: {sum(lengths)/len(lengths):.0f}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Collect open-source Chinese human text data")
    parser.add_argument('--csl', type=int, default=1000, help='Number of CSL samples')
    parser.add_argument('--toutiao', type=int, default=500, help='Number of Toutiao samples')
    parser.add_argument('--douban', type=int, default=500, help='Number of Douban samples')
    parser.add_argument('--output', type=str, default='human_opensource', help='Output filename prefix')

    args = parser.parse_args()

    print("="*60)
    print("Open-Source Human Text Data Collection")
    print("="*60)
    print(f"Target: CSL={args.csl}, Toutiao={args.toutiao}, Douban={args.douban}")

    # Collect data
    samples = collect_all_human_data(
        csl_count=args.csl,
        toutiao_count=args.toutiao,
        douban_count=args.douban
    )

    if not samples:
        print("\n[ERROR] No samples collected!")
        return 1

    # Save data
    output_path = save_collected_data(samples, args.output)

    # Print statistics
    print_statistics(samples)

    print("\n" + "="*60)
    print("Collection complete!")
    print(f"Output: {output_path}")
    print("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
