#!/usr/bin/env python3
"""
第一步：构建干净测试集并保存到磁盘
后续评估脚本直接加载，避免重复计算
"""
import os, json, hashlib
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
os.chdir(PROJECT_ROOT)

# 所有模型训练数据路径
TRAIN_PATHS = [
    "datasets/active/core_v1/train.csv",
    "datasets/active/core_v2/train.csv",
    "datasets/active/core_v3/train.csv",
    "datasets/merged_v1/train.csv",
    "datasets/merged_v2/train.csv",
    "datasets/paired/paired_v3_all_train.csv",
]

def text_hash(text):
    return hashlib.md5(str(text)[:200].encode('utf-8')).hexdigest()

def main():
    print("收集所有训练数据指纹...")
    all_hashes = set()
    for p in TRAIN_PATHS:
        fp = PROJECT_ROOT / p
        if not fp.exists():
            print(f"  跳过: {p}")
            continue
        try:
            df = pd.read_csv(fp, encoding='utf-8-sig', usecols=['text'])
            hashes = set(df['text'].apply(text_hash))
            print(f"  {p}: {len(hashes)} 条")
            all_hashes.update(hashes)
        except Exception as e:
            print(f"  错误 {p}: {e}")
    print(f"训练数据合计 {len(all_hashes)} 个唯一指纹\n")

    output_dir = PROJECT_ROOT / "datasets/eval/fair_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 测试集1: core_v1/test 干净版 ──
    df = pd.read_csv("datasets/active/core_v1/test.csv", encoding='utf-8-sig')
    df['_hash'] = df['text'].apply(text_hash)
    clean = df[~df['_hash'].isin(all_hashes)].drop(columns=['_hash'])
    leaked = len(df) - len(clean)
    print(f"core_v1/test: {len(df)} → 干净 {len(clean)} (泄露 {leaked})")
    clean.to_csv(output_dir / "core_v1_test_clean.csv", index=False, encoding='utf-8-sig')

    # ── 测试集2: merged_v2/val 干净版 ──
    p = PROJECT_ROOT / "datasets/merged_v2/val.csv"
    if p.exists():
        df = pd.read_csv(p, encoding='utf-8-sig')
        df['_hash'] = df['text'].apply(text_hash)
        clean = df[~df['_hash'].isin(all_hashes)].drop(columns=['_hash'])
        leaked = len(df) - len(clean)
        print(f"merged_v2/val: {len(df)} → 干净 {len(clean)} (泄露 {leaked})")
        clean.to_csv(output_dir / "merged_v2_val_clean.csv", index=False, encoding='utf-8-sig')

    # ── 测试集3: 独立Human数据 ──
    human_dfs = []
    sources = [
        ("datasets/human_consolidated/all_human_consolidated.jsonl", "jsonl"),
        ("datasets/defense_patch/defense_patch_v2.jsonl", "jsonl"),
        ("datasets/external/processed/human_opensource.jsonl", "jsonl"),
        ("datasets/human_consolidated/toutiao_tech_human.jsonl", "jsonl"),
    ]
    for src, fmt in sources:
        fp = PROJECT_ROOT / src
        if not fp.exists():
            continue
        try:
            df = pd.read_json(fp, lines=True)
            df['_hash'] = df['text'].apply(text_hash)
            clean = df[~df['_hash'].isin(all_hashes)]
            print(f"{src}: {len(df)} → 干净 {len(clean)}")
            human_dfs.append(clean)
        except Exception as e:
            print(f"  错误 {src}: {e}")

    if human_dfs:
        combined = pd.concat(human_dfs, ignore_index=True)
        combined = combined.drop_duplicates(subset=['_hash'])
        combined = combined.drop(columns=['_hash'])
        # 只保留 text 和 label 列
        keep_cols = [c for c in ['text', 'label', 'source', 'category'] if c in combined.columns]
        combined = combined[keep_cols]
        print(f"\n独立数据合计: {len(combined)} 条")
        print(f"  label分布: {combined['label'].value_counts().to_dict()}")
        combined.to_csv(output_dir / "independent_data.csv", index=False, encoding='utf-8-sig')

    # 保存元信息
    meta = {
        "train_fingerprints": len(all_hashes),
        "test_sets": {}
    }
    for f in output_dir.glob("*.csv"):
        df = pd.read_csv(f, encoding='utf-8-sig')
        meta["test_sets"][f.stem] = {
            "count": len(df),
            "labels": df['label'].value_counts().to_dict() if 'label' in df.columns else {}
        }
    with open(output_dir / "meta.json", 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 干净测试集已保存到 {output_dir}")
    print(json.dumps(meta, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
