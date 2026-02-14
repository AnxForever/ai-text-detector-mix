#!/usr/bin/env python3
"""
合并Human补充数据到训练集

用法:
    python scripts/data_cleaning/merge_human_supplement.py
    python scripts/data_cleaning/merge_human_supplement.py --dry-run
"""
import argparse
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from collections import Counter
import hashlib

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def load_current_training_data():
    """加载当前训练数据"""
    merged_dir = PROJECT_ROOT / "datasets" / "merged_v1"

    train_df = pd.read_csv(merged_dir / "train.csv", encoding='utf-8-sig')
    val_df = pd.read_csv(merged_dir / "val.csv", encoding='utf-8-sig')

    print(f"当前训练集: {len(train_df)} 条")
    print(f"当前验证集: {len(val_df)} 条")

    return train_df, val_df


def load_human_supplement():
    """加载Human补充数据"""
    supplement_file = PROJECT_ROOT / "datasets" / "human_consolidated" / "human_supplement_final.jsonl"

    samples = []
    with open(supplement_file, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))

    df = pd.DataFrame(samples)
    print(f"Human补充数据: {len(df)} 条")

    return df


def deduplicate(train_df, supplement_df):
    """去重: 移除supplement中与train重复的文本"""
    # 用文本hash去重
    def text_hash(text):
        return hashlib.md5(str(text)[:200].encode()).hexdigest()

    train_hashes = set(train_df['text'].apply(text_hash))

    original_count = len(supplement_df)
    supplement_df = supplement_df[~supplement_df['text'].apply(text_hash).isin(train_hashes)]

    print(f"去重: {original_count} -> {len(supplement_df)} (移除 {original_count - len(supplement_df)} 重复)")

    return supplement_df


def merge_datasets(train_df, val_df, supplement_df, val_ratio=0.1):
    """合并数据集"""
    # 从supplement中分出验证集
    if len(supplement_df) > 100:
        supp_train, supp_val = train_test_split(
            supplement_df,
            test_size=val_ratio,
            random_state=42
        )
    else:
        supp_train = supplement_df
        supp_val = pd.DataFrame()

    # 确保列一致
    required_cols = ['text', 'label', 'source']

    for col in required_cols:
        if col not in supp_train.columns:
            supp_train[col] = ''
        if col not in supp_val.columns:
            supp_val[col] = ''

    # 选择需要的列
    train_cols = list(train_df.columns)
    supp_train = supp_train.reindex(columns=train_cols, fill_value='')
    supp_val = supp_val.reindex(columns=train_cols, fill_value='')

    # 合并
    new_train = pd.concat([train_df, supp_train], ignore_index=True)
    new_val = pd.concat([val_df, supp_val], ignore_index=True)

    print(f"合并后训练集: {len(new_train)} 条 (+{len(supp_train)})")
    print(f"合并后验证集: {len(new_val)} 条 (+{len(supp_val)})")

    return new_train, new_val


def print_statistics(df, name="数据集"):
    """打印数据集统计"""
    print(f"\n{name} 统计:")
    print(f"  总数: {len(df)}")

    if 'label' in df.columns:
        labels = df['label'].value_counts()
        human = labels.get(0, 0)
        ai = labels.get(1, 0)
        print(f"  Human: {human} ({human/len(df)*100:.1f}%)")
        print(f"  AI: {ai} ({ai/len(df)*100:.1f}%)")

    if 'source' in df.columns:
        print(f"  来源数: {df['source'].nunique()}")
        top_sources = df['source'].value_counts().head(5)
        print(f"  Top来源: {dict(top_sources)}")


def save_merged_data(train_df, val_df, output_dir):
    """保存合并后的数据"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_dir / "train.csv", index=False, encoding='utf-8-sig')
    val_df.to_csv(output_dir / "val.csv", index=False, encoding='utf-8-sig')

    # 保存合并日志
    log = {
        "timestamp": datetime.now().isoformat(),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "human_count": int((train_df['label'] == 0).sum()),
        "ai_count": int((train_df['label'] == 1).sum()),
        "sources": {k: int(v) for k, v in train_df['source'].value_counts().head(20).items()},
    }

    with open(output_dir / "merge_log.json", 'w', encoding='utf-8') as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 已保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="合并Human补充数据")
    parser.add_argument('--dry-run', action='store_true', help='仅分析，不保存')
    parser.add_argument('--output', default='datasets/merged_v2', help='输出目录')
    args = parser.parse_args()

    print("=" * 60)
    print("合并Human补充数据到训练集")
    print("=" * 60)

    # 1. 加载数据
    train_df, val_df = load_current_training_data()
    supplement_df = load_human_supplement()

    # 2. 去重
    supplement_df = deduplicate(train_df, supplement_df)

    # 3. 合并
    new_train, new_val = merge_datasets(train_df, val_df, supplement_df)

    # 4. 统计
    print_statistics(new_train, "新训练集")
    print_statistics(new_val, "新验证集")

    # 5. 保存
    if not args.dry_run:
        output_dir = PROJECT_ROOT / args.output
        save_merged_data(new_train, new_val, output_dir)
        print(f"\n下一步: 使用新数据训练模型")
        print(f"  python scripts/training/train_merged_v1.py --data {output_dir}")
    else:
        print("\n[DRY-RUN] 未保存数据")


if __name__ == "__main__":
    main()
