#!/usr/bin/env python3
"""
数据合并脚本 - 为 V8 训练准备扩充数据集

将 merged_v2 + 4 个新数据源合并为 merged_v3：
1. my_generated_ai 未使用部分（多模型 AI 文本）
2. human_formal_samples.jsonl（LCSTS 新闻摘要）
3. human_supplement template 子集（口语/通知类人类文本）
4. defense_patch_v2.jsonl（正式文本防御数据）

用法:
    python scripts/data_cleaning/merge_for_v8.py [--output datasets/merged_v3] [--dry-run]
"""
import os
import json
import hashlib
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent.parent


def load_merged_v2():
    """加载 merged_v2 训练集和验证集"""
    train_path = ROOT / "datasets" / "merged_v2" / "train.csv"
    val_path = ROOT / "datasets" / "merged_v2" / "val.csv"

    train_df = pd.read_csv(train_path, encoding="utf-8-sig")
    val_df = pd.read_csv(val_path, encoding="utf-8-sig")

    print(f"[merged_v2] train: {len(train_df)}, val: {len(val_df)}")
    return train_df, val_df


def load_generated_ai_unused(existing_texts: set):
    """加载 my_generated_ai 中未被 merged_v2 使用的样本"""
    path = ROOT / "datasets" / "my_generated_ai" / "all_generated.jsonl"
    if not path.exists():
        print(f"[SKIP] {path} 不存在")
        return pd.DataFrame()

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            text = d.get("text", "").strip()
            if not text:
                continue
            # 用文本前200字符的hash去重（避免全文比较太慢）
            text_key = text[:200]
            if text_key in existing_texts:
                continue
            model = d.get("model", "unknown")
            records.append({
                "text": text,
                "label": 1,  # AI 生成
                "source": f"generated_ai_{model}",
            })

    df = pd.DataFrame(records)
    print(f"[generated_ai] 未使用部分: {len(df)} 条")
    if len(df) > 0:
        models = df["source"].value_counts()
        for src, cnt in models.head(5).items():
            print(f"  {src}: {cnt}")
    return df


def load_human_formal_samples():
    """加载 LCSTS 新闻摘要人类文本"""
    path = ROOT / "datasets" / "human_formal_samples.jsonl"
    if not path.exists():
        print(f"[SKIP] {path} 不存在")
        return pd.DataFrame()

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            text = d.get("text", "").strip()
            if not text:
                continue
            records.append({
                "text": text,
                "label": int(d.get("label", 0)),
                "source": d.get("source", "LCSTS-news"),
            })

    df = pd.DataFrame(records)
    print(f"[human_formal_samples] {len(df)} 条")
    return df


def load_human_supplement_templates():
    """加载 human_supplement 中的 template 子集"""
    path = ROOT / "datasets" / "human_supplement" / "diverse_human_samples.csv"
    if not path.exists():
        print(f"[SKIP] {path} 不存在")
        return pd.DataFrame()

    df = pd.read_csv(path, encoding="utf-8-sig")
    # 过滤 template 开头的 source
    template_mask = df["source"].str.startswith("template_", na=False)
    df = df[template_mask][["text", "label", "source"]].copy()

    print(f"[human_supplement templates] {len(df)} 条")
    if len(df) > 0:
        for src, cnt in df["source"].value_counts().items():
            print(f"  {src}: {cnt}")
    return df


def load_defense_patch():
    """加载 defense_patch v2 数据"""
    path = ROOT / "datasets" / "defense_patch" / "defense_patch_v2.jsonl"
    if not path.exists():
        print(f"[SKIP] {path} 不存在")
        return pd.DataFrame()

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            text = d.get("text", "").strip()
            if not text:
                continue
            records.append({
                "text": text,
                "label": int(d.get("label", 0)),
                "source": d.get("source", "defense_patch"),
            })

    df = pd.DataFrame(records)
    print(f"[defense_patch_v2] {len(df)} 条")
    return df


def deduplicate(df, column="text"):
    """基于文本去重"""
    before = len(df)
    df = df.drop_duplicates(subset=[column], keep="first")
    after = len(df)
    if before != after:
        print(f"  去重: {before} -> {after} (移除 {before - after})")
    return df


def split_train_val(df, val_ratio=0.2, seed=42):
    """按标签分层划分训练/验证集"""
    from sklearn.model_selection import train_test_split

    train_df, val_df = train_test_split(
        df, test_size=val_ratio, random_state=seed, stratify=df["label"]
    )
    return train_df, val_df


def print_summary(df, name="dataset"):
    """打印数据集摘要"""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  总样本数: {len(df)}")
    label_counts = df["label"].value_counts().sort_index()
    for label, cnt in label_counts.items():
        pct = cnt / len(df) * 100
        tag = "Human" if label == 0 else "AI"
        print(f"  Label {label} ({tag}): {cnt} ({pct:.1f}%)")

    if "source" in df.columns:
        print(f"  来源数: {df['source'].nunique()}")
        top_sources = df["source"].value_counts().head(10)
        for src, cnt in top_sources.items():
            print(f"    {src}: {cnt}")

    lengths = df["text"].str.len()
    print(f"  长度: 均值={lengths.mean():.0f}, "
          f"中位数={lengths.median():.0f}, "
          f"std={lengths.std():.0f}")


def main():
    parser = argparse.ArgumentParser(description="合并数据集为 merged_v3")
    parser.add_argument(
        "--output", default="datasets/merged_v3",
        help="输出目录（默认 datasets/merged_v3）"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只分析不写入"
    )
    args = parser.parse_args()

    output_dir = ROOT / args.output

    print("=" * 60)
    print("  数据合并: merged_v2 + 新数据 -> merged_v3")
    print("=" * 60)

    # 1. 加载 merged_v2
    train_v2, val_v2 = load_merged_v2()
    merged_v2_all = pd.concat([train_v2, val_v2], ignore_index=True)

    # 建立已有文本索引（用前200字符快速匹配）
    existing_texts = set(merged_v2_all["text"].str[:200].tolist())
    print(f"\n已有文本指纹: {len(existing_texts)}")

    # 2. 加载新数据源
    print(f"\n{'='*60}")
    print("  加载新数据源")
    print(f"{'='*60}")

    new_dfs = []

    # 2a. AI 生成未使用部分
    df_ai = load_generated_ai_unused(existing_texts)
    if len(df_ai) > 0:
        new_dfs.append(df_ai)

    # 2b. LCSTS 人类新闻
    df_formal = load_human_formal_samples()
    if len(df_formal) > 0:
        new_dfs.append(df_formal)

    # 2c. template 人类文本
    df_template = load_human_supplement_templates()
    if len(df_template) > 0:
        new_dfs.append(df_template)

    # 2d. defense patch
    df_defense = load_defense_patch()
    if len(df_defense) > 0:
        new_dfs.append(df_defense)

    if not new_dfs:
        print("\n[ERROR] 没有找到新数据，退出")
        return

    # 3. 合并新数据
    new_data = pd.concat(new_dfs, ignore_index=True)
    print(f"\n新数据合计: {len(new_data)} 条")

    # 4. 与 merged_v2 已有文本去重
    new_data_keys = new_data["text"].str[:200]
    dup_mask = new_data_keys.isin(existing_texts)
    if dup_mask.sum() > 0:
        print(f"  与 merged_v2 重复: {dup_mask.sum()} 条，已移除")
        new_data = new_data[~dup_mask]

    # 5. 新数据内部去重
    new_data = deduplicate(new_data)

    print_summary(new_data, "新增数据")

    # 6. 新数据划分 train/val (8:2)
    new_train, new_val = split_train_val(new_data, val_ratio=0.2)
    print(f"\n新数据划分: train={len(new_train)}, val={len(new_val)}")

    # 7. 与 merged_v2 合并
    final_train = pd.concat([train_v2, new_train], ignore_index=True)
    final_val = pd.concat([val_v2, new_val], ignore_index=True)

    # 确保只保留 text, label, source 三列
    for col in ["text", "label", "source"]:
        if col not in final_train.columns:
            if col == "source":
                final_train["source"] = "unknown"
                final_val["source"] = "unknown"

    final_train = final_train[["text", "label", "source"]]
    final_val = final_val[["text", "label", "source"]]

    # 最终去重
    final_train = deduplicate(final_train)
    final_val = deduplicate(final_val)

    print_summary(final_train, "最终训练集 (merged_v3)")
    print_summary(final_val, "最终验证集 (merged_v3)")

    # 8. 写入
    if args.dry_run:
        print("\n[DRY RUN] 不写入文件")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"

    final_train.to_csv(train_path, index=False, encoding="utf-8-sig")
    final_val.to_csv(val_path, index=False, encoding="utf-8-sig")

    # 保存合并日志
    log = {
        "base": "merged_v2",
        "new_sources": {
            "generated_ai_unused": len(df_ai) if len(df_ai) > 0 else 0,
            "human_formal_samples": len(df_formal) if len(df_formal) > 0 else 0,
            "human_supplement_templates": len(df_template) if len(df_template) > 0 else 0,
            "defense_patch_v2": len(df_defense) if len(df_defense) > 0 else 0,
        },
        "final_train": len(final_train),
        "final_val": len(final_val),
        "total": len(final_train) + len(final_val),
    }
    with open(output_dir / "merge_log.json", "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 已保存到 {output_dir}")
    print(f"  train.csv: {len(final_train)} 条")
    print(f"  val.csv: {len(final_val)} 条")
    print(f"  merge_log.json: 合并日志")


if __name__ == "__main__":
    main()
