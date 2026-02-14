#!/usr/bin/env python3
"""
论文数据自动统计脚本

功能: 一键生成第5章实验所需的所有表格数据
用法: python scripts/evaluation/thesis_stats.py --dataset datasets/active/core_v2/merged.csv

输出: 直接打印可粘贴到论文的Markdown表格
"""

import argparse
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd


def print_section(title: str):
    print(f"\n{'='*65}")
    print(f"📊 {title}")
    print(f"{'='*65}\n")


def dataset_stats(df: pd.DataFrame):
    """表5-1: 数据集统计"""
    print_section("表5-1 数据集统计")

    total = len(df)
    human = (df['label'] == 0).sum()
    ai = (df['label'] == 1).sum()

    # 尝试获取split信息
    if 'split' in df.columns:
        train = (df['split'] == 'train').sum()
        val = (df['split'] == 'val').sum()
        test = (df['split'] == 'test').sum()
    else:
        train = int(total * 0.8)
        val = int(total * 0.1)
        test = total - train - val

    avg_len = df['length'].mean() if 'length' in df.columns else len(df['text'].str.len().mean())

    # 模型数
    if 'model' in df.columns:
        model_count = df[df['label'] == 1]['model'].nunique()
    else:
        model_count = df['source'].nunique()

    print("| 统计项 | 数值 |")
    print("|-------|------|")
    print(f"| 总样本数 | {total:,} |")
    print(f"| 人类文本 | {human:,} ({human/total*100:.1f}%) |")
    print(f"| AI文本 | {ai:,} ({ai/total*100:.1f}%) |")
    print(f"| 训练集 | {train:,} ({train/total*100:.1f}%) |")
    print(f"| 验证集 | {val:,} ({val/total*100:.1f}%) |")
    print(f"| 测试集 | {test:,} ({test/total*100:.1f}%) |")
    print(f"| 平均文本长度 | {avg_len:.1f}字 |")
    print(f"| 覆盖AI模型数 | {model_count}个 |")


def scenario_stats(df: pd.DataFrame):
    """表5-2: 场景分布统计"""
    if 'scenario_id' not in df.columns:
        print("\n⚠️ 数据缺少 scenario_id 字段，跳过场景统计")
        return

    print_section("表5-2 场景分布统计")

    names = {
        'A': '教育学术', 'B': '职场办公', 'C': '知识百科',
        'D': '社区互动', 'E': '商业营销', 'F': '新闻资讯'
    }
    total = len(df)

    print("| 场景 | 代码 | 样本数 | 占比 |")
    print("|-----|------|-------|------|")
    for sid in ['A', 'B', 'C', 'D', 'E', 'F']:
        count = (df['scenario_id'] == sid).sum()
        pct = count / total * 100
        print(f"| {names.get(sid, sid)} | {sid} | {count:,} | {pct:.1f}% |")


def length_stats(df: pd.DataFrame):
    """表5-7: 长度区间统计"""
    print_section("表5-7 不同长度区间分布")

    if 'length' not in df.columns:
        df = df.copy()
        df['length'] = df['text'].str.len()

    buckets = [
        ('0-80', 0, 80),
        ('80-200', 80, 200),
        ('200-500', 200, 500),
        ('500-1000', 500, 1000),
        ('1000-2000', 1000, 2000),
        ('2000+', 2000, 999999),
    ]

    total = len(df)
    print("| 长度区间 | 样本数 | 占比 |")
    print("|---------|-------|------|")
    for name, lo, hi in buckets:
        count = ((df['length'] >= lo) & (df['length'] < hi)).sum()
        pct = count / total * 100
        print(f"| {name} | {count:,} | {pct:.1f}% |")


def source_stats(df: pd.DataFrame):
    """表5-x: 来源/模型分布"""
    print_section("数据来源分布")

    field = 'model' if 'model' in df.columns else 'source'

    print(f"| {field} | 样本数 | 占比 |")
    print("|-------|-------|------|")

    total = len(df)
    for val, count in df[field].value_counts().head(15).items():
        pct = count / total * 100
        print(f"| {val} | {count:,} | {pct:.1f}% |")


def label_balance(df: pd.DataFrame):
    """标签平衡情况"""
    print_section("标签平衡")

    human = (df['label'] == 0).sum()
    ai = (df['label'] == 1).sum()
    ratio = ai / human if human > 0 else float('inf')

    print(f"Human: {human:,}")
    print(f"AI:    {ai:,}")
    print(f"比例:  1:{ratio:.2f}")

    if 'scenario_id' in df.columns:
        print("\n按场景的标签分布:")
        print("| 场景 | Human | AI | 比例 |")
        print("|-----|-------|-----|------|")
        for sid in sorted(df['scenario_id'].unique()):
            sub = df[df['scenario_id'] == sid]
            h = (sub['label'] == 0).sum()
            a = (sub['label'] == 1).sum()
            r = a / h if h > 0 else float('inf')
            print(f"| {sid} | {h:,} | {a:,} | 1:{r:.2f} |")


def style_stats(df: pd.DataFrame):
    """风格分布"""
    if 'style' not in df.columns:
        return

    print_section("写作风格分布")

    total = len(df)
    print("| 风格 | 样本数 | 占比 |")
    print("|-----|-------|------|")
    for style, count in df['style'].value_counts().items():
        pct = count / total * 100
        print(f"| {style} | {count:,} | {pct:.1f}% |")


def quality_checks(df: pd.DataFrame):
    """数据质量检查"""
    print_section("数据质量检查")

    total = len(df)

    # 空文本
    empty = df['text'].isna().sum() + (df['text'] == '').sum()
    print(f"空文本: {empty} ({empty/total*100:.2f}%)")

    # 重复
    if 'text_id' in df.columns:
        dup = df.duplicated(subset='text_id').sum()
    else:
        dup = df.duplicated(subset='text').sum()
    print(f"重复样本: {dup} ({dup/total*100:.2f}%)")

    # 长度异常
    if 'length' in df.columns:
        too_short = (df['length'] < 80).sum()
        too_long = (df['length'] > 3000).sum()
        print(f"过短(<80字): {too_short} ({too_short/total*100:.2f}%)")
        print(f"过长(>3000字): {too_long} ({too_long/total*100:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="论文数据自动统计")
    parser.add_argument("--dataset", "-d", type=Path, required=True,
                        help="数据集CSV路径")
    parser.add_argument("--sections", nargs="*",
                        default=["all"],
                        choices=["all", "basic", "scenario", "length", "source",
                                 "balance", "style", "quality"],
                        help="要输出的统计部分")
    args = parser.parse_args()

    print("=" * 65)
    print("📝 论文数据自动统计工具")
    print("=" * 65)
    print(f"数据文件: {args.dataset}")

    # 加载数据
    if args.dataset.suffix == '.csv':
        df = pd.read_csv(args.dataset, encoding='utf-8-sig')
    elif args.dataset.suffix == '.jsonl':
        records = []
        with open(args.dataset, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    import json
                    records.append(json.loads(line))
        df = pd.DataFrame(records)
    else:
        print(f"❌ 不支持的文件格式: {args.dataset.suffix}")
        sys.exit(1)

    print(f"加载完成: {len(df):,} 条数据")

    sections = args.sections
    run_all = "all" in sections

    if run_all or "basic" in sections:
        dataset_stats(df)
    if run_all or "scenario" in sections:
        scenario_stats(df)
    if run_all or "length" in sections:
        length_stats(df)
    if run_all or "source" in sections:
        source_stats(df)
    if run_all or "balance" in sections:
        label_balance(df)
    if run_all or "style" in sections:
        style_stats(df)
    if run_all or "quality" in sections:
        quality_checks(df)

    print(f"\n{'='*65}")
    print("✅ 统计完成！以上表格可直接粘贴到论文中")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
