#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集审计脚本 - 分析数据集结构和问题
"""

import pandas as pd
import os
import json
from collections import Counter

def analyze_dataset(path, name):
    """分析单个数据集"""
    print(f"\n{'='*60}")
    print(f"📊 {name}")
    print(f"{'='*60}")
    
    if not os.path.exists(path):
        print(f"❌ 文件不存在: {path}")
        return None
    
    # 读取数据
    if path.endswith('.csv'):
        df = pd.read_csv(path, encoding='utf-8-sig')
    elif path.endswith('.jsonl'):
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        df = pd.DataFrame(data)
    else:
        print(f"❌ 不支持的格式: {path}")
        return None
    
    print(f"📁 文件: {path}")
    print(f"📏 行数: {len(df):,}")
    print(f"📋 列名: {list(df.columns)}")
    
    # 标签分布
    if 'label' in df.columns:
        print(f"\n🏷️ 标签分布:")
        label_counts = df['label'].value_counts()
        for label, count in label_counts.items():
            pct = count / len(df) * 100
            label_name = 'Human' if label == 0 else 'AI'
            print(f"   Label {label} ({label_name}): {count:,} ({pct:.1f}%)")
        
        # 平衡度
        if len(label_counts) == 2:
            ratio = label_counts.min() / label_counts.max()
            balance_status = "✅ 平衡" if ratio > 0.8 else "⚠️ 不平衡" if ratio > 0.5 else "❌ 严重不平衡"
            print(f"   平衡比: {ratio:.2f} {balance_status}")
    
    # 来源分布
    if 'source' in df.columns:
        print(f"\n📦 来源分布 (Top 10):")
        source_counts = df['source'].value_counts().head(10)
        for source, count in source_counts.items():
            pct = count / len(df) * 100
            print(f"   {source}: {count:,} ({pct:.1f}%)")
    
    # 类别分布
    if 'category' in df.columns:
        print(f"\n📂 类别分布:")
        cat_counts = df['category'].value_counts()
        for cat, count in cat_counts.items():
            pct = count / len(df) * 100
            print(f"   {cat}: {count:,} ({pct:.1f}%)")
    
    # 文本长度分析
    if 'text' in df.columns:
        df['text_len'] = df['text'].astype(str).str.len()
        print(f"\n📐 文本长度统计:")
        print(f"   平均: {df['text_len'].mean():.1f} 字符")
        print(f"   中位数: {df['text_len'].median():.1f} 字符")
        print(f"   最小: {df['text_len'].min()} 字符")
        print(f"   最大: {df['text_len'].max()} 字符")
        print(f"   标准差: {df['text_len'].std():.1f}")
        
        # 按标签的长度分布
        if 'label' in df.columns:
            print(f"\n📐 按标签的平均长度:")
            for label in sorted(df['label'].unique()):
                avg_len = df[df['label']==label]['text_len'].mean()
                label_name = 'Human' if label == 0 else 'AI'
                print(f"   Label {label} ({label_name}): {avg_len:.1f} 字符")
            
            # 长度差异
            human_len = df[df['label']==0]['text_len'].mean()
            ai_len = df[df['label']==1]['text_len'].mean()
            diff = abs(human_len - ai_len)
            diff_pct = diff / min(human_len, ai_len) * 100
            if diff_pct > 20:
                print(f"   ⚠️ 长度差异: {diff:.1f} 字符 ({diff_pct:.1f}%) - 可能导致长度偏差学习")
            else:
                print(f"   ✅ 长度差异: {diff:.1f} 字符 ({diff_pct:.1f}%) - 相对平衡")
        
        # 长度分桶
        print(f"\n📊 长度分桶分布:")
        bins = [0, 100, 200, 500, 1000, 2000, float('inf')]
        labels_bin = ['<100', '100-200', '200-500', '500-1000', '1000-2000', '>2000']
        df['len_bucket'] = pd.cut(df['text_len'], bins=bins, labels=labels_bin)
        bucket_counts = df['len_bucket'].value_counts().sort_index()
        for bucket, count in bucket_counts.items():
            pct = count / len(df) * 100
            print(f"   {bucket}: {count:,} ({pct:.1f}%)")
    
    return df


def main():
    print("🔍 数据集审计报告")
    print("=" * 60)
    
    # 1. Core V1 训练集
    analyze_dataset('datasets/active/core_v1/train.csv', 'Core V1 训练集')
    
    # 2. Core V1 测试集
    analyze_dataset('datasets/active/core_v1/test.csv', 'Core V1 测试集')
    
    # 3. Core V2 训练集
    analyze_dataset('datasets/active/core_v2/train.csv', 'Core V2 训练集')
    
    # 4. Core V3 训练集
    analyze_dataset('datasets/active/core_v3/train.csv', 'Core V3 训练集')
    
    # 5. 检查bert_v2_overnight数据集
    analyze_dataset('datasets/bert_v2_overnight/train.csv', 'BERT V2 Overnight 训练集')
    analyze_dataset('datasets/bert_v2_overnight/train_balanced.csv', 'BERT V2 Balanced 训练集')
    
    # 6. 外部数据集概览
    print(f"\n{'='*60}")
    print("📦 外部数据集概览")
    print("=" * 60)
    
    external_dirs = [
        'datasets/external/HC3-Chinese',
        'datasets/external/M4',
        'datasets/external/THUCNews',
        'datasets/external/LCSTS',
    ]
    
    for ext_dir in external_dirs:
        if os.path.exists(ext_dir):
            files = os.listdir(ext_dir)
            total_size = sum(os.path.getsize(os.path.join(ext_dir, f)) 
                           for f in files if os.path.isfile(os.path.join(ext_dir, f)))
            print(f"   {ext_dir}: {len(files)} 文件, {total_size/1024/1024:.1f} MB")
    
    print(f"\n{'='*60}")
    print("✅ 审计完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
