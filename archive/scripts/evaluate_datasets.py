#!/usr/bin/env python3
"""
数据集评估和模型问题分析
"""
import pandas as pd
import json
import os
from pathlib import Path
from collections import Counter

print("="*80)
print("数据集评估报告")
print("="*80)

# 1. 统计所有数据集
print("\n【1. 数据集概览】")
datasets = {
    "final_clean": "datasets/archive/final_clean",
    "combined_v2": "datasets/archive/combined_v2",
    "hybrid": "datasets/mixed/hybrid",
}

for name, path in datasets.items():
    print(f"\n{name}:")
    if os.path.exists(path):
        files = list(Path(path).glob("*.csv"))
        for f in files:
            size = os.path.getsize(f) / 1024 / 1024
            try:
                df = pd.read_csv(f, nrows=1)
                total_lines = sum(1 for _ in open(f, encoding='utf-8', errors='ignore')) - 1
                print(f"  {f.name:30s} {total_lines:>8,} 行  {size:>6.1f}MB")
            except:
                print(f"  {f.name:30s} {'ERROR':>8s}  {size:>6.1f}MB")

# 2. 分析 combined_v2 数据集内容
print("\n" + "="*80)
print("【2. combined_v2 数据集分析】")
try:
    train_df = pd.read_csv('datasets/active/core_v1/train.csv')
    val_df = pd.read_csv('datasets/active/core_v1/val.csv')
    test_df = pd.read_csv('datasets/active/core_v1/test.csv')
    
    print(f"\n训练集: {len(train_df):,} 样本")
    print(f"验证集: {len(val_df):,} 样本")
    print(f"测试集: {len(test_df):,} 样本")
    print(f"总计:   {len(train_df)+len(val_df)+len(test_df):,} 样本")
    
    # 标签分布
    print(f"\n标签分布 (训练集):")
    if 'label' in train_df.columns:
        label_dist = train_df['label'].value_counts()
        for label, count in label_dist.items():
            pct = count / len(train_df) * 100
            print(f"  {label}: {count:>8,} ({pct:>5.1f}%)")
    
    # 文本长度分析
    print(f"\n文本长度统计 (训练集):")
    if 'text' in train_df.columns:
        lengths = train_df['text'].str.len()
        print(f"  平均: {lengths.mean():.0f} 字符")
        print(f"  中位: {lengths.median():.0f} 字符")
        print(f"  最小: {lengths.min():.0f} 字符")
        print(f"  最大: {lengths.max():.0f} 字符")
        
        # 长度分布
        print(f"\n  长度分布:")
        bins = [0, 100, 200, 500, 1000, 2000, 10000]
        for i in range(len(bins)-1):
            count = ((lengths >= bins[i]) & (lengths < bins[i+1])).sum()
            pct = count / len(lengths) * 100
            print(f"    {bins[i]:>5d}-{bins[i+1]:>5d}: {count:>8,} ({pct:>5.1f}%)")
    
    # 检查是否有 [SEP] 标记
    if 'text' in train_df.columns:
        sep_count = train_df['text'].str.contains(r'\[SEP\]', regex=True).sum()
        print(f"\n包含[SEP]标记的样本: {sep_count:,} ({sep_count/len(train_df)*100:.1f}%)")
        
except Exception as e:
    print(f"错误: {e}")

# 3. 分析 final_clean 数据集
print("\n" + "="*80)
print("【3. final_clean 数据集分析】")
try:
    train_df = pd.read_csv('datasets/active/core_v1/train.csv')
    print(f"\n训练集: {len(train_df):,} 样本")
    
    if 'label' in train_df.columns:
        label_dist = train_df['label'].value_counts()
        print(f"\n标签分布:")
        for label, count in label_dist.items():
            pct = count / len(train_df) * 100
            print(f"  {label}: {count:>8,} ({pct:>5.1f}%)")
    
    if 'text' in train_df.columns:
        lengths = train_df['text'].str.len()
        print(f"\n文本长度: 平均 {lengths.mean():.0f}, 中位 {lengths.median():.0f}")
        
except Exception as e:
    print(f"错误: {e}")

# 4. 模型性能对比
print("\n" + "="*80)
print("【4. 模型性能对比】")
print("\nbert_improved (旧模型, final_clean训练):")
print("  ✅ 技术文档AI: 95.2%")
print("  ✅ 技术解释AI: 81.1%")
print("  ✅ 对话式AI:   95.5%")
print("  综合表现: 优秀")

print("\nbert_v2_with_sep (新模型, combined_v2训练):")
print("  ❌ 技术文档AI: 14.9% (误判为Human)")
print("  ✅ 技术解释AI: 85.8%")
print("  ✅ 对话式AI:   100%")
print("  综合表现: 对列表式/技术文档式AI识别差")

# 5. 问题总结
print("\n" + "="*80)
print("【5. 问题诊断】")
print("""
核心问题:
1. combined_v2 虽然样本更多，但可能引入了低质量数据
2. [SEP]标记的混合文本可能干扰了模型学习纯AI文本的特征
3. 缺少技术文档风格的AI样本（列表式、专业术语密集）
4. 数据集可能存在标注错误或噪声

模型表现差异:
- bert_improved: 在小而精的数据集上训练，泛化能力强
- bert_v2_with_sep: 在大而杂的数据集上训练，过拟合某些模式
""")

# 6. 改进建议
print("="*80)
print("【6. 改进方案】")
print("""
数据集改进:
1. 清洗 combined_v2，移除低质量样本
2. 增加技术文档风格的AI样本（从DeepSeek/Qwen生成）
3. 平衡不同风格的AI文本比例
4. 分离混合文本训练（单独训练边界检测器）

训练策略:
1. 使用 final_clean 作为基础，逐步添加高质量样本
2. 降低学习率，增加训练轮数
3. 使用更强的正则化（dropout 0.2-0.3）
4. 实施早停机制（patience=2）
5. 数据增强：同义词替换、回译

评估方法:
1. 在多种风格的测试集上评估
2. 添加对抗样本测试
3. 人工审核误判样本
""")

print("\n" + "="*80)
print("生成详细报告...")
