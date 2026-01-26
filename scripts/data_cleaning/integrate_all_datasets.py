#!/usr/bin/env python3
"""
数据整合脚本 - 整合所有高质量数据源
"""

import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import hashlib

def load_datasets():
    """加载所有数据源"""
    datasets = []
    
    # 1. 当前去偏数据集
    try:
        train_df = pd.read_csv('datasets/bert_debiased/train.csv')
        val_df = pd.read_csv('datasets/bert_debiased/val.csv')
        test_df = pd.read_csv('datasets/bert_debiased/test.csv')
        current_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        current_df['source'] = 'bert_debiased'
        datasets.append(current_df)
        print(f"✓ 加载去偏数据集: {len(current_df)} 条")
    except:
        print("✗ 去偏数据集未找到")
    
    # 2. 原始AI数据
    try:
        ai_df = pd.read_csv('datasets/final/ai_generated_texts.csv')
        ai_df['label'] = 1
        ai_df['source'] = 'ai_generated'
        datasets.append(ai_df)
        print(f"✓ 加载AI数据: {len(ai_df)} 条")
    except:
        print("✗ AI数据未找到")
    
    # 3. 人类数据
    try:
        human_df = pd.read_csv('datasets/human_texts/human_texts.csv')
        human_df['label'] = 0
        human_df['source'] = 'human_texts'
        datasets.append(human_df)
        print(f"✓ 加载人类数据: {len(human_df)} 条")
    except:
        print("✗ 人类数据未找到")
    
    # 4. 标准数据集
    for dataset_name in ['hc3', 'thucnews', 'mgtbench']:
        try:
            std_df = pd.read_csv(f'datasets/standard/{dataset_name}_processed.csv')
            std_df['source'] = dataset_name
            datasets.append(std_df)
            print(f"✓ 加载{dataset_name}: {len(std_df)} 条")
        except:
            print(f"✗ {dataset_name}数据未找到")
    
    return datasets

def standardize_format(datasets):
    """统一数据格式"""
    standardized = []
    
    for df in datasets:
        # 确保必要列存在
        if 'text' not in df.columns:
            continue
        
        # 标准化列名
        std_df = pd.DataFrame()
        std_df['text'] = df['text']
        std_df['label'] = df.get('label', 0)
        std_df['source'] = df.get('source', 'unknown')
        
        # 计算长度
        std_df['length'] = std_df['text'].str.len()
        
        # 添加文本哈希用于去重
        std_df['text_hash'] = std_df['text'].apply(
            lambda x: hashlib.md5(str(x).encode()).hexdigest()
        )
        
        standardized.append(std_df)
    
    return pd.concat(standardized, ignore_index=True)

def remove_duplicates(df):
    """去重"""
    print(f"去重前: {len(df)} 条")
    df_dedup = df.drop_duplicates(subset=['text_hash'], keep='first')
    print(f"去重后: {len(df_dedup)} 条 (移除 {len(df) - len(df_dedup)} 条)")
    return df_dedup.drop('text_hash', axis=1)

def filter_quality(df):
    """过滤低质量数据"""
    print(f"质量过滤前: {len(df)} 条")
    
    # 过滤条件
    mask = (
        (df['length'] >= 50) &  # 最少50字符
        (df['length'] <= 5000) &  # 最多5000字符
        (df['text'].str.strip() != '') &  # 非空
        (~df['text'].str.contains(r'^[\s\n\r]*$', regex=True))  # 非纯空白
    )
    
    df_filtered = df[mask].copy()
    print(f"质量过滤后: {len(df_filtered)} 条 (移除 {len(df) - len(df_filtered)} 条)")
    return df_filtered

def balance_length(df, target_samples=20000):
    """长度平衡 - 分层采样"""
    print(f"长度平衡前: {len(df)} 条")
    
    # 定义长度区间
    df['length_bin'] = pd.cut(df['length'], 
                             bins=[0, 200, 500, 1000, 2000, float('inf')], 
                             labels=['very_short', 'short', 'medium', 'long', 'very_long'])
    
    balanced_dfs = []
    
    for label in [0, 1]:
        label_df = df[df['label'] == label].copy()
        
        # 每个标签目标样本数
        label_target = target_samples // 2
        
        # 按长度区间分层采样
        sampled_parts = []
        for bin_name in ['very_short', 'short', 'medium', 'long', 'very_long']:
            bin_df = label_df[label_df['length_bin'] == bin_name]
            if len(bin_df) > 0:
                # 每个区间采样相同比例
                bin_target = min(len(bin_df), label_target // 5)
                if len(bin_df) > bin_target:
                    bin_sampled = bin_df.sample(n=bin_target, random_state=42)
                else:
                    bin_sampled = bin_df
                sampled_parts.append(bin_sampled)
        
        if sampled_parts:
            label_balanced = pd.concat(sampled_parts, ignore_index=True)
            balanced_dfs.append(label_balanced)
    
    result = pd.concat(balanced_dfs, ignore_index=True).drop('length_bin', axis=1)
    print(f"长度平衡后: {len(result)} 条")
    
    # 打印长度分布
    print("\n长度分布:")
    for label in [0, 1]:
        label_data = result[result['label'] == label]
        print(f"标签{label}: 平均长度 {label_data['length'].mean():.0f}, "
              f"中位数 {label_data['length'].median():.0f}")
    
    return result

def split_dataset(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """划分数据集"""
    # 分层划分保持标签平衡
    train_df, temp_df = train_test_split(
        df, test_size=(val_ratio + test_ratio), 
        stratify=df['label'], random_state=42
    )
    
    val_df, test_df = train_test_split(
        temp_df, test_size=test_ratio/(val_ratio + test_ratio),
        stratify=temp_df['label'], random_state=42
    )
    
    print(f"\n数据集划分:")
    print(f"训练集: {len(train_df)} 条 ({len(train_df)/len(df)*100:.1f}%)")
    print(f"验证集: {len(val_df)} 条 ({len(val_df)/len(df)*100:.1f}%)")
    print(f"测试集: {len(test_df)} 条 ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df

def save_datasets(train_df, val_df, test_df, output_dir):
    """保存数据集"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存分割数据
    train_df.to_csv(output_path / 'train.csv', index=False)
    val_df.to_csv(output_path / 'val.csv', index=False)
    test_df.to_csv(output_path / 'test.csv', index=False)
    
    # 保存完整数据
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    full_df.to_csv(output_path / 'full_dataset.csv', index=False)
    
    # 保存统计信息
    stats = {
        'total_samples': len(full_df),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'ai_samples': len(full_df[full_df['label'] == 1]),
        'human_samples': len(full_df[full_df['label'] == 0]),
        'sources': full_df['source'].value_counts().to_dict(),
        'avg_length_ai': full_df[full_df['label'] == 1]['length'].mean(),
        'avg_length_human': full_df[full_df['label'] == 0]['length'].mean()
    }
    
    with open(output_path / 'dataset_stats.txt', 'w', encoding='utf-8') as f:
        f.write("数据集统计信息\n")
        f.write("=" * 50 + "\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\n✓ 数据集已保存到: {output_path}")
    print(f"✓ 统计信息已保存到: {output_path / 'dataset_stats.txt'}")

def main():
    parser = argparse.ArgumentParser(description='整合所有数据集')
    parser.add_argument('--output', '-o', default='datasets/integrated',
                       help='输出目录 (默认: datasets/integrated)')
    parser.add_argument('--samples', '-s', type=int, default=20000,
                       help='目标样本数 (默认: 20000)')
    
    args = parser.parse_args()
    
    print("开始数据整合...")
    
    # 1. 加载数据
    datasets = load_datasets()
    if not datasets:
        print("❌ 未找到任何数据集")
        return
    
    # 2. 统一格式
    df = standardize_format(datasets)
    print(f"\n统一格式后: {len(df)} 条")
    
    # 3. 去重
    df = remove_duplicates(df)
    
    # 4. 质量过滤
    df = filter_quality(df)
    
    # 5. 长度平衡
    df = balance_length(df, args.samples)
    
    # 6. 划分数据集
    train_df, val_df, test_df = split_dataset(df)
    
    # 7. 保存
    save_datasets(train_df, val_df, test_df, args.output)
    
    print("\n🎉 数据整合完成!")

if __name__ == '__main__':
    main()