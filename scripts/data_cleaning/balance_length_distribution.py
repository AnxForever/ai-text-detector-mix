#!/usr/bin/env python3
"""
长度分布平衡
解决AI文本(1680)和人类文本(942)的长度偏差问题
"""

import pandas as pd
import numpy as np
from pathlib import Path

def balance_length(input_file, output_file, target_avg=1200, max_length=2500):
    """
    平衡长度分布
    
    策略：
    1. 截断过长的AI文本（>2500字符）
    2. 过滤过短的人类文本（<300字符）
    3. 使目标平均长度接近1200
    """
    print(f"读取数据: {input_file}")
    df = pd.read_csv(input_file, encoding='utf-8-sig')
    
    print(f"\n原始数据:")
    print(f"  总数: {len(df)}")
    print(f"  AI: {(df['label']==1).sum()}, 平均长度: {df[df['label']==1]['length'].mean():.0f}")
    print(f"  人类: {(df['label']==0).sum()}, 平均长度: {df[df['label']==0]['length'].mean():.0f}")
    
    # 分离AI和人类
    df_ai = df[df['label'] == 1].copy()
    df_human = df[df['label'] == 0].copy()
    
    # 1. 截断过长AI文本
    long_ai = df_ai['length'] > max_length
    print(f"\n截断 {long_ai.sum()} 条过长AI文本 (>{max_length})")
    
    df_ai.loc[long_ai, 'text'] = df_ai.loc[long_ai, 'text'].str[:max_length]
    df_ai.loc[long_ai, 'length'] = max_length
    
    # 2. 过滤过短人类文本
    min_length = 300
    df_human = df_human[df_human['length'] >= min_length]
    print(f"过滤过短人类文本 (<{min_length}), 剩余: {len(df_human)}")
    
    # 3. 按长度分层采样，使分布更接近
    def stratified_sample(df, target_count, target_avg_length=1200):
        """按长度分层采样，目标平均长度"""
        bins = [0, 500, 1000, 1500, 2000, 2500, 10000]
        df['length_bin'] = pd.cut(df['length'], bins=bins)
        
        # 优先采样中等长度区间
        samples = []
        
        # 为每个区间设置权重（优先中等长度）
        weights = {
            pd.Interval(0, 500, closed='right'): 0.1,
            pd.Interval(500, 1000, closed='right'): 0.25,
            pd.Interval(1000, 1500, closed='right'): 0.3,
            pd.Interval(1500, 2000, closed='right'): 0.25,
            pd.Interval(2000, 2500, closed='right'): 0.08,
            pd.Interval(2500, 10000, closed='right'): 0.02
        }
        
        for bin_interval, weight in weights.items():
            bin_df = df[df['length_bin'] == bin_interval]
            if len(bin_df) > 0:
                n_sample = int(target_count * weight)
                n_sample = min(n_sample, len(bin_df))
                if n_sample > 0:
                    samples.append(bin_df.sample(n=n_sample, random_state=42))
        
        if not samples:
            # 如果没有采样到，直接返回原数据
            df = df.drop('length_bin', axis=1)
            return df.sample(n=min(target_count, len(df)), random_state=42)
        
        result = pd.concat(samples)
        result = result.drop('length_bin', axis=1)
        
        # 如果不够，从中等长度区间补充
        if len(result) < target_count:
            remaining_df = df[~df.index.isin(result.index)]
            remaining_df = remaining_df.drop('length_bin', axis=1)
            mid_length = remaining_df[(remaining_df['length'] >= 800) & (remaining_df['length'] <= 1800)]
            if len(mid_length) > 0:
                extra = mid_length.sample(n=min(target_count - len(result), len(mid_length)), random_state=42)
                result = pd.concat([result, extra])
        
        return result.sample(n=min(target_count, len(result)), random_state=42)
    
    # 采样到目标数量
    target_count = min(len(df_ai), len(df_human))
    
    df_ai_balanced = stratified_sample(df_ai, target_count)
    df_human_balanced = stratified_sample(df_human, target_count)
    
    # 合并
    df_balanced = pd.concat([df_ai_balanced, df_human_balanced], ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 保存
    df_balanced.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n平衡后数据:")
    print(f"  总数: {len(df_balanced)}")
    print(f"  AI: {(df_balanced['label']==1).sum()}, 平均长度: {df_balanced[df_balanced['label']==1]['length'].mean():.0f}")
    print(f"  人类: {(df_balanced['label']==0).sum()}, 平均长度: {df_balanced[df_balanced['label']==0]['length'].mean():.0f}")
    
    # 长度分布对比
    print(f"\n长度分布:")
    bins = [0, 500, 1000, 1500, 2000, 2500, 10000]
    for i in range(len(bins)-1):
        ai_count = ((df_balanced['label']==1) & (df_balanced['length']>=bins[i]) & (df_balanced['length']<bins[i+1])).sum()
        human_count = ((df_balanced['label']==0) & (df_balanced['length']>=bins[i]) & (df_balanced['length']<bins[i+1])).sum()
        print(f"  {bins[i]}-{bins[i+1]}: AI={ai_count}, 人类={human_count}")
    
    print(f"\n✓ 已保存到: {output_file}")
    
    return df_balanced


if __name__ == "__main__":
    balance_length(
        input_file="datasets/bert_v2/full_dataset_labeled.csv",
        output_file="datasets/bert_v2/full_dataset_length_balanced.csv",
        max_length=2500
    )
