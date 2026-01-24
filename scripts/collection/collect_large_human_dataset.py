#!/usr/bin/env python3
"""
大规模人类文本数据收集工具
目标：收集5万条高质量中文人类文本
"""

import os
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict

class LargeScaleCollector:
    def __init__(self, output_dir="datasets/human_large"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_length = 300
        self.max_length = 3000
        
    def collect_thucnews(self, target=20000):
        """THUCNews新闻数据集（74万篇）"""
        print(f"\n[1/5] THUCNews - 目标: {target}条")
        try:
            from datasets import load_dataset
            ds = load_dataset("oyxy2019/THUCNewsText", split="train")
            df = pd.DataFrame(ds)
            df['length'] = df['text'].str.len()
            df = df[(df['length'] >= self.min_length) & (df['length'] <= self.max_length)]
            df = df.sample(n=min(target, len(df)), random_state=42)
            df['source'] = 'thucnews'
            self._save(df, 'thucnews')
            return df
        except Exception as e:
            print(f"✗ 错误: {e}")
            return pd.DataFrame()
    
    def collect_wikipedia(self, target=10000):
        """Wikipedia中文百科"""
        print(f"\n[2/5] Wikipedia - 目标: {target}条")
        try:
            from datasets import load_dataset
            ds = load_dataset("wikipedia", "20220301.zh", split="train")
            df = pd.DataFrame(ds)
            df['length'] = df['text'].str.len()
            df = df[(df['length'] >= self.min_length) & (df['length'] <= self.max_length)]
            df = df.sample(n=min(target, len(df)), random_state=42)
            df['source'] = 'wikipedia'
            self._save(df, 'wikipedia')
            return df
        except Exception as e:
            print(f"✗ 错误: {e}")
            return pd.DataFrame()
    
    def collect_clue(self, target=10000):
        """CLUE中文语料"""
        print(f"\n[3/5] CLUE - 目标: {target}条")
        try:
            from datasets import load_dataset
            # CLUE包含多个子任务，这里用TNEWS（今日头条新闻）
            ds = load_dataset("clue", "tnews", split="train")
            df = pd.DataFrame(ds)
            if 'sentence' in df.columns:
                df = df.rename(columns={'sentence': 'text'})
            df['length'] = df['text'].str.len()
            df = df[(df['length'] >= self.min_length) & (df['length'] <= self.max_length)]
            df = df.sample(n=min(target, len(df)), random_state=42)
            df['source'] = 'clue'
            self._save(df, 'clue')
            return df
        except Exception as e:
            print(f"✗ 错误: {e}")
            return pd.DataFrame()
    
    def collect_weibo(self, target=5000):
        """微博文本（如果可用）"""
        print(f"\n[4/5] Weibo - 目标: {target}条")
        try:
            from datasets import load_dataset
            ds = load_dataset("dirtycomputer/weibo_senti_100k", split="train")
            df = pd.DataFrame(ds)
            if 'review' in df.columns:
                df = df.rename(columns={'review': 'text'})
            df['length'] = df['text'].str.len()
            df = df[(df['length'] >= self.min_length) & (df['length'] <= self.max_length)]
            df = df.sample(n=min(target, len(df)), random_state=42)
            df['source'] = 'weibo'
            self._save(df, 'weibo')
            return df
        except Exception as e:
            print(f"✗ 错误: {e}")
            return pd.DataFrame()
    
    def collect_csl(self, target=5000):
        """中文科学文献（CSL）"""
        print(f"\n[5/5] CSL - 目标: {target}条")
        try:
            from datasets import load_dataset
            ds = load_dataset("neuclir/csl", split="train")
            df = pd.DataFrame(ds)
            if 'abst' in df.columns:
                df = df.rename(columns={'abst': 'text'})
            df['length'] = df['text'].str.len()
            df = df[(df['length'] >= self.min_length) & (df['length'] <= self.max_length)]
            df = df.sample(n=min(target, len(df)), random_state=42)
            df['source'] = 'csl'
            self._save(df, 'csl')
            return df
        except Exception as e:
            print(f"✗ 错误: {e}")
            return pd.DataFrame()
    
    def _save(self, df, name):
        if len(df) == 0:
            return
        output = self.output_dir / f"{name}_{len(df)}.csv"
        df[['text', 'source', 'length']].to_csv(output, index=False, encoding='utf-8-sig')
        print(f"✓ 保存 {len(df)} 条 -> {output}")
        print(f"  平均长度: {df['length'].mean():.0f} 字符")
    
    def merge_all(self):
        """合并所有数据"""
        print(f"\n{'='*60}")
        print("合并所有数据...")
        print('='*60)
        
        files = list(self.output_dir.glob("*.csv"))
        if not files:
            print("✗ 没有找到数据文件")
            return None
        
        dfs = []
        for f in files:
            if f.name.startswith("merged_"):
                continue
            df = pd.read_csv(f, encoding='utf-8-sig')
            dfs.append(df)
            print(f"  {f.name}: {len(df)} 条")
        
        merged = pd.concat(dfs, ignore_index=True)
        merged = merged.drop_duplicates(subset=['text'], keep='first')
        
        output = self.output_dir / f"merged_human_{len(merged)}.csv"
        merged.to_csv(output, index=False, encoding='utf-8-sig')
        
        print(f"\n✓ 合并完成: {len(merged)} 条（去重后）")
        print(f"  保存到: {output}")
        print(f"  平均长度: {merged['length'].mean():.0f}")
        print(f"  长度范围: {merged['length'].min()} - {merged['length'].max()}")
        print(f"\n数据源分布:")
        print(merged['source'].value_counts())
        
        return merged


def main():
    print("="*60)
    print("大规模人类文本数据收集")
    print("目标: 50,000条高质量中文文本")
    print("="*60)
    
    collector = LargeScaleCollector()
    
    # 收集各数据源
    collector.collect_thucnews(target=20000)
    collector.collect_wikipedia(target=10000)
    collector.collect_clue(target=10000)
    collector.collect_weibo(target=5000)
    collector.collect_csl(target=5000)
    
    # 合并
    merged = collector.merge_all()
    
    if merged is not None and len(merged) >= 50000:
        print(f"\n🎉 成功！收集了 {len(merged)} 条人类文本")
    else:
        print(f"\n⚠️  当前收集了 {len(merged) if merged is not None else 0} 条，未达到5万目标")
        print("   可以调整各数据源的target参数或添加更多数据源")


if __name__ == "__main__":
    main()
