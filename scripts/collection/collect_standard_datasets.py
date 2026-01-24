#!/usr/bin/env python3
"""
多源数据集收集工具
支持HC3, NLPCC, MGTBench等标准数据集
"""

import pandas as pd
from datasets import load_dataset
from pathlib import Path
from typing import Optional
import json

class MultiSourceCollector:
    def __init__(self, output_dir="datasets/multisource"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_length = 300
        self.max_length = 3000
    
    def collect_hc3(self, target_count=10000):
        """HC3中文数据集 - 人类 vs ChatGPT"""
        print(f"\n[HC3] 收集中文对比数据...")
        try:
            # HC3数据集在HuggingFace上
            ds = load_dataset("Hello-SimpleAI/HC3-Chinese", split="train")
            
            data = []
            for item in ds:
                # 人类回答
                if 'human_answers' in item and item['human_answers']:
                    for answer in item['human_answers'][:1]:  # 取第一个
                        if self.min_length <= len(answer) <= self.max_length:
                            data.append({
                                'text': answer,
                                'label': 0,
                                'source': 'hc3',
                                'type': 'human',
                                'length': len(answer)
                            })
                
                # ChatGPT回答
                if 'chatgpt_answers' in item and item['chatgpt_answers']:
                    for answer in item['chatgpt_answers'][:1]:
                        if self.min_length <= len(answer) <= self.max_length:
                            data.append({
                                'text': answer,
                                'label': 1,
                                'source': 'hc3',
                                'type': 'chatgpt',
                                'length': len(answer)
                            })
                
                if len(data) >= target_count:
                    break
            
            df = pd.DataFrame(data[:target_count])
            output = self.output_dir / f"hc3_{len(df)}.csv"
            df.to_csv(output, index=False, encoding='utf-8-sig')
            
            print(f"✓ HC3: {len(df)} 条")
            print(f"  人类: {(df['label']==0).sum()}")
            print(f"  AI: {(df['label']==1).sum()}")
            return df
            
        except Exception as e:
            print(f"✗ HC3加载失败: {e}")
            print("  提示: pip install datasets")
            return None
    
    def collect_mgtbench(self, target_count=10000):
        """MGTBench - 多模型生成文本"""
        print(f"\n[MGTBench] 收集多模型数据...")
        try:
            # MGTBench可能需要手动下载或从特定源加载
            # 这里提供框架，具体实现取决于数据集格式
            
            print("⚠️  MGTBench需要手动下载")
            print("  下载地址: https://github.com/xinleihe/MGTBench")
            print("  下载后放到: datasets/mgtbench/")
            
            # 如果已下载，尝试加载
            mgtbench_path = Path("datasets/mgtbench")
            if mgtbench_path.exists():
                # 假设格式为CSV
                files = list(mgtbench_path.glob("*.csv"))
                if files:
                    dfs = [pd.read_csv(f, encoding='utf-8-sig') for f in files]
                    df = pd.concat(dfs, ignore_index=True)
                    
                    # 标准化列名
                    if 'text' in df.columns and 'label' in df.columns:
                        df['source'] = 'mgtbench'
                        df['length'] = df['text'].str.len()
                        df = df[(df['length'] >= self.min_length) & 
                               (df['length'] <= self.max_length)]
                        
                        if len(df) > target_count:
                            df = df.sample(n=target_count, random_state=42)
                        
                        output = self.output_dir / f"mgtbench_{len(df)}.csv"
                        df.to_csv(output, index=False, encoding='utf-8-sig')
                        print(f"✓ MGTBench: {len(df)} 条")
                        return df
            
            return None
            
        except Exception as e:
            print(f"✗ MGTBench加载失败: {e}")
            return None
    
    def collect_nlpcc2025(self, target_count=10000):
        """NLPCC 2025中文AI检测数据"""
        print(f"\n[NLPCC 2025] 收集中文AI检测数据...")
        try:
            print("⚠️  NLPCC 2025需要注册下载")
            print("  官网: http://tcci.ccf.org.cn/conference/2025/")
            print("  下载后放到: datasets/nlpcc2025/")
            
            nlpcc_path = Path("datasets/nlpcc2025")
            if nlpcc_path.exists():
                files = list(nlpcc_path.glob("*.csv")) + list(nlpcc_path.glob("*.json"))
                if files:
                    data = []
                    for file in files:
                        if file.suffix == '.csv':
                            df = pd.read_csv(file, encoding='utf-8-sig')
                            data.append(df)
                        elif file.suffix == '.json':
                            with open(file, 'r', encoding='utf-8') as f:
                                json_data = json.load(f)
                                df = pd.DataFrame(json_data)
                                data.append(df)
                    
                    if data:
                        df = pd.concat(data, ignore_index=True)
                        df['source'] = 'nlpcc2025'
                        df['length'] = df['text'].str.len()
                        df = df[(df['length'] >= self.min_length) & 
                               (df['length'] <= self.max_length)]
                        
                        if len(df) > target_count:
                            df = df.sample(n=target_count, random_state=42)
                        
                        output = self.output_dir / f"nlpcc2025_{len(df)}.csv"
                        df.to_csv(output, index=False, encoding='utf-8-sig')
                        print(f"✓ NLPCC 2025: {len(df)} 条")
                        return df
            
            return None
            
        except Exception as e:
            print(f"✗ NLPCC 2025加载失败: {e}")
            return None
    
    def collect_thucnews(self, target_count=20000):
        """THUCNews - 真实人类新闻文本"""
        print(f"\n[THUCNews] 收集人类新闻文本...")
        try:
            ds = load_dataset("oyxy2019/THUCNewsText", split="train")
            df = pd.DataFrame(ds)
            
            df['label'] = 0  # 人类文本
            df['source'] = 'thucnews'
            df['type'] = 'human'
            df['length'] = df['text'].str.len()
            
            df = df[(df['length'] >= self.min_length) & 
                   (df['length'] <= self.max_length)]
            
            if len(df) > target_count:
                df = df.sample(n=target_count, random_state=42)
            
            output = self.output_dir / f"thucnews_{len(df)}.csv"
            df[['text', 'label', 'source', 'type', 'length']].to_csv(
                output, index=False, encoding='utf-8-sig'
            )
            
            print(f"✓ THUCNews: {len(df)} 条人类文本")
            return df
            
        except Exception as e:
            print(f"✗ THUCNews加载失败: {e}")
            return None
    
    def merge_all(self):
        """合并所有收集的数据"""
        print(f"\n{'='*60}")
        print("合并所有数据源...")
        print('='*60)
        
        files = list(self.output_dir.glob("*.csv"))
        if not files:
            print("✗ 没有找到数据文件")
            return None
        
        dfs = []
        for file in files:
            if file.name.startswith("merged_"):
                continue
            df = pd.read_csv(file, encoding='utf-8-sig')
            dfs.append(df)
            print(f"  {file.name}: {len(df)} 条")
        
        merged = pd.concat(dfs, ignore_index=True)
        
        # 去重
        merged = merged.drop_duplicates(subset=['text'], keep='first')
        
        # 平衡AI和人类数量
        ai_count = (merged['label'] == 1).sum()
        human_count = (merged['label'] == 0).sum()
        target_count = min(ai_count, human_count)
        
        df_ai = merged[merged['label'] == 1].sample(n=target_count, random_state=42)
        df_human = merged[merged['label'] == 0].sample(n=target_count, random_state=42)
        
        balanced = pd.concat([df_ai, df_human], ignore_index=True)
        balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)
        
        output = self.output_dir / f"merged_balanced_{len(balanced)}.csv"
        balanced.to_csv(output, index=False, encoding='utf-8-sig')
        
        print(f"\n✓ 合并完成: {len(balanced)} 条（1:1平衡）")
        print(f"  AI: {(balanced['label']==1).sum()}")
        print(f"  人类: {(balanced['label']==0).sum()}")
        print(f"  平均长度: {balanced['length'].mean():.0f}")
        print(f"\n数据源分布:")
        print(balanced['source'].value_counts())
        
        return balanced


def main():
    print("="*60)
    print("多源数据集收集工具")
    print("="*60)
    
    collector = MultiSourceCollector()
    
    print("\n可用数据源:")
    print("  1. HC3 (中文, HuggingFace)")
    print("  2. THUCNews (中文人类文本, HuggingFace)")
    print("  3. MGTBench (需手动下载)")
    print("  4. NLPCC 2025 (需手动下载)")
    print("  5. 收集所有可用数据源")
    print("  6. 合并已收集的数据")
    
    choice = input("\n请选择 (1-6): ").strip()
    
    if choice == '1':
        collector.collect_hc3(target_count=10000)
    elif choice == '2':
        collector.collect_thucnews(target_count=20000)
    elif choice == '3':
        collector.collect_mgtbench(target_count=10000)
    elif choice == '4':
        collector.collect_nlpcc2025(target_count=10000)
    elif choice == '5':
        collector.collect_hc3(target_count=10000)
        collector.collect_thucnews(target_count=20000)
        collector.collect_mgtbench(target_count=10000)
        collector.collect_nlpcc2025(target_count=10000)
        collector.merge_all()
    elif choice == '6':
        collector.merge_all()
    
    print("\n" + "="*60)
    print("完成！")
    print("="*60)


if __name__ == "__main__":
    main()
