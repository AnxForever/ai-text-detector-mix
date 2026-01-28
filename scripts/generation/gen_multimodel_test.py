#!/usr/bin/env python3
"""
多模型数据生成 - 小批量测试版
"""
import json
import random
import time
import os
import sys
from datetime import datetime

import pandas as pd
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from scripts.utils.api_config import get_proxy_config

API = get_proxy_config("local_proxy", "LOCAL_PROXY", "http://192.168.60.105:8317/v1")
CLIENT = OpenAI(base_url=API["url"], api_key=API["key"])

MODELS = ['deepseek-v3', 'qwen3-32b', 'glm-4.7', 'gpt-4', 'gemini-2.5-flash']

def generate(model, prompt, max_retries=3):
    for _ in range(max_retries):
        try:
            resp = CLIENT.chat.completions.create(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=300,
                timeout=30
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"    重试: {e}")
            time.sleep(2)
    return None

def main():
    # 加载数据
    df = pd.read_csv('datasets/active/core_v1/all_human.csv')
    texts = df['text'].dropna().tolist()
    print(f"人类文本: {len(texts)} 条\n")
    
    results = []
    n_per_model = 10  # 每个模型10条
    
    for model in MODELS:
        print(f">>> {model}")
        samples = random.sample(texts, n_per_model)
        
        for i, text in enumerate(samples):
            cut = int(len(text) * random.uniform(0.3, 0.6))
            prefix = text[:cut]
            
            content = generate(model, f'直接续写以下文本，约100字，不要开场白：\n\n{prefix}')
            
            if content:
                results.append({
                    'text': prefix + '[SEP]' + content,
                    'boundary': len(prefix),
                    'model': model,
                    'label': 1,
                    'category': 'C2'
                })
                print(f"  [{i+1}/{n_per_model}] ✓")
            else:
                print(f"  [{i+1}/{n_per_model}] ✗")
            
            time.sleep(0.5)
    
    # 保存
    out = f'datasets/mixed/hybrid/multimodel_test.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n完成: {len(results)} 条 -> {out}")
    
    # 统计
    from collections import Counter
    model_counts = Counter(r['model'] for r in results)
    for m, c in model_counts.items():
        print(f"  {m}: {c}")

if __name__ == '__main__':
    main()
