#!/usr/bin/env python3
"""
多模型数据生成 - 稳定长时间运行版
目标: 生成1000+条多模型混合数据
预计时间: 3-4小时
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

# 配置
API = get_proxy_config("local_proxy", "LOCAL_PROXY", "http://192.168.60.105:8317/v1")
CLIENT = OpenAI(base_url=API["url"], api_key=API["key"])

# 稳定模型列表 (按优先级)
STABLE_MODELS = [
    'deepseek-v3',
    'qwen3-32b', 
    'glm-4.7',
    'gpt-3.5-turbo',
    'gemini-2.5-flash'
]

OUTPUT_DIR = 'datasets/mixed/hybrid/multimodel'
LOG_FILE = 'logs/multimodel_overnight.log'

def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

def generate(model, prompt, max_retries=5):
    """带重试的生成"""
    for attempt in range(max_retries):
        try:
            resp = CLIENT.chat.completions.create(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=400,
                temperature=0.7,
                timeout=60
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                wait = (attempt + 1) * 5
                log(f"  重试 {attempt+1}/{max_retries}: {str(e)[:50]}... 等待{wait}s")
                time.sleep(wait)
            else:
                log(f"  失败: {str(e)[:80]}")
                return None
    return None

def save_checkpoint(data, filename):
    """保存检查点"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = f"{OUTPUT_DIR}/{filename}"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path

def gen_c2(texts, model, n=50):
    """生成C2续写数据"""
    results = []
    samples = random.sample(texts, min(n, len(texts)))
    
    for i, text in enumerate(samples):
        cut = int(len(text) * random.uniform(0.3, 0.6))
        prefix = text[:cut]
        
        content = generate(model, f'直接续写以下文本，约150字，不要任何开场白或解释：\n\n{prefix}')
        
        if content and len(content) > 50:
            results.append({
                'text': prefix + '[SEP]' + content,
                'human_part': prefix,
                'ai_part': content,
                'boundary': len(prefix),
                'model': model,
                'label': 1,
                'category': 'C2'
            })
            log(f"  C2 [{i+1}/{n}] ✓ {len(content)}字")
        else:
            log(f"  C2 [{i+1}/{n}] ✗")
        
        time.sleep(1.5)
    
    return results

def gen_c3(texts, model, n=30):
    """生成C3改写数据"""
    results = []
    samples = random.sample(texts, min(n, len(texts)))
    
    for i, text in enumerate(samples):
        src = text[:500] if len(text) > 500 else text
        content = generate(model, f'用不同的表达方式改写以下文本，保持原意，直接输出结果：\n\n{src}')
        
        if content and len(content) > 50:
            results.append({
                'text': content,
                'original': src,
                'model': model,
                'label': 1,
                'category': 'C3'
            })
            log(f"  C3 [{i+1}/{n}] ✓")
        else:
            log(f"  C3 [{i+1}/{n}] ✗")
        
        time.sleep(1.5)
    
    return results

def gen_c4(texts, model, n=30):
    """生成C4润色数据"""
    results = []
    samples = random.sample(texts, min(n, len(texts)))
    
    for i, text in enumerate(samples):
        src = text[:500] if len(text) > 500 else text
        content = generate(model, f'润色以下文本，使其更流畅专业，直接输出结果：\n\n{src}')
        
        if content and len(content) > 50:
            results.append({
                'text': content,
                'original': src,
                'model': model,
                'label': 1,
                'category': 'C4'
            })
            log(f"  C4 [{i+1}/{n}] ✓")
        else:
            log(f"  C4 [{i+1}/{n}] ✗")
        
        time.sleep(1.5)
    
    return results

def main():
    log("=" * 60)
    log("多模型数据生成 - 夜间长时间运行")
    log(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 60)
    
    # 加载人类文本
    df = pd.read_csv('datasets/active/core_v1/all_human.csv')
    texts = df['text'].dropna().tolist()
    log(f"人类文本: {len(texts)} 条")
    
    all_results = []
    
    # 每个模型生成数据
    for model in STABLE_MODELS:
        log(f"\n{'='*60}")
        log(f"模型: {model}")
        log(f"{'='*60}")
        
        # C2: 50条/模型
        log(f"\n生成C2 (续写)...")
        c2 = gen_c2(texts, model, n=50)
        all_results.extend(c2)
        
        # C3: 30条/模型
        log(f"\n生成C3 (改写)...")
        c3 = gen_c3(texts, model, n=30)
        all_results.extend(c3)
        
        # C4: 30条/模型
        log(f"\n生成C4 (润色)...")
        c4 = gen_c4(texts, model, n=30)
        all_results.extend(c4)
        
        # 每个模型完成后保存检查点
        checkpoint = save_checkpoint(all_results, f'checkpoint_{model.replace("/", "_")}.json')
        log(f"\n检查点已保存: {checkpoint} ({len(all_results)} 条)")
        
        # 模型间休息
        log(f"休息30秒...")
        time.sleep(30)
    
    # 最终保存
    final_file = save_checkpoint(all_results, f'multimodel_final_{datetime.now().strftime("%Y%m%d_%H%M")}.json')
    
    log(f"\n{'='*60}")
    log(f"生成完成!")
    log(f"总计: {len(all_results)} 条")
    log(f"保存: {final_file}")
    log(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"{'='*60}")
    
    # 统计
    from collections import Counter
    model_counts = Counter(r['model'] for r in all_results)
    cat_counts = Counter(r['category'] for r in all_results)
    
    log("\n模型分布:")
    for m, c in model_counts.items():
        log(f"  {m}: {c}")
    
    log("\n类别分布:")
    for cat, c in cat_counts.items():
        log(f"  {cat}: {c}")

if __name__ == '__main__':
    main()
