#!/usr/bin/env python3
"""
多模型数据生成 - 使用本地API
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

# 本地API配置
API = get_proxy_config("local_proxy", "LOCAL_PROXY", "http://192.168.60.105:8317/v1")
CLIENT = OpenAI(base_url=API["url"], api_key=API["key"])

# 多模型配置 (按家族分组)
MODELS = {
    'deepseek': ['deepseek-v3', 'deepseek-v3.1', 'deepseek-r1'],
    'qwen': ['qwen3-32b', 'qwen3-235b'],
    'glm': ['glm-4.6', 'glm-4.7'],
    'gpt': ['gpt-3.5-turbo', 'gpt-4'],
    'gemini': ['gemini-2.5-flash', 'gemini-3-pro-preview'],
    'other': ['minimax-m2.1', 'kimi-k2-thinking']
}

ALL_MODELS = [m for ms in MODELS.values() for m in ms]

def generate(model: str, system: str, prompt: str, max_tokens=512) -> str:
    """调用API生成"""
    try:
        resp = CLIENT.chat.completions.create(
            model=model,
            messages=[
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"  ✗ {model}: {e}")
        return None

def gen_c2_multimodel(human_texts: list, n_per_model: int = 50):
    """多模型生成C2续写数据"""
    results = []
    
    # 每个模型家族选一个代表
    selected_models = ['deepseek-v3', 'qwen3-32b', 'glm-4.7', 'gpt-4', 'gemini-2.5-flash']
    
    for model in selected_models:
        print(f"\n>>> 模型: {model}")
        samples = random.sample(human_texts, min(n_per_model, len(human_texts)))
        
        for i, text in enumerate(samples):
            # 随机截断 (30%-70%)
            cut = int(len(text) * random.uniform(0.3, 0.7))
            prefix = text[:cut]
            
            content = generate(
                model=model,
                system='你是写作助手。直接续写，不要任何开场白或解释。',
                prompt=f'续写以下文本，约100-200字：\n\n{prefix}'
            )
            
            if content:
                results.append({
                    'text': prefix + '[SEP]' + content,
                    'human_part': prefix,
                    'ai_part': content,
                    'boundary': len(prefix),
                    'model': model,
                    'label': 1,
                    'category': 'C2'
                })
                print(f"  [{i+1}/{len(samples)}] ✓ {len(content)}字")
            
            time.sleep(1)
    
    return results

def gen_c3_multimodel(human_texts: list, n_per_model: int = 30):
    """多模型生成C3改写数据"""
    results = []
    selected_models = ['deepseek-v3', 'qwen3-32b', 'glm-4.7', 'gpt-4']
    
    for model in selected_models:
        print(f"\n>>> 模型: {model}")
        samples = random.sample(human_texts, min(n_per_model, len(human_texts)))
        
        for i, text in enumerate(samples):
            content = generate(
                model=model,
                system='你是改写专家。直接输出改写结果，不要解释。',
                prompt=f'用不同的表达方式改写以下文本，保持原意：\n\n{text[:500]}'
            )
            
            if content:
                results.append({
                    'text': content,
                    'original': text[:500],
                    'model': model,
                    'label': 1,
                    'category': 'C3'
                })
                print(f"  [{i+1}/{len(samples)}] ✓")
            
            time.sleep(1)
    
    return results

def gen_c4_multimodel(human_texts: list, n_per_model: int = 30):
    """多模型生成C4润色数据"""
    results = []
    selected_models = ['deepseek-v3', 'qwen3-32b', 'glm-4.7', 'gpt-4']
    
    for model in selected_models:
        print(f"\n>>> 模型: {model}")
        samples = random.sample(human_texts, min(n_per_model, len(human_texts)))
        
        for i, text in enumerate(samples):
            content = generate(
                model=model,
                system='你是编辑。直接输出润色结果。',
                prompt=f'润色以下文本，使其更流畅专业：\n\n{text[:500]}'
            )
            
            if content:
                results.append({
                    'text': content,
                    'original': text[:500],
                    'model': model,
                    'label': 1,
                    'category': 'C4'
                })
                print(f"  [{i+1}/{len(samples)}] ✓")
            
            time.sleep(1)
    
    return results

def main():
    print("=" * 60)
    print("多模型数据生成")
    print("=" * 60)
    
    # 加载人类文本
    human_df = pd.read_csv('datasets/active/core_v1/all_human.csv')
    human_texts = human_df['text'].dropna().tolist()
    print(f"人类文本: {len(human_texts)} 条")
    
    # 生成C2
    print("\n" + "=" * 60)
    print("生成C2 (续写) - 多模型")
    print("=" * 60)
    c2_data = gen_c2_multimodel(human_texts, n_per_model=50)
    
    # 生成C3
    print("\n" + "=" * 60)
    print("生成C3 (改写) - 多模型")
    print("=" * 60)
    c3_data = gen_c3_multimodel(human_texts, n_per_model=30)
    
    # 生成C4
    print("\n" + "=" * 60)
    print("生成C4 (润色) - 多模型")
    print("=" * 60)
    c4_data = gen_c4_multimodel(human_texts, n_per_model=30)
    
    # 保存
    all_data = c2_data + c3_data + c4_data
    
    output_file = f'datasets/mixed/hybrid/multimodel_{datetime.now().strftime("%Y%m%d_%H%M")}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n" + "=" * 60)
    print(f"生成完成!")
    print(f"  C2: {len(c2_data)} 条")
    print(f"  C3: {len(c3_data)} 条")
    print(f"  C4: {len(c4_data)} 条")
    print(f"  总计: {len(all_data)} 条")
    print(f"  保存: {output_file}")
    print("=" * 60)

if __name__ == '__main__':
    main()
