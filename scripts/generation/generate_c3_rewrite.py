#!/usr/bin/env python3
"""生成C3改写数据：AI文本 → AI改写"""
import json, random, time, os, sys
sys.stdout.reconfigure(line_buffering=True)
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# 本地代理
client = OpenAI(base_url="http://192.168.60.105:8317/v1", api_key="cliproxyapi-test-key-2026")

REWRITE_PROMPTS = [
    "请将以下文本改写得更口语化，保持原意：\n\n{text}",
    "请用更简洁的方式重写以下内容：\n\n{text}",
    "请将以下文本改写得更正式、更学术化：\n\n{text}",
    "请用不同的表达方式重写以下内容，保持核心意思不变：\n\n{text}",
    "请将以下文本润色，使其更流畅自然：\n\n{text}",
]

def load_ai_texts():
    """加载AI生成的文本作为改写源"""
    texts = []
    # 从原始AI数据加载
    try:
        with open('datasets/final_clean/all_ai.csv', 'r') as f:
            import csv
            reader = csv.DictReader(f)
            for row in reader:
                if len(row['text']) > 100:
                    texts.append(row['text'])
    except: pass
    return texts[:2000]  # 取前2000条

MODELS = ["gemini-2.5-flash", "qwen3-235b-a22b-instruct", "glm-4.7"]

def rewrite_text(text, idx):
    """改写单条文本"""
    prompt = random.choice(REWRITE_PROMPTS).format(text=text[:1500])
    try:
        resp = client.chat.completions.create(
            model=random.choice(MODELS),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.7
        )
        rewritten = resp.choices[0].message.content
        if not rewritten:
            return None
        rewritten = rewritten.strip()
        # 过滤掉包含说明性文字的结果
        if any(x in rewritten[:50] for x in ['版本', '以下是', '改写后', '这是']):
            lines = rewritten.split('\n')
            for i, line in enumerate(lines):
                if len(line) > 50 and not any(x in line for x in ['版本', '以下是', '改写']):
                    rewritten = '\n'.join(lines[i:])
                    break
        return {"text_id": f"c3_new_{idx}", "text": rewritten, "category": "C3", "label": 1, "source_model": "deepseek"}
    except Exception as e:
        print(f"Error {idx}: {e}")
        return None

def main():
    ai_texts = load_ai_texts()
    print(f"Loaded {len(ai_texts)} AI texts")
    
    results = []
    target = 1000
    
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(rewrite_text, t, i): i for i, t in enumerate(random.sample(ai_texts, min(target, len(ai_texts))))}
        for future in as_completed(futures):
            r = future.result()
            if r and len(r['text']) > 80:
                results.append(r)
                if len(results) % 50 == 0:
                    print(f"Progress: {len(results)}/{target}")
    
    out_path = 'datasets/hybrid/c3_new.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} to {out_path}")

if __name__ == "__main__":
    main()
