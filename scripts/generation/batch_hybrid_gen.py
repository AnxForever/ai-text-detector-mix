#!/usr/bin/env python3
"""批量生成混合数据集 - 长时间后台运行"""
import json, random, time, os, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)

client = OpenAI(base_url="http://192.168.60.105:8317/v1", api_key="cliproxyapi-test-key-2026")
MODELS = ["gemini-2.5-flash", "qwen3-235b-a22b-instruct", "glm-4.7"]

# 加载人类文本和AI文本
def load_texts():
    human, ai = [], []
    try:
        df = pd.read_csv('datasets/final_clean/all_human.csv')
        human = [t for t in df['text'].tolist() if len(str(t)) > 100][:3000]
    except: pass
    try:
        df = pd.read_csv('datasets/final_clean/all_ai.csv')
        ai = [t for t in df['text'].tolist() if len(str(t)) > 100][:3000]
    except: pass
    return human, ai

def call_api(prompt, max_tokens=800):
    try:
        resp = client.chat.completions.create(
            model=random.choice(MODELS),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=0.7
        )
        return resp.choices[0].message.content
    except Exception as e:
        return None

def gen_c2(human_text, idx):
    """C2: 人类开头 + AI续写"""
    sentences = human_text.replace('。', '。\n').split('\n')
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if len(sentences) < 2: return None
    cut = random.randint(1, min(3, len(sentences)-1))
    human_part = ''.join(sentences[:cut])
    prompt = f"请继续写下面这段话，保持风格一致，写200-400字：\n\n{human_part}"
    ai_part = call_api(prompt)
    if not ai_part or len(ai_part) < 50: return None
    return {"text_id": f"c2_batch_{idx}", "text": human_part + ai_part, 
            "category": "C2", "label": 1, "boundary": len(human_part)}

def gen_c3(ai_text, idx):
    """C3: AI文本改写"""
    prompts = [
        f"请将以下文本改写得更口语化：\n\n{ai_text[:1200]}",
        f"请用更简洁的方式重写：\n\n{ai_text[:1200]}",
        f"请改写以下内容，使其更正式：\n\n{ai_text[:1200]}",
    ]
    result = call_api(random.choice(prompts))
    if not result or len(result) < 80: return None
    # 清理可能的说明文字
    lines = result.split('\n')
    clean = []
    for line in lines:
        if len(line) > 30 and not any(x in line[:20] for x in ['版本', '以下', '改写', '这是', '好的']):
            clean.append(line)
    if not clean: return None
    return {"text_id": f"c3_batch_{idx}", "text": '\n'.join(clean), "category": "C3", "label": 1}

def gen_c4(human_text, idx):
    """C4: 人类文本 + AI润色"""
    prompt = f"请对以下文本进行轻微润色，修正语法错误，使表达更流畅，但保持原意和风格：\n\n{human_text[:1200]}"
    result = call_api(prompt, max_tokens=1000)
    if not result or len(result) < 80: return None
    return {"text_id": f"c4_batch_{idx}", "text": result, "category": "C4", "label": 1}

def main():
    print("Loading texts...")
    human_texts, ai_texts = load_texts()
    print(f"Loaded: {len(human_texts)} human, {len(ai_texts)} AI texts")
    
    results = {"C2": [], "C3": [], "C4": []}
    targets = {"C2": 1000, "C3": 1000, "C4": 500}  # 目标数量
    
    idx = 0
    round_num = 0
    
    while any(len(results[k]) < targets[k] for k in targets):
        round_num += 1
        print(f"\n=== Round {round_num} ===")
        print(f"Current: C2={len(results['C2'])}, C3={len(results['C3'])}, C4={len(results['C4'])}")
        
        tasks = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            # C2
            if len(results['C2']) < targets['C2']:
                for t in random.sample(human_texts, min(20, len(human_texts))):
                    tasks.append(executor.submit(gen_c2, t, idx))
                    idx += 1
            # C3
            if len(results['C3']) < targets['C3']:
                for t in random.sample(ai_texts, min(20, len(ai_texts))):
                    tasks.append(executor.submit(gen_c3, t, idx))
                    idx += 1
            # C4
            if len(results['C4']) < targets['C4']:
                for t in random.sample(human_texts, min(10, len(human_texts))):
                    tasks.append(executor.submit(gen_c4, t, idx))
                    idx += 1
            
            for future in as_completed(tasks):
                r = future.result()
                if r and 'category' in r:
                    cat = r['category']
                    if len(results[cat]) < targets[cat]:
                        results[cat].append(r)
        
        # 每轮保存
        for cat in results:
            if results[cat]:
                with open(f'datasets/hybrid/{cat.lower()}_batch.json', 'w') as f:
                    json.dump(results[cat], f, ensure_ascii=False, indent=2)
        
        print(f"Saved: C2={len(results['C2'])}, C3={len(results['C3'])}, C4={len(results['C4'])}")
        time.sleep(2)  # 避免API限流
    
    print("\n=== Complete ===")
    print(f"Final: C2={len(results['C2'])}, C3={len(results['C3'])}, C4={len(results['C4'])}")

if __name__ == "__main__":
    main()
