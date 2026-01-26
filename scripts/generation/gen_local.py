#!/usr/bin/env python3
"""使用本地反代生成混合数据 (高速)"""
import os, json, time, random, hashlib
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor

API = {"url": "http://192.168.60.105:8317/v1", "key": "cliproxyapi-test-key-2026"}
MODELS = ["deepseek-v3.1", "qwen3-32b", "glm-4.7"]
TOPICS = ["人工智能", "健康生活", "城市发展", "环境保护", "教育问题", "科技创新", "职场经验", "旅行见闻"]
OUTPUT_DIR = "datasets/hybrid"

def call(model, prompt):
    try:
        r = requests.post(f"{API['url']}/chat/completions",
            headers={"Authorization": f"Bearer {API['key']}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 400, "temperature": 0.8},
            timeout=90)
        if r.status_code == 200:
            msg = r.json()["choices"][0]["message"]
            return msg.get("content") or msg.get("reasoning_content", "")
    except Exception as e:
        print(f"  Err: {str(e)[:30]}", flush=True)
    return None

def save(data, name):
    with open(f"{OUTPUT_DIR}/{name}", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    print("本地反代生成开始...", flush=True)
    human_df = pd.read_csv("datasets/final_clean/all_human.csv")
    humans = human_df["text"].sample(300).tolist()
    
    c2_results, c3_results, c4_results = [], [], []
    
    # C2: 续写 100条
    print("\n=== C2 续写 ===", flush=True)
    for h in humans[:120]:
        if len(h) < 80: continue
        cut = random.randint(int(len(h)*0.3), int(len(h)*0.5))
        human_part = h[:cut]
        model = random.choice(MODELS)
        ai_part = call(model, f"续写以下文字，100-150字：\n{human_part}")
        if ai_part and len(ai_part) > 30:
            full = human_part + ai_part
            c2_results.append({
                "text_id": hashlib.md5(full.encode()).hexdigest()[:12],
                "text": full, "category": "C2", "label": 1,
                "source_model": model, "api": "local", "boundary": len(human_part)
            })
            print(f"[C2 {len(c2_results)}/100]", flush=True)
            save(c2_results, "c2_local.json")
        if len(c2_results) >= 100: break
        time.sleep(0.1)  # 3账号轮询，可以很快
    
    # C3: 改写 50条
    print("\n=== C3 改写 ===", flush=True)
    for i in range(60):
        topic = random.choice(TOPICS)
        m1, m2 = random.sample(MODELS, 2)
        text = call(m1, f"写一段关于{topic}的短文，150字。")
        if not text: continue
        edited = call(m2, f"改写这段话使其更口语化：\n{text}")
        if edited and len(edited) > 50:
            c3_results.append({
                "text_id": hashlib.md5(edited.encode()).hexdigest()[:12],
                "text": edited, "category": "C3", "label": 1,
                "source_model": f"{m1}+{m2}", "api": "local"
            })
            print(f"[C3 {len(c3_results)}/50]", flush=True)
            save(c3_results, "c3_local.json")
        if len(c3_results) >= 50: break
        time.sleep(0.1)
    
    # C4: 润色 100条
    print("\n=== C4 润色 ===", flush=True)
    for h in humans[120:]:
        if len(h) < 50 or len(h) > 500: continue
        model = random.choice(MODELS)
        polished = call(model, f"润色这段文字：\n{h[:400]}")
        if polished and len(polished) > 30:
            c4_results.append({
                "text_id": hashlib.md5(polished.encode()).hexdigest()[:12],
                "text": polished, "category": "C4", "label": 1,
                "source_model": model, "api": "local", "original_human": h
            })
            print(f"[C4 {len(c4_results)}/100]", flush=True)
            save(c4_results, "c4_local.json")
        if len(c4_results) >= 100: break
        time.sleep(0.1)
    
    print(f"\n完成! C2:{len(c2_results)} C3:{len(c3_results)} C4:{len(c4_results)}", flush=True)

if __name__ == "__main__":
    main()
