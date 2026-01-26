#!/usr/bin/env python3
"""C4: 人类+AI润色 (使用x666.me)"""
import os, json, time, random, hashlib
import pandas as pd
import requests

API = {"url": "https://x666.me/v1", "key": "sk-6igx2LDfddaSqTGaK1izWRN6bLsMDBd5DVeImqMCF1mK1ipS"}
MODELS = ["gemini-3-flash-preview", "gpt-5.2", "gpt-4.1-mini"]
OUTPUT = "datasets/hybrid/c4_polished_x666.json"

def call(model, prompt):
    try:
        r = requests.post(f"{API['url']}/chat/completions",
            headers={"Authorization": f"Bearer {API['key']}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 500, "temperature": 0.8},
            timeout=90)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        print(f"  API {r.status_code}", flush=True)
    except Exception as e:
        print(f"  Error: {str(e)[:30]}", flush=True)
    return None

def save(data):
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    print("C4(x666)开始...", flush=True)
    human_df = pd.read_csv("datasets/final_clean/all_human.csv")
    humans = human_df["text"].sample(150).tolist()
    
    results = []
    for i, h in enumerate(humans):
        if len(h) < 50 or len(h) > 500:  # 限制长度
            continue
        
        model = random.choice(MODELS)
        polished = call(model, f"润色这段文字：\n{h[:400]}")
        
        if polished and len(polished) > 30:
            results.append({
                "text_id": hashlib.md5(polished.encode()).hexdigest()[:12],
                "text": polished, "category": "C4", "label": 1,
                "source_model": model, "api": "x666", "original_human": h
            })
            print(f"[C4-x666 {len(results)}/100]", flush=True)
            save(results)
        
        if len(results) >= 100:
            break
        time.sleep(12)  # RPM=5
    
    print(f"C4(x666)完成: {len(results)}条", flush=True)

if __name__ == "__main__":
    main()
