#!/usr/bin/env python3
"""C4: 人类原文+AI润色 (gemini-3-pro)"""
import os, json, time, random, hashlib
import pandas as pd
import requests

API = {"url": "https://ai.hybgzs.com/v1", "key": "sk-LQO5p6niHb31gBt_2T_yqGMgiKwgnm9b2os3QCgjuRphSTITFlsnBWFBGCI", "model": "gemini-3-pro-preview", "rpm": 5}
OUTPUT = "datasets/hybrid/c4_polished.json"
os.makedirs("datasets/hybrid", exist_ok=True)

def call_api(prompt):
    try:
        r = requests.post(f"{API['url']}/chat/completions",
            headers={"Authorization": f"Bearer {API['key']}", "Content-Type": "application/json"},
            json={"model": API["model"], "messages": [{"role": "user", "content": prompt}], "max_tokens": 600, "temperature": 0.7},
            timeout=90)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error: {e}")
    return None

def main():
    human_df = pd.read_csv("datasets/final_clean/all_human.csv")
    humans = human_df["text"].sample(250).tolist()
    
    results = []
    for i, h in enumerate(humans):
        if len(h) < 50 or len(h) > 1000:
            continue
        prompt = f"请润色以下文字，使其更通顺流畅，但保持原意：\n\n{h}"
        polished = call_api(prompt)
        
        if polished and len(polished) > 30:
            results.append({
                "text_id": hashlib.md5(polished.encode()).hexdigest()[:12],
                "text": polished,
                "category": "C4",
                "label": 1,
                "source_model": API["model"],
                "original_human": h,
                "spans": [{"start": 0, "end": len(polished), "label": "Polished"}]
            })
            print(f"[{len(results)}/200] C4 generated")
        
        if len(results) >= 200:
            break
        time.sleep(60 / API["rpm"])
    
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"C4完成: {len(results)}条 -> {OUTPUT}")

if __name__ == "__main__":
    main()
