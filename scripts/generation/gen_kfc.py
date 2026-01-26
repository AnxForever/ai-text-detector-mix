#!/usr/bin/env python3
"""C3+C4混合生成 (kfc-api, RPM=12)"""
import os, json, time, random, hashlib
import pandas as pd
import requests

API = {"url": "https://kfc-api.sxxe.net/v1", "key": "sk-YT09CWSuyzu9tRWANoAmXCuL64JlkLTJl2CY1t6bgUKgePUa"}
MODELS = ["DeepSeek-V3.1", "cursor2-gpt-5", "kimi-k2-instruct"]
TOPICS = ["人工智能", "健康生活", "城市发展", "环境保护", "教育问题", "科技创新", "职场经验", "旅行见闻"]
OUTPUT_DIR = "datasets/hybrid"

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
        print(f"  Err: {str(e)[:30]}", flush=True)
    return None

def save(data, name):
    with open(f"{OUTPUT_DIR}/{name}", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    print("kfc-api生成开始...", flush=True)
    human_df = pd.read_csv("datasets/final_clean/all_human.csv")
    humans = human_df["text"].sample(200).tolist()
    
    c3_results, c4_results = [], []
    
    # C3: 50条
    print("\n=== C3 (kfc) ===", flush=True)
    for i in range(60):
        topic = random.choice(TOPICS)
        m1, m2 = random.sample(MODELS, 2)
        text = call(m1, f"写一段关于{topic}的短文，150字。")
        if not text:
            time.sleep(5)
            continue
        edited = call(m2, f"改写这段话使其更口语化：\n{text}")
        if edited and len(edited) > 50:
            c3_results.append({
                "text_id": hashlib.md5(edited.encode()).hexdigest()[:12],
                "text": edited, "category": "C3", "label": 1,
                "source_model": f"{m1}+{m2}", "api": "kfc"
            })
            print(f"[C3-kfc {len(c3_results)}/50]", flush=True)
            save(c3_results, "c3_edited_kfc.json")
        if len(c3_results) >= 50:
            break
        time.sleep(5)
    
    # C4: 100条
    print("\n=== C4 (kfc) ===", flush=True)
    for h in humans:
        if len(h) < 50 or len(h) > 500:
            continue
        model = random.choice(MODELS)
        polished = call(model, f"润色这段文字：\n{h[:400]}")
        if polished and len(polished) > 30:
            c4_results.append({
                "text_id": hashlib.md5(polished.encode()).hexdigest()[:12],
                "text": polished, "category": "C4", "label": 1,
                "source_model": model, "api": "kfc", "original_human": h
            })
            print(f"[C4-kfc {len(c4_results)}/100]", flush=True)
            save(c4_results, "c4_polished_kfc.json")
        if len(c4_results) >= 100:
            break
        time.sleep(5)
    
    print(f"\nkfc完成! C3:{len(c3_results)} C4:{len(c4_results)}", flush=True)

if __name__ == "__main__":
    main()
