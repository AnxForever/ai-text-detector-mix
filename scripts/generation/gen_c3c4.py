#!/usr/bin/env python3
"""补充生成C3和C4"""
import os, json, time, random, hashlib
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

API = {"url": "http://192.168.60.105:8317/v1", "key": "cliproxyapi-test-key-2026"}
MODELS = ["deepseek-v3.1", "qwen3-32b", "glm-4.7"]
OUTPUT_DIR = "datasets/hybrid"

def call(model, prompt):
    try:
        r = requests.post(f"{API['url']}/chat/completions",
            headers={"Authorization": f"Bearer {API['key']}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 400, "temperature": 0.8},
            timeout=120)
        if r.status_code == 200:
            msg = r.json()["choices"][0]["message"]
            return msg.get("content") or msg.get("reasoning_content", "")
    except Exception as e:
        print(f"  err: {str(e)[:30]}", flush=True)
    return None

def save(data, name):
    with open(f"{OUTPUT_DIR}/{name}", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    print("补充生成C3和C4...", flush=True)
    human_df = pd.read_csv("datasets/final_clean/all_human.csv")
    humans = human_df["text"].sample(600).tolist()
    topics = ["人工智能", "健康生活", "城市发展", "环境保护", "教育问题", "科技创新", "职场经验", "旅行见闻", "美食文化", "电影音乐"]
    
    c3_results, c4_results = [], []
    
    # C3: 200条 (串行，因为每条需要2次调用)
    print("\n=== C3 改写 ===", flush=True)
    for i in range(300):
        topic = random.choice(topics)
        m1, m2 = random.sample(MODELS, 2)
        text = call(m1, f"写一段关于{topic}的短文，150字。")
        if not text or len(text) < 50:
            continue
        edited = call(m2, f"改写这段话使其更口语化：\n{text}")
        if edited and len(edited) > 50:
            c3_results.append({
                "text_id": hashlib.md5(edited.encode()).hexdigest()[:12],
                "text": edited, "category": "C3", "label": 1, "source_model": f"{m1}+{m2}"
            })
            print(f"[C3 {len(c3_results)}/200]", flush=True)
            if len(c3_results) % 20 == 0:
                save(c3_results, "c3_local_v2.json")
        if len(c3_results) >= 200:
            break
    save(c3_results, "c3_local_v2.json")
    
    # C4: 400条 (并发)
    print("\n=== C4 润色 ===", flush=True)
    def gen_c4(h):
        if len(h) < 50 or len(h) > 600: return None
        model = random.choice(MODELS)
        polished = call(model, f"润色这段文字：\n{h[:500]}")
        if polished and len(polished) > 30:
            return {"text": polished, "category": "C4", "label": 1, "source_model": model, "original_human": h}
        return None
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(gen_c4, h) for h in humans]
        for f in as_completed(futures):
            r = f.result()
            if r:
                r["text_id"] = hashlib.md5(r["text"].encode()).hexdigest()[:12]
                c4_results.append(r)
                if len(c4_results) % 20 == 0:
                    print(f"[C4 {len(c4_results)}/400]", flush=True)
                    save(c4_results, "c4_local_v2.json")
            if len(c4_results) >= 400:
                break
    save(c4_results, "c4_local_v2.json")
    
    print(f"\n完成! C3:{len(c3_results)} C4:{len(c4_results)}", flush=True)

if __name__ == "__main__":
    main()
