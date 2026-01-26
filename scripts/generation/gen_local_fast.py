#!/usr/bin/env python3
"""本地反代并发生成"""
import os, json, time, random, hashlib
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    except:
        pass
    return None

def gen_c2(h):
    if len(h) < 80: return None
    cut = random.randint(int(len(h)*0.3), int(len(h)*0.5))
    human_part = h[:cut]
    model = random.choice(MODELS)
    ai_part = call(model, f"续写以下文字，100-150字：\n{human_part}")
    if ai_part and len(ai_part) > 30:
        full = human_part + ai_part
        return {"text": full, "category": "C2", "label": 1, "source_model": model, "boundary": len(human_part)}
    return None

def gen_c4(h):
    if len(h) < 50 or len(h) > 500: return None
    model = random.choice(MODELS)
    polished = call(model, f"润色这段文字：\n{h[:400]}")
    if polished and len(polished) > 30:
        return {"text": polished, "category": "C4", "label": 1, "source_model": model, "original_human": h}
    return None

def save(data, name):
    with open(f"{OUTPUT_DIR}/{name}", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    print("并发生成开始 (3 workers)...", flush=True)
    human_df = pd.read_csv("datasets/final_clean/all_human.csv")
    humans = human_df["text"].sample(400).tolist()
    
    c2_results, c4_results = [], []
    
    # C2并发
    print("\n=== C2 续写 (并发) ===", flush=True)
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(gen_c2, h) for h in humans[:150]]
        for f in as_completed(futures):
            r = f.result()
            if r:
                r["text_id"] = hashlib.md5(r["text"].encode()).hexdigest()[:12]
                c2_results.append(r)
                print(f"[C2 {len(c2_results)}]", flush=True)
                if len(c2_results) % 10 == 0:
                    save(c2_results, "c2_local.json")
            if len(c2_results) >= 100:
                break
    save(c2_results, "c2_local.json")
    
    # C4并发
    print("\n=== C4 润色 (并发) ===", flush=True)
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(gen_c4, h) for h in humans[150:]]
        for f in as_completed(futures):
            r = f.result()
            if r:
                r["text_id"] = hashlib.md5(r["text"].encode()).hexdigest()[:12]
                c4_results.append(r)
                print(f"[C4 {len(c4_results)}]", flush=True)
                if len(c4_results) % 10 == 0:
                    save(c4_results, "c4_local.json")
            if len(c4_results) >= 100:
                break
    save(c4_results, "c4_local.json")
    
    print(f"\n完成! C2:{len(c2_results)} C4:{len(c4_results)}", flush=True)

if __name__ == "__main__":
    main()
