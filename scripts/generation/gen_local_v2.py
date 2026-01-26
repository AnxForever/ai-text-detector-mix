#!/usr/bin/env python3
"""本地反代高速生成 (4并发)"""
import os, json, time, random, hashlib
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

API = {"url": "http://192.168.60.105:8317/v1", "key": "cliproxyapi-test-key-2026"}
MODELS = ["deepseek-v3.1", "qwen3-32b", "glm-4.7", "qwen3-235b"]
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
    except:
        pass
    return None

def gen_c2(args):
    h, idx = args
    if len(h) < 80: return None
    cut = random.randint(int(len(h)*0.3), int(len(h)*0.5))
    human_part = h[:cut]
    model = random.choice(MODELS)
    ai_part = call(model, f"续写以下文字，100-150字：\n{human_part}")
    if ai_part and len(ai_part) > 30:
        full = human_part + ai_part
        return {"text": full, "category": "C2", "label": 1, "source_model": model, "boundary": len(human_part)}
    return None

def gen_c4(args):
    h, idx = args
    if len(h) < 50 or len(h) > 600: return None
    model = random.choice(MODELS)
    polished = call(model, f"润色这段文字使其更通顺：\n{h[:500]}")
    if polished and len(polished) > 30:
        return {"text": polished, "category": "C4", "label": 1, "source_model": model, "original_human": h}
    return None

def gen_c3(args):
    topic, idx = args
    m1, m2 = random.sample(MODELS, 2)
    text = call(m1, f"写一段关于{topic}的短文，150字。")
    if not text or len(text) < 50: return None
    edited = call(m2, f"改写这段话使其更口语化自然：\n{text}")
    if edited and len(edited) > 50:
        return {"text": edited, "category": "C3", "label": 1, "source_model": f"{m1}+{m2}"}
    return None

def save(data, name):
    with open(f"{OUTPUT_DIR}/{name}", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    print("高速生成 (4并发)...", flush=True)
    human_df = pd.read_csv("datasets/final_clean/all_human.csv")
    humans = human_df["text"].sample(1500).tolist()
    topics = ["人工智能", "健康生活", "城市发展", "环境保护", "教育问题", "科技创新", "职场经验", "旅行见闻", "美食文化", "电影音乐"] * 30
    
    c2_results, c3_results, c4_results = [], [], []
    
    # C2: 400条
    print("\n=== C2 续写 (目标400) ===", flush=True)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(gen_c2, (h, i)) for i, h in enumerate(humans[:500])]
        for f in as_completed(futures):
            r = f.result()
            if r:
                r["text_id"] = hashlib.md5(r["text"].encode()).hexdigest()[:12]
                c2_results.append(r)
                if len(c2_results) % 20 == 0:
                    print(f"[C2 {len(c2_results)}/400]", flush=True)
                    save(c2_results, "c2_local_v2.json")
            if len(c2_results) >= 400: break
    save(c2_results, "c2_local_v2.json")
    print(f"C2完成: {len(c2_results)}条", flush=True)
    
    # C3: 200条
    print("\n=== C3 改写 (目标200) ===", flush=True)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(gen_c3, (t, i)) for i, t in enumerate(topics[:250])]
        for f in as_completed(futures):
            r = f.result()
            if r:
                r["text_id"] = hashlib.md5(r["text"].encode()).hexdigest()[:12]
                c3_results.append(r)
                if len(c3_results) % 20 == 0:
                    print(f"[C3 {len(c3_results)}/200]", flush=True)
                    save(c3_results, "c3_local_v2.json")
            if len(c3_results) >= 200: break
    save(c3_results, "c3_local_v2.json")
    print(f"C3完成: {len(c3_results)}条", flush=True)
    
    # C4: 400条
    print("\n=== C4 润色 (目标400) ===", flush=True)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(gen_c4, (h, i)) for i, h in enumerate(humans[500:1000])]
        for f in as_completed(futures):
            r = f.result()
            if r:
                r["text_id"] = hashlib.md5(r["text"].encode()).hexdigest()[:12]
                c4_results.append(r)
                if len(c4_results) % 20 == 0:
                    print(f"[C4 {len(c4_results)}/400]", flush=True)
                    save(c4_results, "c4_local_v2.json")
            if len(c4_results) >= 400: break
    save(c4_results, "c4_local_v2.json")
    print(f"C4完成: {len(c4_results)}条", flush=True)
    
    print(f"\n=== 总计: C2:{len(c2_results)} C3:{len(c3_results)} C4:{len(c4_results)} ===", flush=True)

if __name__ == "__main__":
    main()
