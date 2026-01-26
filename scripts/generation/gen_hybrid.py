#!/usr/bin/env python3
"""C3+C4混合生成（边生成边保存）"""
import os, json, time, random, hashlib
import pandas as pd
import requests

APIS = {
    "hotaru": {"url": "https://api.hotaruapi.top/v1", "key": "sk-WPzv2WwpDbnLp02uro0DyPUy0LyI3VjIRmngMj8fm7BLQqSq"},
    "hybgzs": {"url": "https://ai.hybgzs.com/v1", "key": "sk-LQO5p6niHb31gBt_2T_yqGMgiKwgnm9b2os3QCgjuRphSTITFlsnBWFBGCI"},
}
MODELS = ["gemini-3-pro-preview", "gpt-4.1-mini"]
TOPICS = ["人工智能", "健康生活", "城市发展", "环境保护", "教育问题", "科技创新", "职场经验", "旅行见闻"]
OUTPUT_DIR = "datasets/hybrid"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def call(api_name, model, prompt):
    api = APIS[api_name]
    try:
        r = requests.post(f"{api['url']}/chat/completions",
            headers={"Authorization": f"Bearer {api['key']}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 500, "temperature": 0.8},
            timeout=90)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        print(f"  API {r.status_code}", flush=True)
    except Exception as e:
        print(f"  Error: {str(e)[:30]}", flush=True)
    return None

def save(data, filename):
    with open(f"{OUTPUT_DIR}/{filename}", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    print("加载人类文本...", flush=True)
    human_df = pd.read_csv("datasets/final_clean/all_human.csv")
    humans = human_df["text"].sample(300).tolist()
    
    c3_results, c4_results = [], []
    
    # C3: AI生成+改写 (100条)
    print("\n=== C3: AI生成+改写 ===", flush=True)
    for i in range(100):
        topic = random.choice(TOPICS)
        m1, m2 = random.sample(MODELS, 2)
        
        text = call("hotaru", m1, f"写一段关于{topic}的短文，150字左右。")
        if not text:
            time.sleep(8)
            continue
        
        edited = call("hotaru", m2, f"改写这段话使其更口语化自然：\n{text}")
        if edited and len(edited) > 50:
            c3_results.append({
                "text_id": hashlib.md5(edited.encode()).hexdigest()[:12],
                "text": edited, "category": "C3", "label": 1,
                "source_model": f"{m1}+{m2}", "original": text
            })
            print(f"[C3 {len(c3_results)}/100]", flush=True)
            save(c3_results, "c3_edited.json")
        time.sleep(8)
    
    # C4: 人类+AI润色 (200条)
    print("\n=== C4: 人类+AI润色 ===", flush=True)
    for i, h in enumerate(humans[:200]):
        if len(h) < 50 or len(h) > 800:
            continue
        
        model = random.choice(MODELS)
        api = "hybgzs" if "gemini" in model else "hotaru"
        
        polished = call(api, model, f"润色这段文字使其更通顺，保持原意：\n{h}")
        if polished and len(polished) > 30:
            c4_results.append({
                "text_id": hashlib.md5(polished.encode()).hexdigest()[:12],
                "text": polished, "category": "C4", "label": 1,
                "source_model": model, "original_human": h
            })
            print(f"[C4 {len(c4_results)}/200]", flush=True)
            save(c4_results, "c4_polished.json")
        time.sleep(8)
    
    print(f"\n完成! C3:{len(c3_results)} C4:{len(c4_results)}", flush=True)

if __name__ == "__main__":
    main()
