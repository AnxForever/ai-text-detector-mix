#!/usr/bin/env python3
"""C3: AI生成+改写"""
import os, json, time, random, hashlib
import requests

API = {"url": "https://api.hotaruapi.top/v1", "key": "sk-WPzv2WwpDbnLp02uro0DyPUy0LyI3VjIRmngMj8fm7BLQqSq", "models": ["gemini-3-pro-preview", "gpt-4.1-mini"]}
OUTPUT = "datasets/hybrid/c3_edited.json"
TOPICS = ["人工智能", "健康生活", "城市发展", "环境保护", "教育问题", "网络文化", "传统节日", "科技创新", "职场经验", "旅行见闻"]
os.makedirs("datasets/hybrid", exist_ok=True)

def call_api(model, prompt):
    try:
        r = requests.post(f"{API['url']}/chat/completions",
            headers={"Authorization": f"Bearer {API['key']}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 500, "temperature": 0.8},
            timeout=90)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        print(f"API错误: {r.status_code}")
    except Exception as e:
        print(f"异常: {e}")
    return None

def main():
    import sys
    sys.stdout.flush()
    print("C3生成开始...", flush=True)
    results = []
    for i in range(120):
        topic = random.choice(TOPICS)
        model1, model2 = random.sample(API["models"], 2)
        
        # 先生成
        ai_text = call_api(model1, f"写一段关于{topic}的短文，200字左右。")
        if not ai_text or len(ai_text) < 50:
            time.sleep(6)
            continue
        
        # 再改写
        edited = call_api(model2, f"请改写以下文字，使表达更口语化、更自然，但保持原意：\n\n{ai_text}")
        if edited and len(edited) > 50:
            results.append({
                "text_id": hashlib.md5(edited.encode()).hexdigest()[:12],
                "text": edited,
                "category": "C3",
                "label": 1,
                "source_model": f"{model1}+{model2}",
                "original_ai": ai_text,
                "spans": [{"start": 0, "end": len(edited), "label": "Mixed"}]
            })
            print(f"[{len(results)}/100] C3")
        
        if len(results) >= 100:
            break
        time.sleep(6)
    
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"C3完成: {len(results)}条 -> {OUTPUT}")

if __name__ == "__main__":
    main()
