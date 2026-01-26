#!/usr/bin/env python3
"""C2: 人类开头+AI续写 (多模型)"""
import os, json, time, random, hashlib
import pandas as pd
import requests

APIS = [
    {"url": "https://api.hotaruapi.top/v1", "key": "sk-WPzv2WwpDbnLp02uro0DyPUy0LyI3VjIRmngMj8fm7BLQqSq", "models": ["gpt-4.1-mini", "qwen-3-235b-a22b-instruct-2507", "llama-3.3-70b"]},
    {"url": "https://api.daiju.live/v1", "key": "sk-p14OddEwPKmsWMVkBsmckJrKnMQRo8xSlzOhNcYmAtZ5JSbO", "models": ["DeepSeek-V3.1", "gpt-5", "Qwen3-235B"]},
]
OUTPUT = "datasets/hybrid/c2_continuation.json"
os.makedirs("datasets/hybrid", exist_ok=True)

def call_api(api, model, prompt):
    try:
        r = requests.post(f"{api['url']}/chat/completions",
            headers={"Authorization": f"Bearer {api['key']}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 400, "temperature": 0.8},
            timeout=90)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
    except:
        pass
    return None

def main():
    human_df = pd.read_csv("datasets/final_clean/all_human.csv")
    humans = human_df["text"].sample(400).tolist()
    
    results = []
    idx = 0
    for api in APIS:
        for model in api["models"]:
            for _ in range(50):  # 每模型50条
                if idx >= len(humans):
                    break
                h = humans[idx]
                idx += 1
                if len(h) < 80:
                    continue
                
                cut = random.randint(int(len(h)*0.3), int(len(h)*0.5))
                human_part = h[:cut]
                prompt = f"请续写以下文字，保持风格一致，续写100-200字：\n\n{human_part}"
                ai_part = call_api(api, model, prompt)
                
                if ai_part and len(ai_part) > 30:
                    full = human_part + ai_part
                    results.append({
                        "text_id": hashlib.md5(full.encode()).hexdigest()[:12],
                        "text": full,
                        "category": "C2",
                        "label": 1,
                        "source_model": model,
                        "boundary": len(human_part),
                        "spans": [
                            {"start": 0, "end": len(human_part), "label": "Human"},
                            {"start": len(human_part), "end": len(full), "label": "AI"}
                        ]
                    })
                    print(f"[{len(results)}/300] C2 {model[:15]}")
                
                if len(results) >= 300:
                    break
                time.sleep(6)
            if len(results) >= 300:
                break
        if len(results) >= 300:
            break
    
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"C2完成: {len(results)}条 -> {OUTPUT}")

if __name__ == "__main__":
    main()
