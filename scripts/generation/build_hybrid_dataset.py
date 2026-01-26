#!/usr/bin/env python3
"""
混合文本数据集构建脚本
按Gemini报告的SinHAC方案构建C1-C4类数据
"""
import os, json, time, random, hashlib
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# API配置
APIS = [
    {"name": "hybgzs", "url": "https://ai.hybgzs.com/v1", "key": "sk-LQO5p6niHb31gBt_2T_yqGMgiKwgnm9b2os3QCgjuRphSTITFlsnBWFBGCI", 
     "models": ["gemini-3-pro-preview", "gemini-3-flash-preview", "gpt-4o-mini"], "rpm": 5},
    {"name": "hotaruapi", "url": "https://api.hotaruapi.top/v1", "key": "sk-WPzv2WwpDbnLp02uro0DyPUy0LyI3VjIRmngMj8fm7BLQqSq",
     "models": ["gemini-3-pro-preview", "gpt-4.1-mini", "qwen-3-235b-a22b-instruct-2507", "llama-3.3-70b"], "rpm": 10},
    {"name": "daiju", "url": "https://api.daiju.live/v1", "key": "sk-p14OddEwPKmsWMVkBsmckJrKnMQRo8xSlzOhNcYmAtZ5JSbO",
     "models": ["DeepSeek-V3.1", "gpt-5", "Qwen3-235B", "llama-3.3-70b"], "rpm": 10},
]

OUTPUT_DIR = "datasets/hybrid"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def call_api(api, model, messages, max_tokens=500):
    """调用API"""
    try:
        r = requests.post(f"{api['url']}/chat/completions",
            headers={"Authorization": f"Bearer {api['key']}", "Content-Type": "application/json"},
            json={"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.8},
            timeout=90)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        return None
    except:
        return None

def gen_text_id(text):
    return hashlib.md5(text.encode()).hexdigest()[:12]

def load_human_texts():
    """加载人类文本作为素材"""
    df = pd.read_csv("datasets/final_clean/all_human.csv")
    return df["text"].tolist()

# ========== C1: 纯AI生成 ==========
C1_PROMPTS = [
    "请写一段关于{topic}的文章，200-300字。",
    "用通俗易懂的语言解释{topic}，200字左右。",
    "分享你对{topic}的看法，写成一段随笔。",
]
C1_TOPICS = ["人工智能的发展", "健康饮食", "城市生活", "环境保护", "教育改革", "网络安全", 
             "传统文化", "科技创新", "职场压力", "社交媒体"]

def generate_c1(api, model, n=10):
    """生成C1类：纯AI生成"""
    results = []
    for _ in range(n):
        prompt = random.choice(C1_PROMPTS).format(topic=random.choice(C1_TOPICS))
        text = call_api(api, model, [{"role": "user", "content": prompt}])
        if text and len(text) > 50:
            results.append({
                "text_id": gen_text_id(text),
                "text": text,
                "category": "C1",
                "label": 1,
                "source_model": model,
                "api": api["name"],
                "spans": [{"start": 0, "end": len(text), "label": "AI"}]
            })
        time.sleep(60 / api["rpm"])
    return results

# ========== C2: 人类开头+AI续写 ==========
def generate_c2(api, model, human_texts, n=10):
    """生成C2类：人类开头+AI续写"""
    results = []
    for _ in range(n):
        human = random.choice(human_texts)
        # 取人类文本前30-50%
        cut = random.randint(int(len(human)*0.3), int(len(human)*0.5))
        human_part = human[:cut]
        
        prompt = f"请续写以下文字，保持风格一致，续写150-250字：\n\n{human_part}"
        ai_part = call_api(api, model, [{"role": "user", "content": prompt}])
        
        if ai_part and len(ai_part) > 30:
            full_text = human_part + ai_part
            results.append({
                "text_id": gen_text_id(full_text),
                "text": full_text,
                "category": "C2",
                "label": 1,  # 混合文本标记为AI
                "source_model": model,
                "api": api["name"],
                "spans": [
                    {"start": 0, "end": len(human_part), "label": "Human"},
                    {"start": len(human_part), "end": len(full_text), "label": "AI"}
                ],
                "boundary": len(human_part)
            })
        time.sleep(60 / api["rpm"])
    return results

# ========== C3: AI生成+人类编辑（模拟） ==========
def generate_c3(api, model, n=10):
    """生成C3类：AI生成后模拟人类编辑"""
    results = []
    for _ in range(n):
        # 先生成AI文本
        prompt = random.choice(C1_PROMPTS).format(topic=random.choice(C1_TOPICS))
        ai_text = call_api(api, model, [{"role": "user", "content": prompt}])
        
        if ai_text and len(ai_text) > 100:
            # 用另一个模型模拟人类编辑（改写部分内容）
            edit_prompt = f"请对以下文字进行轻微修改，改变其中2-3个句子的表达方式，使其更口语化，但保持原意：\n\n{ai_text}"
            edited = call_api(api, model, [{"role": "user", "content": edit_prompt}])
            
            if edited and len(edited) > 50:
                results.append({
                    "text_id": gen_text_id(edited),
                    "text": edited,
                    "category": "C3",
                    "label": 1,
                    "source_model": model,
                    "api": api["name"],
                    "original_ai": ai_text,
                    "spans": [{"start": 0, "end": len(edited), "label": "Mixed"}]
                })
        time.sleep(60 / api["rpm"])
    return results

# ========== C4: 人类写+AI润色 ==========
def generate_c4(api, model, human_texts, n=10):
    """生成C4类：人类原文+AI润色"""
    results = []
    for _ in range(n):
        human = random.choice(human_texts)
        if len(human) < 50:
            continue
            
        prompt = f"请润色以下文字，使其更通顺、更有文采，但保持原意不变：\n\n{human}"
        polished = call_api(api, model, [{"role": "user", "content": prompt}])
        
        if polished and len(polished) > 50:
            results.append({
                "text_id": gen_text_id(polished),
                "text": polished,
                "category": "C4",
                "label": 1,
                "source_model": model,
                "api": api["name"],
                "original_human": human,
                "spans": [{"start": 0, "end": len(polished), "label": "Polished"}]
            })
        time.sleep(60 / api["rpm"])
    return results

def main():
    print("=" * 50)
    print("混合文本数据集构建")
    print(f"开始时间: {datetime.now()}")
    print("=" * 50)
    
    human_texts = load_human_texts()
    print(f"加载人类文本: {len(human_texts)} 条")
    
    all_results = []
    
    # 每个API每个模型生成一定数量
    samples_per_model = 20  # 每个模型每类生成20条
    
    for api in APIS:
        print(f"\n>>> API: {api['name']}")
        for model in api["models"]:
            print(f"  模型: {model}")
            
            # C1: 纯AI
            c1 = generate_c1(api, model, samples_per_model)
            all_results.extend(c1)
            print(f"    C1(纯AI): {len(c1)}")
            
            # C2: 续写
            c2 = generate_c2(api, model, human_texts, samples_per_model)
            all_results.extend(c2)
            print(f"    C2(续写): {len(c2)}")
            
            # C3: 编辑
            c3 = generate_c3(api, model, samples_per_model // 2)  # 这个慢，减半
            all_results.extend(c3)
            print(f"    C3(编辑): {len(c3)}")
            
            # C4: 润色
            c4 = generate_c4(api, model, human_texts, samples_per_model)
            all_results.extend(c4)
            print(f"    C4(润色): {len(c4)}")
            
            # 中间保存
            df = pd.DataFrame(all_results)
            df.to_csv(f"{OUTPUT_DIR}/hybrid_dataset_progress.csv", index=False)
    
    # 添加人类文本作为对照
    human_samples = random.sample(human_texts, min(500, len(human_texts)))
    for h in human_samples:
        all_results.append({
            "text_id": gen_text_id(h),
            "text": h,
            "category": "Human",
            "label": 0,
            "source_model": "human",
            "api": "none",
            "spans": [{"start": 0, "end": len(h), "label": "Human"}]
        })
    
    # 保存最终结果
    df = pd.DataFrame(all_results)
    df.to_csv(f"{OUTPUT_DIR}/hybrid_dataset.csv", index=False)
    
    # 保存详细JSON（含spans）
    with open(f"{OUTPUT_DIR}/hybrid_dataset.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 50)
    print("生成完成!")
    print(f"总样本数: {len(all_results)}")
    print(f"类别分布:")
    for cat in df["category"].unique():
        print(f"  {cat}: {len(df[df['category']==cat])}")
    print(f"保存至: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
