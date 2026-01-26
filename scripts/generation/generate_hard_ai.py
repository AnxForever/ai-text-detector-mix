#!/usr/bin/env python3
"""生成困难AI样本：模仿人类风格"""
import os, json, random
import pandas as pd
from openai import OpenAI

client = OpenAI(
    api_key="sk-WPzv2WwpDbnLp02uro0DyPUy0LyI3VjIRmngMj8fm7BLQqSq",
    base_url="https://api.hotaruapi.top/v1"
)
MODEL = "deepseek-ai/deepseek-v3.1"

# 从人类文本中采样作为风格参考
human_df = pd.read_csv('datasets/final_clean/all_human.csv')
human_samples = human_df.sample(50)['text'].tolist()

PROMPTS = [
    "用最口语化、随意的方式回答，像在微信聊天一样，可以有错别字和语气词：",
    "假装你是一个普通网友在论坛回帖，语气随意，不要太正式：",
    "模仿下面这段人类写的文字风格，写一段类似主题的内容：\n{human_text}\n\n现在用同样风格写：",
    "写得像日记一样，有个人情感和口语化表达，不要用任何总结性词汇：",
    "像知乎上的普通回答一样写，可以有主观看法和吐槽：",
]

TOPICS = [
    "今天的天气怎么样",
    "推荐一部最近看的电影",
    "聊聊你对加班的看法", 
    "分享一个生活小技巧",
    "说说你最喜欢的食物",
    "谈谈对手机依赖的感受",
    "讲一个有趣的经历",
    "对网购的看法",
    "周末一般怎么过",
    "对当代年轻人的看法",
]

def generate_hard_sample(topic, style_prompt, human_ref=None):
    prompt = style_prompt.format(human_text=human_ref) if human_ref else style_prompt
    prompt += f"\n话题：{topic}"
    
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.9,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    results = []
    n_samples = 200  # 生成200条困难样本
    
    for i in range(n_samples):
        topic = random.choice(TOPICS)
        prompt = random.choice(PROMPTS)
        human_ref = random.choice(human_samples) if '{human_text}' in prompt else None
        
        text = generate_hard_sample(topic, prompt, human_ref)
        if text and len(text) > 20:
            results.append({'text': text, 'label': 1, 'type': 'hard_ai'})
            print(f"[{i+1}/{n_samples}] {text[:50]}...")
    
    df = pd.DataFrame(results)
    df.to_csv('datasets/hard_test/hard_ai_samples.csv', index=False)
    print(f"\n生成 {len(df)} 条困难AI样本")

if __name__ == '__main__':
    os.makedirs('datasets/hard_test', exist_ok=True)
    main()
