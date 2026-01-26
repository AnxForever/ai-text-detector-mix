#!/usr/bin/env python3
"""生成高难度AI文本测试集 - 模仿人类写作风格"""

import requests
import json
import time
from pathlib import Path

API_KEY = "sk-WPzv2WwpDbnLp02uro0DyPUy0LyI3VjIRmngMj8fm7BLQqSq"
API_BASE = "https://api.hotaruapi.top/v1"
MODEL = "deepseek-ai/deepseek-v3.1"

# 高难度提示词 - 模仿真实人类写作
PROMPTS = [
    {
        "topic": "人工智能发展",
        "prompt": """请以一个普通科技记者的口吻，写一篇400字左右的新闻报道，介绍人工智能的最新发展。

要求：
1. 使用真实新闻的写作风格，不要过于热情或夸张
2. 包含一些具体的数据和事实
3. 语言平实，避免使用明显的AI生成痕迹（如emoji、网络用语）
4. 不要使用markdown格式，纯文本即可
5. 模仿《人民日报》或《新华社》的新闻风格"""
    },
    {
        "topic": "经济分析",
        "prompt": """请以一个经济分析师的口吻，写一篇350-450字的经济评论，分析当前的经济形势。

要求：
1. 使用专业但不晦涩的语言
2. 包含一些经济术语和数据
3. 语气客观、理性，像真实的财经评论
4. 不要使用emoji、表情符号
5. 纯文本，无格式"""
    },
    {
        "topic": "教育观察",
        "prompt": """请以一个教育工作者的视角，写一篇400字左右的教育观察文章，讨论当前教育的某个问题。

要求：
1. 语言朴实，像真实的教师或教育工作者在写作
2. 可以有个人观点，但不要过于主观
3. 避免使用网络流行语和emoji
4. 纯文本，自然的段落划分
5. 模仿真实教育类文章的风格"""
    },
    {
        "topic": "社会评论",
        "prompt": """请写一篇350-450字的社会评论，讨论某个社会现象。

要求：
1. 以普通公民的视角，不要过于官方或学术
2. 语言自然，像真人在表达观点
3. 可以有情感，但不要过度煽情
4. 不使用markdown、emoji等格式
5. 纯文本，像在报纸上发表的评论文章"""
    },
    {
        "topic": "科技观察",
        "prompt": """请写一篇400字左右的科技观察文章，介绍某项新技术。

要求：
1. 以科技爱好者的口吻，不要过于专业或过于通俗
2. 语言流畅自然，像真人在分享见解
3. 避免使用"亲爱的读者"、"小伙伴们"等明显的AI痕迹
4. 纯文本，无格式
5. 模仿知乎或科技媒体的文章风格"""
    }
]

def generate_human_like_text(prompt_dict):
    """生成模仿人类的AI文本"""
    try:
        response = requests.post(
            f"{API_BASE}/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}"
            },
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt_dict["prompt"]}],
                "max_tokens": 800,
                "temperature": 0.7,  # 降低温度，更自然
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            text = result["choices"][0]["message"]["content"].strip()
            return text
        else:
            print(f"错误: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"生成失败: {e}")
        return None

def main():
    output_dir = Path("datasets/hard_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "human_like_ai_texts.jsonl"
    
    print("=" * 60)
    print("生成高难度AI文本测试集（模仿人类写作）")
    print("=" * 60)
    print(f"模型: {MODEL}")
    print(f"策略: 使用详细提示词约束AI模仿真实人类写作风格")
    print(f"输出: {output_file}")
    print("=" * 60)
    
    generated = 0
    
    with open(output_file, "w", encoding="utf-8") as f:
        # 每个提示词生成多个样本
        for round_num in range(1, 6):  # 5轮，每轮5个主题 = 25个样本
            print(f"\n第 {round_num} 轮生成:")
            for i, prompt_dict in enumerate(PROMPTS, 1):
                print(f"  [{i}/5] {prompt_dict['topic']}...", end=" ")
                
                text = generate_human_like_text(prompt_dict)
                
                if text:
                    data = {
                        "text": text,
                        "label": "AI",
                        "source": "deepseek-v3.1-human-like",
                        "topic": prompt_dict["topic"],
                        "difficulty": "hard",
                        "strategy": "detailed_prompt_constraint",
                        "length": len(text)
                    }
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")
                    f.flush()
                    generated += 1
                    print(f"✓ ({len(text)}字)")
                else:
                    print("✗")
                
                time.sleep(2)  # 避免请求过快
    
    print("\n" + "=" * 60)
    print(f"生成完成! 共 {generated} 个样本")
    print(f"文件: {output_file}")
    print("=" * 60)
    print("\n下一步:")
    print("1. 从人类数据集中随机抽取25个样本")
    print("2. 混合后让模型预测")
    print("3. 评估在高难度数据上的准确率")

if __name__ == "__main__":
    main()
