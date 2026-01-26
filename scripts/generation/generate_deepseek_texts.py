#!/usr/bin/env python3
"""使用DeepSeek API生成AI文本数据集"""

import requests
import json
import time
from pathlib import Path

API_KEY = "sk-WPzv2WwpDbnLp02uro0DyPUy0LyI3VjIRmngMj8fm7BLQqSq"
API_BASE = "https://api.hotaruapi.top/v1"
MODEL = "deepseek-ai/deepseek-v3.1"

# 生成主题列表
TOPICS = [
    "人工智能的发展历程",
    "量子计算的原理与应用",
    "区块链技术在金融领域的应用",
    "5G通信技术的特点",
    "新能源汽车的发展趋势",
    "基因编辑技术的伦理问题",
    "虚拟现实技术的应用场景",
    "云计算与边缘计算的区别",
    "物联网技术的安全挑战",
    "机器学习在医疗诊断中的应用",
    "自动驾驶技术的发展现状",
    "大数据分析在商业决策中的作用",
    "网络安全的重要性",
    "智能家居的发展前景",
    "可再生能源的利用",
    "太空探索的意义",
    "生物技术在农业中的应用",
    "教育信息化的发展",
    "远程医疗的优势与挑战",
    "数字货币的未来"
]

def generate_text(topic, length_range="350-500"):
    """生成单个文本"""
    prompt = f"请用{length_range}字纯文本（无格式）介绍{topic}。不要用markdown、列表、加粗等格式，像写普通文章一样。"
    
    try:
        response = requests.post(
            f"{API_BASE}/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}"
            },
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 800,
                "temperature": 0.8
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            text = result["choices"][0]["message"]["content"].strip()
            return text
        else:
            print(f"错误: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"生成失败: {e}")
        return None

def main():
    output_dir = Path("datasets/deepseek_generated")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "ai_texts.jsonl"
    
    print(f"开始生成AI文本数据集...")
    print(f"模型: {MODEL}")
    print(f"主题数量: {len(TOPICS)}")
    print(f"输出文件: {output_file}")
    print("-" * 50)
    
    generated = 0
    failed = 0
    
    with open(output_file, "w", encoding="utf-8") as f:
        for i, topic in enumerate(TOPICS, 1):
            print(f"[{i}/{len(TOPICS)}] 生成: {topic}")
            
            text = generate_text(topic)
            
            if text:
                data = {
                    "text": text,
                    "label": "AI",
                    "source": "deepseek-v3.1",
                    "topic": topic,
                    "length": len(text)
                }
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
                f.flush()
                generated += 1
                print(f"  ✓ 成功 (长度: {len(text)})")
            else:
                failed += 1
                print(f"  ✗ 失败")
            
            # 避免请求过快
            time.sleep(1)
    
    print("-" * 50)
    print(f"生成完成!")
    print(f"成功: {generated}")
    print(f"失败: {failed}")
    print(f"文件: {output_file}")

if __name__ == "__main__":
    main()
