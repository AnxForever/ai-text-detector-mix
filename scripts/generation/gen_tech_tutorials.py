#!/usr/bin/env python3
"""
生成AI技术教程数据
用于平衡Wikipedia百科风格的Human数据
"""
import json
import os
import random
import time
import csv
from pathlib import Path
from openai import OpenAI

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 技术主题列表
TECH_TOPICS = [
    # 深度学习
    "梯度消失问题及解决方法",
    "梯度爆炸的原因和应对策略",
    "过拟合的表现和预防措施",
    "欠拟合的诊断与改进",
    "Batch Normalization的原理和作用",
    "Dropout正则化的工作机制",
    "学习率调度策略对比",
    "权重初始化方法的选择",
    "激活函数的对比分析",
    "损失函数的选择指南",

    # 自然语言处理
    "BERT模型的预训练原理",
    "Transformer的自注意力机制",
    "Word2Vec词向量的训练方法",
    "文本分类的常见算法对比",
    "命名实体识别的实现方法",
    "情感分析的技术路线",
    "机器翻译的编码器-解码器架构",
    "文本生成的采样策略",

    # 计算机视觉
    "卷积神经网络的基本原理",
    "池化层的作用和类型",
    "目标检测算法的发展历程",
    "图像分割的主流方法",
    "数据增强的常用技术",
    "迁移学习的应用场景",

    # 编程基础
    "Python装饰器的原理和用法",
    "多线程与多进程的区别",
    "内存泄漏的排查方法",
    "数据库索引的优化策略",
    "缓存机制的设计原则",
    "API设计的最佳实践",
    "Git分支管理策略",
    "单元测试的编写规范",

    # 算法
    "时间复杂度的分析方法",
    "动态规划的解题思路",
    "贪心算法的适用条件",
    "二分查找的变体应用",
    "排序算法的性能对比",
    "哈希表的冲突处理",
]

PROMPT_TEMPLATE = """请用中文详细解释以下技术概念，要求：
1. 先给出定义和背景
2. 解释为什么会出现这个问题/为什么需要这个技术
3. 列出常见表现或特点
4. 给出具体的解决方法或最佳实践
5. 语言风格：技术性强但通俗易懂，像是一个资深工程师在给初学者讲解

主题：{topic}

请直接开始讲解，不要有开场白或总结。"""


def load_api_config():
    """加载API配置"""
    config_path = PROJECT_ROOT / "config" / "api.local.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_tutorial(client, topic, model="deepseek-chat"):
    """生成单个技术教程"""
    prompt = PROMPT_TEMPLATE.format(topic=topic)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [ERROR] {topic}: {e}")
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=50, help='生成数量')
    parser.add_argument('--output', type=str, default='ai_tech_tutorials', help='输出文件名')
    parser.add_argument('--proxy', type=str, default='remote_proxy', help='使用的API代理')
    args = parser.parse_args()

    print("=" * 60)
    print("AI技术教程生成")
    print("=" * 60)

    # 加载API配置
    config = load_api_config()
    proxy_config = config.get(args.proxy, {})

    if not proxy_config:
        print(f"[ERROR] 找不到代理配置: {args.proxy}")
        return 1

    print(f"使用代理: {args.proxy}")
    print(f"API地址: {proxy_config['base_url']}")

    client = OpenAI(
        base_url=proxy_config['base_url'],
        api_key=proxy_config['api_key'],
    )

    # 随机选择主题
    random.seed(42)
    topics = random.sample(TECH_TOPICS, min(args.count, len(TECH_TOPICS)))

    # 如果需要更多，重复列表
    while len(topics) < args.count:
        topics.extend(random.sample(TECH_TOPICS, min(args.count - len(topics), len(TECH_TOPICS))))

    print(f"\n将生成 {len(topics)} 个技术教程")

    # 生成
    samples = []
    output_dir = PROJECT_ROOT / "datasets" / "generated_tutorials"
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, topic in enumerate(topics):
        print(f"\n[{i+1}/{len(topics)}] 生成: {topic}")

        text = generate_tutorial(client, topic)
        if text and len(text) > 100:
            samples.append({
                'text': text,
                'label': 1,  # AI
                'source': 'AI_Tutorial_DeepSeek',
                'genre': '技术教程',
                'topic': topic,
            })
            print(f"  成功! 长度: {len(text)}")
        else:
            print(f"  失败或太短")

        # 避免速率限制
        time.sleep(1)

    # 保存
    output_path = output_dir / f"{args.output}.csv"
    with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['text', 'label', 'source', 'genre', 'topic'])
        writer.writeheader()
        writer.writerows(samples)

    print(f"\n已保存 {len(samples)} 条到 {output_path}")

    # 同时保存JSONL
    jsonl_path = output_dir / f"{args.output}.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')

    return 0


if __name__ == "__main__":
    exit(main())
