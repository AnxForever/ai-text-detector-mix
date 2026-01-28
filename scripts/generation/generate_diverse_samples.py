#!/usr/bin/env python3
"""
多样化AI文本生成任务 - 基于Gemini研究方向
预计运行时间: 4-6小时
"""
import requests
import json
import time
import random
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from scripts.utils.api_config import get_proxy_config

# API配置 - WSL访问Windows服务
API = get_proxy_config("local_proxy", "LOCAL_PROXY", "http://192.168.60.105:8317/v1")
API_BASE = f"{API['url'].rstrip('/')}/chat/completions"
API_KEY = API["key"]
MODEL = "glm-4.7"

# 输出目录
OUTPUT_DIR = Path("datasets/logs/augmented_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 日志文件
LOG_FILE = OUTPUT_DIR / f"generation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

def log(message):
    """记录日志"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_msg + "\n")

def call_api(prompt, temperature=0.8, max_tokens=1000):
    """调用API生成文本"""
    try:
        response = requests.post(
            API_BASE,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=240  # 增加超时时间到240秒
        )
        
        if response.status_code == 200:
            result = response.json()
            # 处理不同的响应格式
            message = result['choices'][0]['message']
            content = message.get('content') or message.get('reasoning_content', '')
            return content.strip() if content else None
        else:
            log(f"API错误: {response.status_code}")
            return None
    except requests.exceptions.Timeout:
        log(f"请求超时")
        return None
    except requests.exceptions.JSONDecodeError as e:
        log(f"JSON解析错误: {str(e)[:100]}")
        return None
    except Exception as e:
        log(f"请求失败: {str(e)[:100]}")
        return None

# ============================================================================
# 任务1: 技术文档风格AI文本 (1000样本, ~1小时)
# ============================================================================
TECHNICAL_PROMPTS = [
    # API文档风格
    """生成一段API文档说明，描述一个RESTful接口的功能、参数和返回值。
要求：
- 使用技术术语
- 包含参数列表
- 格式规范
- 不要出现"作为AI"等口语化表达
- 200-400字""",
    
    # 技术规范
    """编写一段技术规范文档，说明某个功能模块的实现要求。
要求：
- 使用条目列表
- 包含技术指标
- 语言专业、简洁
- 不要对话式语气
- 150-300字""",
    
    # 算法说明
    """描述一个常见算法（如排序、搜索、优化）的工作原理和步骤。
要求：
- 分步骤说明
- 使用专业术语
- 包含时间复杂度分析
- 避免"我认为"等主观表达
- 200-400字""",
    
    # 系统架构
    """说明一个分布式系统的架构设计，包括各组件的职责和交互方式。
要求：
- 列举关键组件
- 说明数据流向
- 技术性强
- 客观描述
- 250-450字""",
    
    # 配置说明
    """编写一份配置文件的说明文档，解释各配置项的含义和取值范围。
要求：
- 逐项说明
- 包含默认值
- 注意事项
- 格式清晰
- 150-300字""",
]

def generate_technical_docs(count=1000):
    """生成技术文档风格样本"""
    log(f"开始生成技术文档样本，目标: {count}个")
    samples = []
    
    for i in range(count):
        prompt = random.choice(TECHNICAL_PROMPTS)
        text = call_api(prompt, temperature=0.7, max_tokens=600)
        
        if text:
            samples.append({
                "text": text.strip(),
                "label": 1,  # AI
                "style": "technical_doc",
                "source": "glm-4.7",
                "timestamp": datetime.now().isoformat()
            })
            
            if (i + 1) % 50 == 0:
                log(f"技术文档: {i+1}/{count} 完成")
                # 保存中间结果
                with open(OUTPUT_DIR / "technical_docs.jsonl", "a", encoding="utf-8") as f:
                    for s in samples[-50:]:
                        f.write(json.dumps(s, ensure_ascii=False) + "\n")
        
        time.sleep(5)  # 增加延迟到5秒，避免并发冲突
    
    log(f"技术文档生成完成: {len(samples)}个")
    return samples

# ============================================================================
# 任务2: 学术论文风格 (500样本, ~40分钟)
# ============================================================================
ACADEMIC_PROMPTS = [
    """生成一段学术论文的摘要，介绍某个研究的背景、方法和结论。
要求：
- 学术语言
- 包含研究方法
- 客观陈述
- 不要第一人称
- 200-350字""",
    
    """编写论文的方法部分，详细说明实验设计和数据收集过程。
要求：
- 严谨的学术表达
- 包含具体步骤
- 使用被动语态
- 避免口语化
- 250-400字""",
    
    """撰写文献综述的一段，总结某个领域的研究现状和发展趋势。
要求：
- 引用多个研究方向
- 客观分析
- 学术规范
- 逻辑清晰
- 200-400字""",
]

def generate_academic_texts(count=500):
    """生成学术论文风格样本"""
    log(f"开始生成学术论文样本，目标: {count}个")
    samples = []
    
    for i in range(count):
        prompt = random.choice(ACADEMIC_PROMPTS)
        text = call_api(prompt, temperature=0.6, max_tokens=600)
        
        if text:
            samples.append({
                "text": text.strip(),
                "label": 1,
                "style": "academic",
                "source": "glm-4.7",
                "timestamp": datetime.now().isoformat()
            })
            
            if (i + 1) % 50 == 0:
                log(f"学术论文: {i+1}/{count} 完成")
                with open(OUTPUT_DIR / "academic_texts.jsonl", "a", encoding="utf-8") as f:
                    for s in samples[-50:]:
                        f.write(json.dumps(s, ensure_ascii=False) + "\n")
        
        time.sleep(2)
    
    log(f"学术论文生成完成: {len(samples)}个")
    return samples

# ============================================================================
# 任务3: 列表式内容 (800样本, ~1小时)
# ============================================================================
LIST_PROMPTS = [
    """生成一份操作步骤说明，包含5-8个步骤。
要求：
- 使用序号或项目符号
- 每步简洁明确
- 包含注意事项
- 不要对话式开头
- 150-300字""",
    
    """列举某个主题的要点总结，包含多个条目。
要求：
- 条目式呈现
- 使用冒号、分号
- 信息密集
- 格式规范
- 200-350字""",
    
    """编写一份检查清单（Checklist），列出需要确认的事项。
要求：
- 清单格式
- 每项独立
- 可操作性强
- 简洁专业
- 150-250字""",
]

def generate_list_contents(count=800):
    """生成列表式内容样本"""
    log(f"开始生成列表式内容，目标: {count}个")
    samples = []
    
    for i in range(count):
        prompt = random.choice(LIST_PROMPTS)
        text = call_api(prompt, temperature=0.7, max_tokens=500)
        
        if text:
            samples.append({
                "text": text.strip(),
                "label": 1,
                "style": "list",
                "source": "glm-4.7",
                "timestamp": datetime.now().isoformat()
            })
            
            if (i + 1) % 50 == 0:
                log(f"列表式: {i+1}/{count} 完成")
                with open(OUTPUT_DIR / "list_contents.jsonl", "a", encoding="utf-8") as f:
                    for s in samples[-50:]:
                        f.write(json.dumps(s, ensure_ascii=False) + "\n")
        
        time.sleep(2)
    
    log(f"列表式内容生成完成: {len(samples)}个")
    return samples

# ============================================================================
# 任务4: 对抗样本 - 带错误的AI文本 (300样本, ~30分钟)
# ============================================================================
ADVERSARIAL_PROMPTS = [
    """生成一段技术说明文本，但故意包含1-2个拼写错误或语法不规范的地方，模拟人类写作的小瑕疵。
要求：
- 内容专业
- 包含轻微错误（错别字、标点不规范等）
- 不要太明显
- 200-350字""",
    
    """编写一段解释性文本，使用略显口语化的表达，但保持信息准确性。
要求：
- 混合正式和非正式语言
- 可以有语气词
- 逻辑清晰
- 150-300字""",
]

def generate_adversarial_samples(count=300):
    """生成对抗样本"""
    log(f"开始生成对抗样本，目标: {count}个")
    samples = []
    
    for i in range(count):
        prompt = random.choice(ADVERSARIAL_PROMPTS)
        text = call_api(prompt, temperature=0.9, max_tokens=500)
        
        if text:
            samples.append({
                "text": text.strip(),
                "label": 1,
                "style": "adversarial",
                "source": "glm-4.7",
                "timestamp": datetime.now().isoformat()
            })
            
            if (i + 1) % 50 == 0:
                log(f"对抗样本: {i+1}/{count} 完成")
                with open(OUTPUT_DIR / "adversarial_samples.jsonl", "a", encoding="utf-8") as f:
                    for s in samples[-50:]:
                        f.write(json.dumps(s, ensure_ascii=False) + "\n")
        
        time.sleep(2)
    
    log(f"对抗样本生成完成: {len(samples)}个")
    return samples

# ============================================================================
# 任务5: 领域特定文本 (400样本, ~40分钟)
# ============================================================================
DOMAIN_PROMPTS = [
    # 代码注释
    """生成一段Python代码的详细注释，解释函数的功能、参数和返回值。
要求：
- 代码注释风格
- 技术准确
- 简洁明了
- 100-200字""",
    
    # 产品说明
    """编写一份产品功能说明，介绍某个软件功能的使用方法。
要求：
- 用户视角
- 步骤清晰
- 专业但易懂
- 150-300字""",
    
    # 技术博客
    """撰写技术博客的一段，分享某个技术问题的解决方案。
要求：
- 问题-解决方案结构
- 包含代码或命令
- 实用性强
- 200-400字""",
]

def generate_domain_specific(count=400):
    """生成领域特定样本"""
    log(f"开始生成领域特定样本，目标: {count}个")
    samples = []
    
    for i in range(count):
        prompt = random.choice(DOMAIN_PROMPTS)
        text = call_api(prompt, temperature=0.7, max_tokens=600)
        
        if text:
            samples.append({
                "text": text.strip(),
                "label": 1,
                "style": "domain_specific",
                "source": "glm-4.7",
                "timestamp": datetime.now().isoformat()
            })
            
            if (i + 1) % 50 == 0:
                log(f"领域特定: {i+1}/{count} 完成")
                with open(OUTPUT_DIR / "domain_specific.jsonl", "a", encoding="utf-8") as f:
                    for s in samples[-50:]:
                        f.write(json.dumps(s, ensure_ascii=False) + "\n")
        
        time.sleep(2)
    
    log(f"领域特定样本生成完成: {len(samples)}个")
    return samples

# ============================================================================
# 主函数
# ============================================================================
def main():
    log("="*80)
    log("多样化AI文本生成任务启动")
    log("="*80)
    log(f"API: {API_BASE}")
    log(f"模型: {MODEL}")
    log(f"输出目录: {OUTPUT_DIR}")
    log("")
    
    start_time = time.time()
    
    # 执行各个任务
    tasks = [
        ("技术文档", generate_technical_docs, 2000),
        ("学术论文", generate_academic_texts, 1500),
        ("列表式内容", generate_list_contents, 2000),
        ("对抗样本", generate_adversarial_samples, 1000),
        ("领域特定", generate_domain_specific, 1500),
    ]
    
    total_samples = 0
    for task_name, task_func, count in tasks:
        log(f"\n{'='*80}")
        log(f"开始任务: {task_name}")
        log(f"{'='*80}")
        
        try:
            samples = task_func(count)
            total_samples += len(samples)
        except KeyboardInterrupt:
            log("\n任务被用户中断")
            break
        except Exception as e:
            log(f"任务失败: {e}")
            continue
    
    # 生成统计报告
    elapsed = time.time() - start_time
    log("")
    log("="*80)
    log("任务完成统计")
    log("="*80)
    log(f"总样本数: {total_samples}")
    log(f"总耗时: {elapsed/3600:.2f} 小时")
    log(f"平均速度: {total_samples/(elapsed/60):.1f} 样本/分钟")
    log(f"输出目录: {OUTPUT_DIR}")
    log("")
    log("生成的文件:")
    for f in OUTPUT_DIR.glob("*.jsonl"):
        lines = sum(1 for _ in open(f, encoding='utf-8'))
        log(f"  - {f.name}: {lines} 样本")

if __name__ == "__main__":
    main()
