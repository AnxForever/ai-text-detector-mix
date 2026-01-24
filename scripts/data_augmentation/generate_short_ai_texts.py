"""
生成短AI文本以解决长度偏差问题
使用项目的全部10个API端点，最大化模型多样性
"""
import sys
import os
import requests
import pandas as pd
import json
import time
import random
from datetime import datetime
from tqdm import tqdm

# ==================== 完整的10个API端点配置（最大化多样性）====================

# ==================== API endpoint configuration ====================

API_CONFIG_PATH = os.getenv("API_CONFIG_PATH", os.path.join("config", "api.txt"))

NAME_ALIASES = {
    "薄荷？": "bohe",
    "薄荷": "bohe"
}

API_ENDPOINTS = [

    {
        "name": "wong",
        "models": ["deepseek-v3.2-chat", "qwen-max-latest", "claude-sonnet-4-5"],
        "weight": 3
    },
    {
        "name": "xiaoyo",
        "models": ["deepseek-v3.2", "Kimi-K2"],
        "weight": 3
    },
    {
        "name": "fovt",
        "models": ["deepseek-ai/DeepSeek-V3", "gpt-4.1-mini"],
        "weight": 2
    },
    {
        "name": "liuge",
        "models": ["deepseek/deepseek-chat-v3-0324", "qwen/qwen-2.5-72b-instruct"],
        "weight": 2
    },
    {
        "name": "kfc",
        "models": ["Qwen3-235B-A22B-Instruct", "cursor2-gpt-5"],
        "weight": 2
    },


    {
        "name": "paolu",
        "models": ["claude-sonnet-4-5-20250929", "claude-haiku-4-5-20251001"],
        "weight": 1
    },
    {
        "name": "b4u",
        "models": ["claude-4.5-sonnet", "claude-4-sonnet"],
        "weight": 1
    },
    {
        "name": "coderouter",
        "models": ["claude-3-7-sonnet-20250219", "claude-haiku-4-5-20251001"],
        "weight": 1
    },
    {
        "name": "xiaodai",
        "models": ["anthropic/claude-3.7-sonnet", "qwen/qwen-2.5-72b-instruct"],  # Claude + Qwen
        "weight": 1
    },


    {
        "name": "bohe",
        "models": ["gpt-4.1-mini", "gemini-2.5-flash"],  # GPT + Gemini
        "weight": 1
    },
    {
        "name": "hybgzs_gemini",
        "models": ["gemini-2.5-flash"],
        "weight": 1
    },
    {
        "name": "hybgzs_gpt",
        "models": ["gpt-4.1-mini", "gpt-4o-mini"],
        "weight": 1
    },
]


def _split_kv(line: str) -> str:
    if ":" in line:
        return line.split(":", 1)[1].strip()
    if "：" in line:
        return line.split("：", 1)[1].strip()
    return ""


def _normalize_name(name: str) -> str:
    raw = name.strip()
    if not raw:
        return ""
    return NAME_ALIASES.get(raw, raw).strip().lower()


def load_api_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        print(f"API config not found: {config_path}")
        return {}

    entries = []
    current = {}
    with open(config_path, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                if current:
                    entries.append(current)
                    current = {}
                continue

            lower = line.lower()
            if lower.startswith("key"):
                current["key"] = _split_kv(line)
                continue
            if lower.startswith("url"):
                current["base_url"] = _split_kv(line)
                continue

            if current:
                entries.append(current)
                current = {}
            current["name"] = line

    if current:
        entries.append(current)

    result = {}
    for entry in entries:
        name = _normalize_name(entry.get("name", ""))
        if not name:
            continue
        key = entry.get("key", "").strip()
        base_url = entry.get("base_url", "").strip()
        if key and base_url:
            result[name] = {"key": key, "base_url": base_url}

    return result


def build_api_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if "/chat/completions" in url:
        return url
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return f"{url}/chat/completions"


def resolve_api_endpoints() -> list:
    api_config = load_api_config(API_CONFIG_PATH)
    endpoints = []
    missing = []

    for endpoint in API_ENDPOINTS:
        name = endpoint["name"].lower()
        cfg = api_config.get(name)
        if not cfg:
            missing.append(endpoint["name"])
            continue

        merged = dict(endpoint)
        merged["key"] = cfg["key"]
        merged["base_url"] = cfg["base_url"]
        merged["url"] = build_api_url(cfg["base_url"])
        endpoints.append(merged)

    if missing:
        print(f"?? ?? API ???{', '.join(missing)}")

    return endpoints


ACTIVE_ENDPOINTS = resolve_api_endpoints()

API_POOL = []
for config in ACTIVE_ENDPOINTS:
    API_POOL.extend([config] * config['weight'])

# 统计模型多样性
ALL_MODELS = []
for config in ACTIVE_ENDPOINTS:
    ALL_MODELS.extend(config['models'])
UNIQUE_MODELS = set(ALL_MODELS)

print(f"\n✅ API配置加载完成:")
print(f"  - API端点数: {len(ACTIVE_ENDPOINTS)}")
print(f"  - 模型总数: {len(ALL_MODELS)}")
print(f"  - 独特模型数: {len(UNIQUE_MODELS)}")
print(f"  - 加权池大小: {len(API_POOL)}")

# ==================== 短文本生成配置 ====================

SHORT_PROMPTS = [
    "用100-300字简要解释：{topic}",
    "简洁概括：{topic}的核心要点",
    "用一段话（200字左右）说明：{topic}",
    "简答：{topic}是什么？为什么重要？",
    "用简短的文字介绍：{topic}",
    "一句话总结：{topic}的关键特征",
    "简要说明：{topic}的主要应用",
    "用200字左右描述：{topic}的基本原理",
    "快速了解：{topic}的核心内容",
    "精简回答：关于{topic}你需要知道的事",
    "用几句话解释：{topic}",
    "简明扼要地说明：{topic}是如何工作的",
    "用一个简短段落介绍：{topic}"
]

TOPICS = [
    # 科技类 (40个)
    "人工智能", "机器学习", "深度学习", "自然语言处理", "计算机视觉",
    "区块链技术", "量子计算", "云计算", "大数据分析", "物联网",
    "5G通信", "边缘计算", "神经网络", "强化学习", "迁移学习",
    "联邦学习", "生成对抗网络", "Transformer模型", "BERT模型", "GPT模型",
    "自动驾驶", "机器人技术", "无人机", "虚拟现实", "增强现实",
    "混合现实", "数字孪生", "元宇宙", "Web3.0", "智能合约",
    "分布式系统", "微服务架构", "容器化技术", "Docker", "Kubernetes",
    "DevOps", "持续集成", "持续部署", "敏捷开发", "Scrum框架",

    # 商业经济 (30个)
    "商业模式", "创业投资", "风险投资", "天使投资", "私募股权",
    "IPO上市", "并购重组", "股权激励", "期权制度", "股票市场",
    "债券市场", "外汇交易", "期货期权", "衍生品", "对冲基金",
    "资产配置", "投资组合", "价值投资", "成长投资", "指数基金",
    "ETF基金", "REIT", "供应链管理", "库存管理", "精益生产",
    "六西格玛", "品牌营销", "数字营销", "社交媒体营销", "内容营销",

    # 社会文化 (30个)
    "可持续发展", "碳中和", "碳达峰", "绿色能源", "新能源汽车",
    "太阳能发电", "风力发电", "氢能源", "核聚变", "智慧城市",
    "智能交通", "共享经济", "平台经济", "零工经济", "远程办公",
    "在线教育", "MOOC", "终身学习", "职业教育", "素质教育",
    "教育公平", "医疗改革", "分级诊疗", "远程医疗", "精准医疗",
    "基因编辑", "CRISPR技术", "细胞治疗", "免疫疗法", "个性化医疗",

    # 科学研究 (30个)
    "量子纠缠", "暗物质", "暗能量", "黑洞", "引力波",
    "弦理论", "标准模型", "粒子物理", "核物理", "凝聚态物理",
    "纳米技术", "纳米材料", "石墨烯", "超导材料", "智能材料",
    "生物材料", "复合材料", "3D打印", "增材制造", "减材制造",
    "基因组学", "蛋白质组学", "代谢组学", "合成生物学", "系统生物学",
    "干细胞", "再生医学", "组织工程", "生物信息学", "计算生物学",

    # 人文艺术 (30个)
    "行为经济学", "认知心理学", "发展心理学", "社会心理学", "积极心理学",
    "人格理论", "动机理论", "情绪管理", "压力管理", "时间管理",
    "领导力", "团队协作", "沟通技巧", "批判性思维", "创造性思维",
    "设计思维", "用户体验", "用户界面", "交互设计", "视觉设计",
    "品牌设计", "平面设计", "工业设计", "建筑设计", "室内设计",
    "景观设计", "服装设计", "游戏设计", "音乐理论", "美术理论",

    # 新兴领域 (30个)
    "脑机接口", "神经科学", "认知科学", "情感计算", "人机交互",
    "语音识别", "语音合成", "机器翻译", "问答系统", "对话系统",
    "推荐系统", "搜索引擎", "知识图谱", "知识推理", "因果推断",
    "迁移学习", "元学习", "少样本学习", "零样本学习", "自监督学习",
    "对比学习", "多模态学习", "跨模态学习", "图神经网络", "时序预测",
    "异常检测", "聚类分析", "降维技术", "特征工程", "模型解释性"
]

def call_api(api_config: dict, prompt: str, max_tokens: int = 300, temperature: float = 0.7) -> tuple:
    """调用单个API端点生成文本"""
    headers = {
        "Authorization": f"Bearer {api_config['key']}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    model = random.choice(api_config['models'])
    url = api_config['url']

    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    try:
        resp = requests.post(url, headers=headers, json=data, timeout=60)
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"].strip()
            return content, api_config['name'], model
        else:
            return None, api_config['name'], f"HTTP {resp.status_code}"
    except Exception as e:
        return None, api_config['name'], str(e)[:50]

def assess_text_quality(text: str, target_min: int = 300, target_max: int = 600) -> dict:
    """评估文本质量"""
    length = len(text)
    if target_min <= length <= target_max:
        length_score = 1.0
    elif length < target_min:
        length_score = length / target_min
    else:
        length_score = max(0, 1.0 - (length - target_max) / target_max)

    has_content = (
        len(text) > 50 and
        not text.startswith("抱歉") and
        not text.startswith("对不起") and
        "无法" not in text[:50] and
        "不能" not in text[:50]
    )
    content_score = 1.0 if has_content else 0.0
    quality_score = 0.7 * length_score + 0.3 * content_score

    return {
        'length': length,
        'in_range': target_min <= length <= target_max,
        'has_content': has_content,
        'quality_score': quality_score
    }

def generate_short_ai_texts(
    target_count: int = 3000,
    target_min: int = 300,
    target_max: int = 600,
    quality_threshold: float = 0.7,
    output_file: str = 'datasets/short_ai_texts/generated_short_texts.csv',
    metadata_file: str = 'datasets/short_ai_texts/generated_metadata.json'
):
    """生成短AI文本数据集"""
    print("\n开始生成短AI文本数据集...\n")
    print("="*70)
    print("短AI文本生成器 - 最大化模型多样性")
    print("="*70)
    print(f"目标数量：{target_count} 条")
    print(f"长度范围：{target_min}-{target_max} 字符")
    print(f"质量阈值：{quality_threshold}")
    print(f"提示词模板：{len(SHORT_PROMPTS)} 种")
    print(f"话题数量：{len(TOPICS)} 个")
    print(f"API端点数：{len(ACTIVE_ENDPOINTS)} 个")
    print(f"独特模型数：{len(UNIQUE_MODELS)} 个")
    print(f"总API池大小：{len(API_POOL)} (加权)")
    print("\n模型家族覆盖:")
    model_families = {
        'DeepSeek': len([m for m in UNIQUE_MODELS if 'deepseek' in m.lower()]),
        'Claude': len([m for m in UNIQUE_MODELS if 'claude' in m.lower()]),
        'Qwen': len([m for m in UNIQUE_MODELS if 'qwen' in m.lower()]),
        'GPT': len([m for m in UNIQUE_MODELS if 'gpt' in m.lower()]),
        'Kimi': len([m for m in UNIQUE_MODELS if 'kimi' in m.lower()]),
        'Gemini': len([m for m in UNIQUE_MODELS if 'gemini' in m.lower()]),
        'Cursor': len([m for m in UNIQUE_MODELS if 'cursor' in m.lower()]),
    }
    for family, count in model_families.items():
        if count > 0:
            print(f"  ✓ {family}: {count} 个变体")
    print("="*70)

    if not API_POOL:
        print("❌ 未加载可用 API 端点，请检查 config/api.txt 配置。")
        return None, {}

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    generated_texts = []
    failed_count = 0
    api_usage = {config['name']: 0 for config in ACTIVE_ENDPOINTS}
    model_usage = {}

    with tqdm(total=target_count, desc="生成进度") as pbar:
        while len(generated_texts) < target_count:
            api_config = random.choice(API_POOL)
            prompt_template = random.choice(SHORT_PROMPTS)
            topic = random.choice(TOPICS)
            prompt = prompt_template.format(topic=topic)

            text, api_name, model_or_error = call_api(api_config, prompt, max_tokens=300, temperature=0.7)

            if text is None:
                failed_count += 1
                time.sleep(0.5)
                continue

            quality = assess_text_quality(text, target_min, target_max)

            if quality['quality_score'] >= quality_threshold:
                generated_texts.append({
                    'text': text,
                    'length': quality['length'],
                    'topic': topic,
                    'api': api_name,
                    'model': model_or_error,
                    'prompt_template': prompt_template,
                    'quality_score': quality['quality_score'],
                    'timestamp': datetime.now().isoformat()
                })

                api_usage[api_name] += 1
                model_usage[model_or_error] = model_usage.get(model_or_error, 0) + 1
                pbar.update(1)
            else:
                failed_count += 1

            time.sleep(0.15)

    df = pd.DataFrame(generated_texts)
    df[['text', 'length']].to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✓ 数据集已保存: {output_file}")

    metadata = {
        "name": "生成的短AI文本数据集",
        "created_at": datetime.now().isoformat(),
        "total_samples": len(df),
        "target_count": target_count,
        "failed_count": failed_count,
        "success_rate": len(df) / (len(df) + failed_count) if (len(df) + failed_count) > 0 else 0,
        "avg_length": float(df['length'].mean()),
        "min_length": int(df['length'].min()),
        "max_length": int(df['length'].max()),
        "length_std": float(df['length'].std()),
        "quality_threshold": quality_threshold,
        "avg_quality_score": float(df['quality_score'].mean()),
        "api_distribution": api_usage,
        "model_distribution": model_usage,
        "unique_models_used": len(set(df['model'])),
        "api_configs_used": len(ACTIVE_ENDPOINTS),
        "prompt_templates_count": len(SHORT_PROMPTS),
        "topics_count": len(TOPICS)
    }

    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"✓ 元数据已保存: {metadata_file}")
    print("\n" + "="*70)
    print("生成完成！统计信息：")
    print("="*70)
    print(f"  成功生成: {len(df)} 条")
    print(f"  失败次数: {failed_count} 次")
    print(f"  成功率: {metadata['success_rate']*100:.1f}%")
    print(f"  实际使用模型数: {metadata['unique_models_used']} / {len(UNIQUE_MODELS)} 个")
    print(f"  平均长度: {metadata['avg_length']:.0f} 字符")
    print(f"  长度范围: {metadata['min_length']}-{metadata['max_length']} 字符")
    print(f"  平均质量分: {metadata['avg_quality_score']:.3f}")

    print("\nAPI使用分布:")
    for api_name, count in sorted(api_usage.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            print(f"    {api_name:12s}: {count:4d} 条 ({count/len(df)*100:5.1f}%)")

    print("\n模型多样性分布 (Top 15):")
    top_models = sorted(model_usage.items(), key=lambda x: x[1], reverse=True)[:15]
    for model, count in top_models:
        print(f"    {model[:45]:45s}: {count:4d} 条 ({count/len(df)*100:5.1f}%)")

    print("="*70)

    return df, metadata

def main():
    """主函数"""
    df, metadata = generate_short_ai_texts(
        target_count=3000,
        target_min=300,
        target_max=600,
        quality_threshold=0.7,
        output_file='datasets/short_ai_texts/generated_short_texts.csv',
        metadata_file='datasets/short_ai_texts/generated_metadata.json'
    )

    print("\n✅ 短AI文本生成完成！")
    print("\n下一步:")
    print("  1. 检查生成的文本质量")
    print("  2. 运行 rebuild_balanced_dataset.py 合并所有数据")
    print("  3. 运行 split_dataset.py 重新划分数据集")

if __name__ == '__main__':
    main()
