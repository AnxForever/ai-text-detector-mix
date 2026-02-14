#!/usr/bin/env python3
"""
P0 数据补全 - Prompt 批量生成器
从现有数据集抽取 topic，按三类模板生成 API 调用用的 prompt JSON。

输出：configs/p0_prompts.json
结构：[{id, category, template, topic, prompt, suggested_models}, ...]

用法：
    python scripts/generation/gen_p0_prompts.py
    python scripts/generation/gen_p0_prompts.py --per-template 80
"""

import csv
import json
import random
import re
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ============================================================
# Topic 抽取
# ============================================================

def extract_topics_from_dataset(path: Path, max_per_source: int = 200) -> dict[str, list[str]]:
    """从 human 样本中按 source 抽取 topic（取文本核心主题）"""
    source_texts: dict[str, list[str]] = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("label", "") != "0":
                continue
            text = row.get("text", "").strip()
            source = row.get("source", "unknown")
            if len(text) < 30:
                continue
            if source not in source_texts:
                source_texts[source] = []
            if len(source_texts[source]) < max_per_source:
                source_texts[source].append(text)
    return source_texts


def text_to_topic(text: str) -> str | None:
    """从文本中提取简短 topic，返回 None 表示质量不够"""
    # 取第一个句号/问号/感叹号前的内容
    match = re.match(r'^(.{10,80}?)[。？！\n]', text)
    topic = match.group(1).strip() if match else text[:50].strip()

    # 质量过滤
    if len(topic) < 15:
        return None
    # 截断不完整
    if topic.endswith(('，', '；', '、', ',')):
        return None
    # 英文占比过高（百科词条定义）
    eng_count = sum(1 for c in topic if c.isascii() and c.isalpha())
    if eng_count / max(len(topic), 1) > 0.35:
        return None
    # 标点开头
    if any(topic.startswith(c) for c in '，。、：-—'):
        return None
    # 含 URL
    if 'http' in topic or 'www' in topic:
        return None

    return topic


def build_topic_pool(source_texts: dict[str, list[str]]) -> dict[str, list[str]]:
    """构建分类 topic 池"""
    pool = {
        "general": [],      # 通用话题（知乎/聊天用）
        "academic": [],      # 学术/教育话题（学生作业用）
        "tech": [],          # 科技话题
        "news": [],          # 新闻话题
        "life": [],          # 生活话题（口语/犹豫用）
    }

    category_map = {
        "hc3_human": "general",
        "thucnews": "news",
        "external_m4_qazh": "general",
        "Wikipedia_CN": "academic",
        "Toutiao_news_tech": "tech",
        "Toutiao_news_finance": "news",
        "Toutiao_news_edu": "academic",
        "Review_ChnSentiCorp_htl_all": "life",
        "Review_waimai_10k": "life",
        "Toutiao_News": "news",
    }

    for source, texts in source_texts.items():
        cat = category_map.get(source, "general")
        topics = [text_to_topic(t) for t in texts]
        topics = [t for t in topics if t is not None]
        pool[cat].extend(topics)

    # 去重
    for cat in pool:
        pool[cat] = list(set(pool[cat]))
        random.shuffle(pool[cat])

    return pool


# ============================================================
# 手工补充 topic（覆盖更多场景）
# ============================================================

MANUAL_TOPICS = {
    "student_homework": [
        "分析《红楼梦》中林黛玉的人物性格",
        "谈谈你对人工智能发展的看法",
        "论述市场经济中政府调控的必要性",
        "简述细胞有丝分裂的过程和意义",
        "大学生应该注重成绩还是实践能力",
        "比较中西方教育理念的异同",
        "互联网对当代大学生社交方式的影响",
        "浅谈可持续发展的重要性",
        "分析当前大学生就业形势及应对策略",
        "论述法治社会建设的必要性和途径",
        "简述光合作用的基本原理及其影响因素",
        "分析《骆驼祥子》中祥子的悲剧命运",
        "谈谈对共享经济的看法",
        "论述大数据时代个人隐私保护的挑战",
        "简述中国改革开放以来的经济成就",
        "分析社交媒体对青少年心理健康的影响",
        "论述创新驱动发展战略的意义",
        "比较唐诗和宋词的艺术特点",
        "谈谈你对碳中和目标的理解",
        "分析当代大学生的消费观念",
    ],
    "casual_chat": [
        "为什么很多程序员都秃头",
        "大学食堂有哪些难以忘记的菜",
        "你见过最离谱的室友是什么样的",
        "考研还是找工作怎么选",
        "为什么现在的年轻人不想结婚了",
        "有哪些看似简单实则很难的事情",
        "你对加班文化怎么看",
        "第一次坐飞机是什么体验",
        "有什么相见恨晚的学习方法",
        "你最后悔没有早点知道的道理是什么",
        "养猫和养狗哪个更适合上班族",
        "你遇到过最尴尬的社死瞬间是什么",
        "有什么好用到哭的手机app推荐",
        "租房有哪些需要注意的坑",
        "你觉得AI会取代哪些职业",
        "大学四年最值得做的事是什么",
        "怎么克服拖延症",
        "外卖和自己做饭哪个更省钱",
        "有哪些让你觉得世界很小的经历",
        "你对35岁职场危机怎么看",
    ],
    "uncertain_topics": [
        "量子计算到底能不能破解现有的加密技术",
        "新能源车的电池回收问题严重吗",
        "中医到底有没有科学依据",
        "房价未来几年会涨还是会跌",
        "元宇宙到底是风口还是泡沫",
        "在家办公真的效率更高吗",
        "学历到底重不重要",
        "转基因食品安全吗",
        "人工智能会产生意识吗",
        "熬夜对身体的影响到底有多大",
        "股票型基金和债券型基金哪个更适合普通人",
        "短视频平台对青少年的影响是利大于弊吗",
        "双减政策实施后教育质量有提高吗",
        "创业和考公哪个更稳妥",
        "电子书会不会完全取代纸质书",
        "线上教育和线下教育哪个效果更好",
        "5G 到底改变了什么",
        "自动驾驶什么时候能真正普及",
        "打工人该不该裸辞",
        "冥想真的对心理健康有用吗",
    ],
}


# ============================================================
# Prompt 模板
# ============================================================

TEMPLATES = {
    # --- P0-1: 学生作业/论文风格 ---
    "student_homework_A": {
        "category": "student",
        "prompt": (
            "你是一个中国大学本科生，正在写课程作业。请就以下主题写一段200-400字的回答。\n"
            "要求：\n"
            "- 像真实学生一样写，不要太完美\n"
            "- 可以有一些口头化表达，比如「我觉得」「其实」「说实话」\n"
            "- 偶尔可以用一些不太精确的表述\n"
            "- 不要用「首先、其次、最后」这种明显的结构化表达\n"
            "- 可以有一两个小的语法不规范之处\n\n"
            "主题：{topic}"
        ),
    },
    "student_homework_B": {
        "category": "student",
        "prompt": (
            "你是一个中国大学生，正在赶期末论文。请针对以下题目写一段论述（300-500字）。\n"
            "语气要求：\n"
            "- 半正式，像大学生而不是学者\n"
            "- 可以引用但不需要严格格式\n"
            "- 适当加入个人观点和「我认为」\n"
            "- 允许一些不够严谨的推理\n"
            "- 不要段落过于工整对称\n\n"
            "题目：{topic}"
        ),
    },
    "student_homework_C": {
        "category": "student",
        "prompt": (
            "你是一个理工科大学生，请写一段实验报告中的「结果分析」部分（150-300字）。\n"
            "要求像学生手写的一样，不要太规范，可以有一些模糊表述如「大概」「差不多」「应该是因为」。\n\n"
            "实验内容：{topic}"
        ),
    },

    # --- P0-2: 口语化/聊天风格 ---
    "casual_zhihu_D": {
        "category": "casual",
        "prompt": (
            "你在知乎上回答一个问题，请用轻松自然的口语化风格写150-350字。\n"
            "要求：\n"
            "- 用「我」开头或直接切入话题\n"
            "- 可以用网络用语（hhh、确实、绷不住、离谱）\n"
            "- 句子可以短，可以不完整\n"
            "- 可以用省略号、破折号\n"
            "- 不要分点罗列，像聊天一样说\n"
            "- 可以夹带个人经历\n\n"
            "问题：{topic}"
        ),
    },
    "casual_tieba_E": {
        "category": "casual",
        "prompt": (
            "你在百度贴吧发帖，用很随意的中文写100-250字。\n"
            "要求：\n"
            "- 像真人打字一样，可以不加标点\n"
            "- 可以用缩写、谐音\n"
            "- 语气随意，甚至可以有点抱怨或吐槽\n"
            "- 不要有任何AI助手的痕迹\n\n"
            "话题：{topic}"
        ),
    },
    "casual_wechat_F": {
        "category": "casual",
        "prompt": (
            "你的朋友问你一个问题，你在微信上用语音转文字的方式回复ta。写150-300字。\n"
            "要求：\n"
            "- 完全口语化，像在说话\n"
            "- 会有「嗯」「啊」「就是说」「那个」之类的语气词\n"
            "- 可以跑题再绕回来\n"
            "- 不要太有逻辑，想到哪说到哪\n\n"
            "问题：{topic}"
        ),
    },

    # --- P0-3: 犹豫/不确定语气 ---
    "hesitant_unsure_G": {
        "category": "hesitant",
        "prompt": (
            "有人向你请教一个问题，你对这个领域了解一些但不是很精通。请用不太确定的语气回答200-350字。\n"
            "要求：\n"
            "- 频繁使用「好像」「大概」「我记得」「不太确定」「可能是」\n"
            "- 在某些关键点表达犹豫：「这个我不太确定啊」\n"
            "- 偶尔自我修正：「等等，好像不是这样...应该是...」\n"
            "- 主动承认知识局限：「这块我不太了解」\n"
            "- 不要给出太确定的结论\n\n"
            "问题：{topic}"
        ),
    },
    "hesitant_ambiguous_H": {
        "category": "hesitant",
        "prompt": (
            "请就以下话题写一段200-300字的看法，但你要表现得很纠结，两面都能看到道理，无法做出明确判断。\n"
            "要求：\n"
            "- 用「一方面...但另一方面...」\n"
            "- 反复权衡：「说到底也不好说」\n"
            "- 用口语化的纠结：「这真的很难讲」\n"
            "- 最终不给明确结论\n"
            "- 不要结构化分析\n\n"
            "话题：{topic}"
        ),
    },
    "hesitant_gossip_I": {
        "category": "hesitant",
        "prompt": (
            "你听别人说了一件事，现在转述给朋友。写150-300字。\n"
            "要求：\n"
            "- 大量使用「听说」「好像是」「据说」「我也不确定真假」\n"
            "- 信息可以有点模糊和不完整\n"
            "- 表达自己的半信半疑\n"
            "- 像真人在八卦一样\n\n"
            "事件：{topic}"
        ),
    },
}


# ============================================================
# 生成逻辑
# ============================================================

SUGGESTED_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "claude-sonnet-4-5",
    "deepseek-v3",
    "qwen-max",
    "gemini-2.5-flash",
]


def generate_prompts(topic_pool: dict[str, list[str]], per_template: int = 60) -> list[dict]:
    """生成 prompt 列表"""
    results = []
    prompt_id = 0

    # topic 选择策略：不同模板类别用不同的 topic 池
    category_topic_map = {
        "student": ["student_homework", "academic", "general", "tech"],
        "casual":  ["casual_chat", "general", "life", "news"],
        "hesitant": ["uncertain_topics", "general", "tech", "academic"],
    }

    for tpl_name, tpl_config in TEMPLATES.items():
        category = tpl_config["category"]
        prompt_template = tpl_config["prompt"]

        # 收集可用 topic
        available_topics = []
        for pool_key in category_topic_map.get(category, ["general"]):
            if pool_key in topic_pool:
                available_topics.extend(topic_pool[pool_key])
            if pool_key in MANUAL_TOPICS:
                available_topics.extend(MANUAL_TOPICS[pool_key])

        # 去重 & shuffle
        available_topics = list(set(available_topics))
        random.shuffle(available_topics)

        if not available_topics:
            print(f"  WARNING: No topics for {tpl_name}, skipping")
            continue

        # 取 per_template 个 topic
        selected = available_topics[:per_template]
        print(f"  {tpl_name}: {len(selected)} prompts (from {len(available_topics)} topics)")

        for topic in selected:
            prompt_id += 1
            # 每个 prompt 随机分配 2 个模型
            models = random.sample(SUGGESTED_MODELS, 2)
            results.append({
                "id": prompt_id,
                "category": category,
                "template": tpl_name,
                "topic": topic,
                "prompt": prompt_template.replace("{topic}", topic),
                "suggested_models": models,
                "label": 1,  # AI 生成 -> label=1
            })

    return results


def main():
    parser = argparse.ArgumentParser(description="P0 Prompt 批量生成器")
    parser.add_argument("--per-template", type=int, default=60,
                        help="每个模板生成多少条 prompt（默认 60）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出路径（默认 configs/p0_prompts.json）")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    output_path = Path(args.output) if args.output else PROJECT_ROOT / "configs" / "p0_prompts.json"

    print("=== P0 Prompt 批量生成器 ===\n")

    # Step 1: 从数据集抽 topic
    print("[1/3] 从 merged_v2/train.csv 抽取 topic...")
    train_path = PROJECT_ROOT / "datasets" / "merged_v2" / "train.csv"
    source_texts = extract_topics_from_dataset(train_path, max_per_source=300)
    topic_pool = build_topic_pool(source_texts)

    for cat, topics in topic_pool.items():
        print(f"  {cat}: {len(topics)} topics")

    # 合入手工 topic
    for key, topics in MANUAL_TOPICS.items():
        topic_pool[key] = topics
        print(f"  {key} (manual): {len(topics)} topics")

    # Step 2: 生成 prompt
    print(f"\n[2/3] 生成 prompt（每模板 {args.per_template} 条）...")
    prompts = generate_prompts(topic_pool, per_template=args.per_template)

    # Step 3: 输出
    print(f"\n[3/3] 写入 {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)

    # 统计
    by_category = {}
    by_template = {}
    for p in prompts:
        by_category[p["category"]] = by_category.get(p["category"], 0) + 1
        by_template[p["template"]] = by_template.get(p["template"], 0) + 1

    print(f"\n=== 统计 ===")
    print(f"总计: {len(prompts)} 条 prompt")
    print(f"\n按类别:")
    for cat, count in sorted(by_category.items()):
        print(f"  {cat}: {count}")
    print(f"\n按模板:")
    for tpl, count in sorted(by_template.items()):
        print(f"  {tpl}: {count}")

    # 每个模型被分配到多少条
    model_counts = {}
    for p in prompts:
        for m in p["suggested_models"]:
            model_counts[m] = model_counts.get(m, 0) + 1
    print(f"\n按模型分配:")
    for m, c in sorted(model_counts.items(), key=lambda x: -x[1]):
        print(f"  {m}: {c}")

    print(f"\n完成！输出文件: {output_path}")


if __name__ == "__main__":
    main()
