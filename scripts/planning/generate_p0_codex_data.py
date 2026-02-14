#!/usr/bin/env python3
"""
Generate P0 AI-style training data for three categories.

Output format: JSONL, each line:
{"text": "...", "label": 1, "source": "p0_student_codex", "category": "student"}
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets" / "p0_generated" / "p0_codex.jsonl"

DOMAINS: Dict[str, List[str]] = {
    "教育": [
        "大学课堂互动", "考试压力", "课程论文选题", "实习与学习冲突", "线上教学体验",
        "学分制度", "学习动力", "研究方法入门", "选课策略", "师生沟通",
        "实验课程安排", "学习小组合作",
    ],
    "科技": [
        "人工智能应用", "隐私与数据安全", "自动驾驶", "算法推荐", "量子计算",
        "芯片国产化", "智能穿戴", "云计算成本", "开源生态", "机器人服务",
        "5G与物联网", "软件漏洞治理",
    ],
    "社会": [
        "城市通勤", "社区互助", "网络暴力", "人口老龄化", "职业内卷",
        "公共卫生意识", "社会信任", "青年就业", "社会保障", "公益参与",
        "城乡差异", "公共空间使用",
    ],
    "文学": [
        "现代小说叙事", "诗歌意象", "人物塑造", "文学经典重读", "读者共鸣",
        "地方文学", "校园文学", "网络文学评价", "文学改编", "散文写作",
        "作家风格", "文学与时代",
    ],
    "经济": [
        "消费降级", "中小企业融资", "房租与生活成本", "就业与薪资预期", "平台经济",
        "外贸波动", "成本控制", "城市商业活力", "创业风险", "价格战",
        "宏观经济信心", "行业周期",
    ],
    "心理": [
        "拖延症", "焦虑管理", "情绪表达", "自我效能感", "社交压力",
        "亲密关系沟通", "习得性无助", "心理韧性", "自我认同", "失眠与作息",
        "情绪劳动", "动力不足",
    ],
    "历史": [
        "历史记忆", "朝代更替", "制度演变", "史料可信度", "文化交流",
        "战争影响", "经济制度变迁", "历史人物评价", "地方史", "历史与现实",
        "社会结构", "科技发展史",
    ],
    "生物": [
        "基因编辑", "免疫机制", "微生物生态", "疫苗研发", "生态系统平衡",
        "生物多样性", "细胞信号传导", "动植物适应", "公共健康", "食品安全",
        "人体代谢", "神经系统",
    ],
    "法律": [
        "劳动合同", "知识产权", "网络侵权", "隐私保护", "合同履行",
        "法律援助", "行政处罚", "校园法规", "消费者权益", "平台治理",
        "刑法边界", "证据采信",
    ],
    "日常生活": [
        "租房体验", "通勤效率", "饮食健康", "居家收纳", "时间管理",
        "邻里关系", "网络购物", "运动习惯", "社交媒体使用", "旅行计划",
        "生活成本", "家务分工",
    ],
}

ANGLES = [
    "的影响与反思", "中的常见问题", "的利与弊", "的现实困境", "的变化趋势",
    "背后的原因分析", "与个人选择", "与社会环境的关系", "带来的挑战", "与未来可能",
    "在实际体验中的差距", "如何改进", "值得讨论的地方", "让我困惑的点",
]

STUDENT_OPENERS = [
    "关于{topic}，我觉得课上讲的内容还挺有启发的",
    "这次作业写到{topic}时，我发现没我想象得那么简单",
    "说实话，{topic}这个话题我之前没认真想过",
    "老师布置的题目里提到{topic}，我有点纠结要怎么展开",
    "其实{topic}和我自己的经历还有点关系，所以写的时候挺有感觉",
]

STUDENT_MIDDLES = [
    "我理解它的核心还是和现实选择有关，但我说得可能不够准确",
    "有些概念我还没完全吃透，所以表述可能会有点笼统",
    "论证的时候我也发现自己逻辑不够顺，可能会有点跳",
    "写着写着就想到别的例子，不过感觉也能说明一点问题",
    "我觉得需要从几个角度看，但我暂时只能写出一个方向",
    "相关材料里有些观点挺矛盾的，我也没完全理清楚",
    "我想举个具体例子，但例子可能不够典型",
]

STUDENT_ENDINGS = [
    "总之我个人更倾向于一种比较中间的看法",
    "写完后发现还有不少漏洞，只能先交个草稿",
    "如果再完善的话，可能要再查些资料补充",
    "我的理解还比较初步，希望后面能更清晰一点",
    "这里的结论不敢下得太死，只能算是一个尝试",
]

CASUAL_OPENERS = [
    "我觉得{topic}这事儿挺有意思的",
    "{topic}我以前也纠结过",
    "这个问题我来随便说两句，关于{topic}",
    "说到{topic}，我第一反应其实挺复杂",
    "我身边有人就遇到{topic}，当时真的有点离谱",
]

CASUAL_MIDDLES = [
    "感觉大家讨论的时候经常把重点搞混了",
    "我自己体验下来就是挺现实的，没那么理想",
    "说白了就是心态问题吧，反正我会这么看",
    "有的人觉得无所谓，有的人就特别在意",
    "这事儿得看具体情况，套模板解释不通",
    "我之前还被吐槽过，哈哈哈",
    "确实有点绷不住，但也能理解",
]

CASUAL_ENDINGS = [
    "反正我的体验就是这样，不代表所有人",
    "你要是有别的想法也可以聊聊",
    "总之别太较真就好",
    "先说到这儿，想起来再补",
    "有懂的来补充一下吧",
]

HESITANT_OPENERS = [
    "{topic}这个问题我好像了解一点",
    "说到{topic}，我记得以前看过一些资料",
    "{topic}吧，我也不是很确定",
    "这个问题挺复杂的，关于{topic}我只能说个大概",
    "我对{topic}有点印象，但可能记得不准",
]

HESITANT_MIDDLES = [
    "大概是因为环境和个人选择都有影响，但我也不敢下结论",
    "有些人说是这个原因，但也有人反驳，说到底不好说",
    "我记得好像有个例子能说明，不过细节想不起来了",
    "等等，好像不是这样……应该还有别的因素",
    "一方面它确实有价值，但另一方面问题也挺明显",
    "这块我不太了解，所以只能说个模糊的印象",
]

HESITANT_ENDINGS = [
    "总之只是我个人的猜测，可能不太准",
    "如果有懂的人可以纠正我",
    "我觉得还需要更多信息才能判断",
    "目前没有明确结论，只能先放着",
    "说到底也不好说，到底哪边更重要",
]


def build_topics(per_domain: int, seed: int) -> List[Tuple[str, str]]:
    rng = random.Random(seed)
    topics: List[Tuple[str, str]] = []
    for domain, subjects in DOMAINS.items():
        local_topics = set()
        while len(local_topics) < per_domain:
            subject = rng.choice(subjects)
            angle = rng.choice(ANGLES)
            variant = rng.choice([
                f"{subject}{angle}",
                f"{angle}：{subject}",
                f"关于{subject}{angle}",
                f"{subject}相关{angle}",
            ])
            local_topics.add(variant)
        for topic in local_topics:
            topics.append((domain, topic))
    rng.shuffle(topics)
    return topics


def assemble_sentences(
    opener_pool: List[str],
    middle_pool: List[str],
    ending_pool: List[str],
    topic: str,
    rng: random.Random,
    min_sentences: int,
    max_sentences: int,
) -> List[str]:
    sentences = [rng.choice(opener_pool).format(topic=topic)]
    middle_count = rng.randint(max(1, min_sentences - 2), max_sentences - 2)
    sentences.extend(rng.sample(middle_pool, k=middle_count))
    sentences.append(rng.choice(ending_pool))
    return sentences


def finalize_text(text: str, min_len: int, max_len: int, rng: random.Random) -> str:
    if len(text) < min_len:
        fillers = [
            "这个话题还是挺值得继续讨论的",
            "我可能说得不够完整，只能先写到这里",
            "具体情况还要看实际背景",
        ]
        while len(text) < min_len:
            text += "。" + rng.choice(fillers)
    if len(text) > max_len:
        text = text[: max_len - 1].rstrip("，。；;") + "。"
    return text


def generate_student(topic: str, rng: random.Random) -> str:
    sentences = assemble_sentences(
        STUDENT_OPENERS, STUDENT_MIDDLES, STUDENT_ENDINGS, topic, rng, 4, 6
    )
    return "。".join(sentences) + "。"


def generate_casual(topic: str, rng: random.Random) -> str:
    sentences = assemble_sentences(
        CASUAL_OPENERS, CASUAL_MIDDLES, CASUAL_ENDINGS, topic, rng, 3, 5
    )
    text = "。".join(sentences) + "。"
    if rng.random() < 0.3:
        text = text.replace("。", "", 1)
    if rng.random() < 0.25:
        text = text.replace("。", "…")
    return text


def generate_hesitant(topic: str, rng: random.Random) -> str:
    sentences = assemble_sentences(
        HESITANT_OPENERS, HESITANT_MIDDLES, HESITANT_ENDINGS, topic, rng, 4, 6
    )
    return "。".join(sentences) + "。"


def create_records(
    topics: List[Tuple[str, str]],
    category: str,
    source: str,
    generator,
    rng: random.Random,
    min_len: int,
    max_len: int,
    limit: int,
    used_texts: set,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    attempts = 0
    index = 0
    while len(records) < limit and index < len(topics):
        _, topic = topics[index]
        index += 1
        text = generator(topic, rng)
        text = finalize_text(text, min_len, max_len, rng)
        attempts += 1
        if text in used_texts:
            if attempts > limit * 5:
                continue
            continue
        used_texts.add(text)
        records.append(
            {
                "text": text,
                "label": 1,
                "source": source,
                "category": category,
            }
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate P0 codex dataset.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    topics = build_topics(per_domain=60, seed=args.seed)
    if len(topics) < 600:
        print("ERROR: Not enough unique topics.", file=sys.stderr)
        sys.exit(1)

    used_texts: set = set()
    student_topics = topics[:200]
    casual_topics = topics[200:400]
    hesitant_topics = topics[400:600]

    student_records = create_records(
        student_topics,
        "student",
        "p0_student_codex",
        generate_student,
        rng,
        150,
        500,
        200,
        used_texts,
    )
    casual_records = create_records(
        casual_topics,
        "casual",
        "p0_casual_codex",
        generate_casual,
        rng,
        150,
        500,
        200,
        used_texts,
    )
    hesitant_records = create_records(
        hesitant_topics,
        "hesitant",
        "p0_hesitant_codex",
        generate_hesitant,
        rng,
        150,
        500,
        200,
        used_texts,
    )

    all_records = student_records + casual_records + hesitant_records
    if len(all_records) != 600:
        print("ERROR: Generated count mismatch.", file=sys.stderr)
        sys.exit(1)

    try:
        with open(output_path, "w", encoding="utf-8") as handle:
            for record in all_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"ERROR: Failed to write output: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Generated {len(all_records)} samples -> {output_path}")
    print(f"student={len(student_records)} casual={len(casual_records)} hesitant={len(hesitant_records)}")
    print(f"generated_at={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
