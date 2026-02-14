#!/usr/bin/env python3
"""
Generate P0 prompt batches from existing datasets.

This script extracts lightweight "topics" from text fields and expands them into
prompt batches for the three P0 blind spots:
1) Student assignment / essay style
2) Chatty / conversational style
3) Hesitant / uncertain style
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_DATASETS = [
    PROJECT_ROOT / "datasets" / "defense_focused" / "train.csv",
    PROJECT_ROOT / "datasets" / "defense_focused" / "val.csv",
    PROJECT_ROOT / "datasets" / "defense_focused" / "test.csv",
]

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "planning" / "p0_prompt_batches_20260210"

TEMPLATES: Dict[str, str] = {
    "A": (
        "你是一个中国大学本科生，正在写课程作业。请就以下主题写一段200-400字的回答。\n"
        "要求：\n"
        "- 像真实学生一样写，不要太完美\n"
        "- 可以有一些口头化表达，比如“我觉得”“其实”“说实话”\n"
        "- 偶尔可以用一些不太精确的表述\n"
        "- 不要用“首先、其次、最后”这种明显的结构化表达\n"
        "- 可以有一两个小的语法不规范之处\n"
        "主题：{topic}"
    ),
    "B": (
        "你是一个中国大学生，正在赶期末论文。请针对以下题目写一段论述（300-500字）。\n"
        "语气要求：\n"
        "- 半正式，像大学生而不是学者\n"
        "- 可以引用但不需要严格格式\n"
        "- 适当加入个人观点和“我认为”\n"
        "- 允许一些不够严谨的推理\n"
        "- 不要段落过于工整对称\n"
        "题目：{topic}"
    ),
    "C": (
        "你是一个理工科大学生，请写一段实验报告中的“结果分析”部分（150-300字）。\n"
        "要求像学生手写的一样，不要太规范，可以有一些模糊表述如“大概”“差不多”“应该是因为”。\n"
        "实验内容：{topic}"
    ),
    "D": (
        "你在知乎上回答一个问题，请用轻松自然的口语化风格写150-350字。\n"
        "要求：\n"
        "- 用“我”开头或直接切入话题\n"
        "- 可以用网络用语（hhh、确实、绷不住、离谱）\n"
        "- 句子可以短，可以不完整\n"
        "- 可以用省略号、破折号\n"
        "- 不要分点罗列，像聊天一样说\n"
        "- 可以夹带个人经历\n"
        "问题：{topic}"
    ),
    "E": (
        "你在百度贴吧发帖，用很随意的中文写100-250字。\n"
        "要求：\n"
        "- 像真人打字一样，可以不加标点\n"
        "- 可以用缩写、谐音、表情符号\n"
        "- 语气随意，甚至可以有点抱怨或吐槽\n"
        "- 不要有任何AI助手的痕迹\n"
        "话题：{topic}"
    ),
    "F": (
        "你的朋友问你一个问题，你在微信上用语音转文字的方式回复TA。写150-300字。\n"
        "要求：\n"
        "- 完全口语化，像在说话\n"
        "- 会有“嗯”“啊”“就是说”“那个”之类的语气词\n"
        "- 可以跑题再绕回来\n"
        "- 不要太有逻辑，想到哪说到哪\n"
        "问题：{topic}"
    ),
    "G": (
        "有人向你请教一个问题，你对这个领域了解一些但不是很精通。请用不太确定的语气回答200-350字。\n"
        "要求：\n"
        "- 频繁使用“好像”“大概”“我记得”“不太确定”“可能是”\n"
        "- 在某些关键点表达犹豫：“这个我不太确定啊”\n"
        "- 偶尔自我修正：“等等，好像不是这样...应该是...”\n"
        "- 主动承认知识局限：“这块我不太了解”\n"
        "- 不要给出太确定的结论\n"
        "问题：{topic}"
    ),
    "H": (
        "请就以下话题写一段200-300字的看法，但你要表现得很纠结，两面都能看到道理，无法做出明确判断。\n"
        "要求：\n"
        "- 用“一方面...但另一方面...”\n"
        "- 反复权衡：“说到底也不好说”\n"
        "- 用口语化的纠结：“这真的很难讲”\n"
        "- 最终不给明确结论\n"
        "- 不要结构化分析\n"
        "话题：{topic}"
    ),
    "I": (
        "你听别人说了一件事，现在转述给朋友。写150-300字。\n"
        "要求：\n"
        "- 大量使用“听说”“好像是”“据说”“我也不确定真假”\n"
        "- 信息可以有点模糊和不完整\n"
        "- 表达自己的半信半疑\n"
        "- 像真人在八卦一样\n"
        "事件：{topic}"
    ),
}

TEMPLATE_CATEGORIES = {
    "A": "student_assignment",
    "B": "student_assignment",
    "C": "student_assignment",
    "D": "chat_style",
    "E": "chat_style",
    "F": "chat_style",
    "G": "hesitant",
    "H": "hesitant",
    "I": "hesitant",
}

TEMPLATE_TARGETS = {
    "A": 200,
    "B": 200,
    "C": 200,
    "D": 120,
    "E": 120,
    "F": 120,
    "G": 80,
    "H": 80,
    "I": 80,
}


def normalize_text(text: str) -> str:
    """Normalize whitespace for topic extraction."""
    return re.sub(r"\s+", " ", text.strip())


def clean_topic(text: str) -> str:
    """Clean leading bullets/quotes and trailing punctuation."""
    text = re.sub(r"^[\-\*\u2022\d]+\s*[).、．]?\s*", "", text)
    text = text.strip(" \t\r\n\"'“”‘’[](){}")
    text = re.sub(r"[。！？!?;；]+$", "", text)
    return text.strip()


def extract_topic(text: str) -> Optional[str]:
    """Extract a short topic string from raw text."""
    if not text:
        return None
    text = normalize_text(text)
    if not text:
        return None

    if "\n" in text:
        first_line = text.split("\n", 1)[0].strip()
        if 4 <= len(first_line) <= 30:
            topic = clean_topic(first_line)
            if 4 <= len(topic) <= 40:
                return topic

    for marker in ("？", "?"):
        idx = text.find(marker)
        if 4 <= idx <= 50:
            topic = clean_topic(text[: idx + 1])
            if 4 <= len(topic) <= 40:
                return topic

    for marker in ("：", ":"):
        idx = text.find(marker)
        if 2 <= idx <= 20:
            prefix = text[:idx].strip()
            if len(prefix) <= 6:
                candidate = text[idx + 1 : idx + 36]
                topic = clean_topic(candidate)
                if 4 <= len(topic) <= 40:
                    return topic

    sentence_match = re.split(r"[。！？!?;；]", text, maxsplit=1)
    if sentence_match:
        topic = clean_topic(sentence_match[0][:40])
        if 4 <= len(topic) <= 40:
            return topic

    fallback = clean_topic(text[:40])
    if 4 <= len(fallback) <= 40:
        return fallback
    return None


def iter_topics(paths: Iterable[Path], label_filter: Optional[int]) -> List[str]:
    """Iterate datasets and return unique topic candidates."""
    topics: List[str] = []
    seen = set()
    for path in paths:
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if label_filter is not None:
                    try:
                        if int(row.get("label", "")) != label_filter:
                            continue
                    except ValueError:
                        continue
                topic = extract_topic(row.get("text", ""))
                if not topic or topic in seen:
                    continue
                seen.add(topic)
                topics.append(topic)
    return topics


def sample_topics(topics: List[str], max_topics: int, seed: int) -> List[str]:
    """Sample a fixed number of topics deterministically."""
    if len(topics) <= max_topics:
        return topics
    rng = random.Random(seed)
    return rng.sample(topics, max_topics)


def build_prompt_batches(
    topics: List[str],
    seed: int,
) -> List[Dict[str, str]]:
    """Expand topics into prompt batches."""
    rng = random.Random(seed)
    batches: List[Dict[str, str]] = []
    for template_id, template in TEMPLATES.items():
        target = TEMPLATE_TARGETS[template_id]
        if len(topics) >= target:
            selected = rng.sample(topics, target)
        else:
            selected = [rng.choice(topics) for _ in range(target)]

        for index, topic in enumerate(selected, start=1):
            batches.append(
                {
                    "template_id": template_id,
                    "category": TEMPLATE_CATEGORIES[template_id],
                    "sequence": str(index),
                    "topic": topic,
                    "prompt": template.format(topic=topic),
                }
            )
    return batches


def write_outputs(
    output_dir: Path,
    topics: List[str],
    batches: List[Dict[str, str]],
    sources: List[Path],
) -> None:
    """Write topics and prompt batches to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    formatted_sources: List[str] = []
    for path in sources:
        resolved = path.resolve()
        try:
            formatted_sources.append(str(resolved.relative_to(PROJECT_ROOT)))
        except ValueError:
            formatted_sources.append(str(path))

    topics_payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sources": formatted_sources,
        "topic_count": len(topics),
        "topics": topics,
    }
    (output_dir / "topics.json").write_text(
        json.dumps(topics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "topics.txt").write_text("\n".join(topics), encoding="utf-8")

    with open(output_dir / "prompt_batches.jsonl", "w", encoding="utf-8") as handle:
        for item in batches:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary_lines = [
        "# P0 Prompt Batches",
        "",
        f"- generated_at: {topics_payload['generated_at']}",
        f"- topics: {len(topics)}",
        f"- batches: {len(batches)}",
        f"- sources: {', '.join(topics_payload['sources'])}",
        "",
        "## Template Targets",
    ]
    for template_id in sorted(TEMPLATE_TARGETS.keys()):
        summary_lines.append(
            f"- {template_id}: {TEMPLATE_TARGETS[template_id]} "
            f"({TEMPLATE_CATEGORIES[template_id]})"
        )
    (output_dir / "README.md").write_text("\n".join(summary_lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate P0 prompt batches.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[str(path) for path in DEFAULT_DATASETS],
        help="Dataset CSV paths.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for topics/prompts.",
    )
    parser.add_argument(
        "--max-topics",
        type=int,
        default=240,
        help="Max unique topics to keep.",
    )
    parser.add_argument(
        "--label",
        type=int,
        default=0,
        help="Label to filter on (default: 0 for human).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_paths = [Path(path) for path in args.datasets]
    topics_raw = iter_topics(dataset_paths, args.label)
    topics = sample_topics(topics_raw, args.max_topics, args.seed)
    batches = build_prompt_batches(topics, args.seed)
    write_outputs(Path(args.output_dir), topics, batches, dataset_paths)
    print(f"Topics: {len(topics)} | Batches: {len(batches)}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
