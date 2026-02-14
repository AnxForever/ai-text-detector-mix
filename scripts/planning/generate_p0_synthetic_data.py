#!/usr/bin/env python3
"""
Generate synthetic AI samples for P0 blind spots without external LLMs.

Input: prompt_batches.jsonl (template_id, category, topic, prompt).
Output: CSV/JSONL with generated "AI-like" texts for quick augmentation.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PROMPT_BATCHES = (
    PROJECT_ROOT
    / "datasets"
    / "planning"
    / "p0_prompt_batches_20260210_full"
    / "prompt_batches.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "datasets" / "planning" / "p0_synthetic_data_20260210"
)

LENGTH_RANGES = {
    "A": (200, 400),
    "B": (300, 500),
    "C": (150, 300),
    "D": (150, 350),
    "E": (100, 250),
    "F": (150, 300),
    "G": (200, 350),
    "H": (200, 300),
    "I": (150, 300),
}


def _join_sentences(sentences: List[str]) -> str:
    return "。".join([s.strip("。") for s in sentences if s]) + "。"


def _trim_length(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    trimmed = text[: max_len - 1].rstrip("，。；;")
    return trimmed + "。"


def _extend_to_min(text: str, min_len: int, fillers: List[str], rng: random.Random) -> str:
    if len(text) >= min_len:
        return text
    sentences = [s for s in text.split("。") if s.strip()]
    while len("。".join(sentences)) < min_len:
        sentences.append(rng.choice(fillers))
    return _join_sentences(sentences)


def _finalize(text: str, min_len: int, max_len: int, fillers: List[str], rng: random.Random) -> str:
    text = _extend_to_min(text, min_len, fillers, rng)
    text = _trim_length(text, max_len)
    return text


def _student_assignment(topic: str, rng: random.Random, template_id: str) -> str:
    starters = [
        f"这次作业让我想到{topic}，其实没有想象中那么简单",
        f"关于{topic}，我觉得课上讲的内容挺有启发",
        f"{topic}这个话题我之前没怎么想过，这次写的时候有点卡",
    ]
    bodies = [
        "我理解它的核心还是和现实生活的选择有关，不过我说得可能不够准确",
        "说实话我有些地方没完全搞懂，只能按自己的理解来写",
        "老师提到的几个概念我还在消化，所以这里写得有点笼统",
        "举个小例子会更好说明，但我想的例子可能不够典型",
    ]
    ends = [
        "总之我的观点是偏向中间一点，但也不敢说得太绝对",
        "后面如果有时间我还想再查查资料补充一下",
        "写完感觉还是有不少漏洞，只能先交个草稿吧",
    ]
    if template_id == "B":
        bodies += [
            "论文里常见的论证路径我也尽量学了一下，但写起来还是不太稳",
            "我会把重点放在观点和例子上，不刻意追求特别严谨的结构",
        ]
        ends += [
            "如果要更严谨，可能还得补充文献和数据支持",
            "我认为这个问题还可以从别的角度再展开",
        ]
    if template_id == "C":
        bodies = [
            f"实验里和{topic}相关的现象比较明显，但数据波动也挺大",
            "我怀疑误差主要来自测量过程，不过也可能和环境条件有关",
            "结果趋势和预期差不多，但有几组数据有点偏离",
        ]
        ends = [
            "综合来看，大概可以说明这个规律是存在的，但还需要更多重复实验",
            "后面如果调整一下实验条件，结果可能会更稳定",
        ]
    sentences = [rng.choice(starters)]
    sentences.extend(rng.sample(bodies, k=min(2, len(bodies))))
    sentences.append(rng.choice(ends))
    return _join_sentences(sentences)


def _chat_style(topic: str, rng: random.Random, template_id: str) -> str:
    starters = [
        f"我觉得{topic}这事儿挺有意思的",
        f"{topic}我以前也纠结过",
        f"这个问题我来随便说两句，关于{topic}",
    ]
    bodies = [
        "感觉大家讨论的时候经常把重点搞混了",
        "我身边也有人这样，确实挺真实的",
        "说白了就是心态问题吧，反正我会这么看",
        "这事儿很看人，有的人就觉得无所谓，有的人就很在意",
    ]
    ends = [
        "反正我的体验就是这样，可能不代表所有人",
        "你要是有别的想法也可以聊聊",
        "总之别太较真就好，hhh",
    ]
    if template_id == "E":
        bodies += ["真的离谱", "我就不理解为啥会这样", "反正我是不太能接受"]
        ends = ["就这样吧", "有懂的来补充一下"]
    if template_id == "F":
        starters = [
            f"嗯这个问题啊，{topic}我也想过",
            f"那个，关于{topic}，我简单说说",
        ]
        bodies = [
            "就是说有时候你以为自己明白了，其实也没那么明白",
            "我觉得最关键还是看个人情况吧",
            "中间有些细节我也记不太清了",
        ]
        ends = [
            "反正大概就是这个意思哈",
            "先这样，有空再细聊",
        ]
    sentences = [rng.choice(starters)]
    sentences.extend(rng.sample(bodies, k=min(2, len(bodies))))
    sentences.append(rng.choice(ends))
    text = _join_sentences(sentences)
    if template_id == "E":
        text = re.sub(r"[，。！？!?;；]", "", text)
        text = text.replace("。", "")
    return text


def _hesitant(topic: str, rng: random.Random, template_id: str) -> str:
    starters = [
        f"{topic}这个问题我好像了解一点",
        f"说到{topic}，我记得以前看过一点资料",
        f"{topic}吧，我也不是很确定",
    ]
    bodies = [
        "大概是因为环境和个人选择都有影响，但我也不敢下结论",
        "有些人说是这个原因，但也有人反驳，说到底不好说",
        "我记得好像有个例子能说明，不过细节想不起来了",
        "等等，好像不是这样……应该还有别的因素",
    ]
    ends = [
        "总之只是我个人的猜测，可能不太准",
        "这块我不太了解，最好还是查一下更靠谱的资料",
        "如果有懂的人可以纠正我",
    ]
    if template_id == "H":
        sentences = [
            f"关于{topic}，一方面我觉得它确实有价值",
            "但另一方面它的问题也挺明显的",
            "说到底也不好说，到底哪边更重要",
            "这真的很难讲，我目前还没有明确结论",
        ]
        return _join_sentences(sentences)
    if template_id == "I":
        sentences = [
            f"我听说{topic}最近挺火的",
            "据说有人已经试过了，不过我也不确定真假",
            "反正听来的消息都挺零散的",
            "如果是真的那还挺有意思，但也可能是夸张了",
        ]
        return _join_sentences(sentences)
    sentences = [rng.choice(starters)]
    sentences.extend(rng.sample(bodies, k=min(2, len(bodies))))
    sentences.append(rng.choice(ends))
    return _join_sentences(sentences)


def generate_text(topic: str, template_id: str, rng: random.Random) -> str:
    if template_id in {"A", "B", "C"}:
        return _student_assignment(topic, rng, template_id)
    if template_id in {"D", "E", "F"}:
        return _chat_style(topic, rng, template_id)
    return _hesitant(topic, rng, template_id)


def load_prompt_batches(path: Path) -> List[Dict[str, str]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Prompt batch file not found: {path}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to read prompt batches: {exc}") from exc


def write_outputs(output_dir: Path, rows: List[Dict[str, str]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "ai_samples.jsonl"
    csv_path = output_dir / "ai_samples.csv"

    with open(jsonl_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["text", "label", "source", "category", "template_id", "topic"],
        )
        writer.writeheader()
        writer.writerows(rows)

    meta_path = output_dir / "README.md"
    meta_path.write_text(
        "\n".join(
            [
                "# P0 Synthetic AI Samples",
                "",
                f"- generated_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"- samples: {len(rows)}",
                f"- jsonl: {jsonl_path}",
                f"- csv: {csv_path}",
            ]
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate P0 synthetic AI samples.")
    parser.add_argument(
        "--prompt-batches",
        default=str(DEFAULT_PROMPT_BATCHES),
        help="Path to prompt_batches.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    batches = load_prompt_batches(Path(args.prompt_batches))
    rows: List[Dict[str, str]] = []
    for item in batches:
        template_id = item.get("template_id", "")
        topic = item.get("topic", "")
        category = item.get("category", "")
        text = generate_text(topic, template_id, rng)
        min_len, max_len = LENGTH_RANGES.get(template_id, (150, 300))
        fillers = ["这个问题还挺复杂的", "我可能说得不够完整", "就先想到这么多"]
        text = _finalize(text, min_len, max_len, fillers, rng)
        rows.append(
            {
                "text": text,
                "label": 1,
                "source": f"p0_synth_{template_id}",
                "category": category,
                "template_id": template_id,
                "topic": topic,
            }
        )

    write_outputs(Path(args.output_dir), rows)
    print(f"Generated {len(rows)} samples -> {args.output_dir}")


if __name__ == "__main__":
    main()
