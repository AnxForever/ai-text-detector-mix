"""Classify dataset samples by scenario and emit bucket stats.

This script appends scenario-related fields (scenario/scenario_id/scenario_conf,
flags) and optionally reuses existing style/length when present. It also emits
scenario x style x length stats for gap analysis.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(PROJECT_ROOT)

from scripts.data_cleaning.classify_dataset import detect_style, length_bucket


SCENARIO_MAP = {
    "education": "A",
    "workplace": "B",
    "knowledge": "C",
    "community": "D",
    "commerce": "E",
    "news": "F",
    "unknown": "U",
}

STYLE_PLAN_ORDER = [
    "dialogue",
    "explanation",
    "list",
    "report",
    "guide",
    "mixed",
]

EDU_STRONG = [
    "课程",
    "作业",
    "实验报告",
    "开题",
    "答辩",
    "学号",
    "指导老师",
    "学院",
    "本科",
    "研究生",
]
EDU_STRUCT = [
    "摘要",
    "关键词",
    "引言",
    "相关工作",
    "方法",
    "实验",
    "结果",
    "讨论",
    "结论",
]

WORK_STYLE = ["周报", "会议纪要", "纪要", "复盘", "邮件", "通知"]
WORK_STRUCT = ["行动项", "负责人", "截止", "里程碑", "风险", "阻塞", "排期", "工单", "回滚", "上线"]
WORK_WEAK = ["OKR", "KPI", "里程碑", "排期", "负责人", "工单", "复盘", "阻塞", "风险"]

COMMUNITY_MARKERS = [
    "哈哈",
    "笑死",
    "无语",
    "离谱",
    "吐槽",
    "求助",
    "顶",
    "踩",
    "求扩散",
    "姐妹们",
    "兄弟们",
    "xdm",
    "yyds",
]

COMMERCE_STRONG = ["参数", "规格", "对比", "测评", "优点", "缺点", "价格", "性价比", "型号", "品牌"]
COMMERCE_WEAK = [
    "参数",
    "规格",
    "对比",
    "测评",
    "优点",
    "缺点",
    "价格",
    "性价比",
    "型号",
    "品牌",
    "到手价",
    "优惠",
    "券",
    "晒单",
    "开箱",
    "物流",
    "售后",
    "客服",
]

KNOWLEDGE_WEAK = ["原理", "机制", "定义", "区别", "例子", "误区", "适用场景", "注意事项", "是什么", "为什么"]

NEWS_STRONG = ["据报道", "发布", "宣布", "通报", "记者", "报道称", "新华社"]
NEWS_EVENT = ["今日", "昨日", "本周", "本月", "今年", "去年", "发生", "事故", "现场", "会议"]

TRANSACTION_INTENT = [
    "购买",
    "下单",
    "价格",
    "优惠",
    "折扣",
    "性价比",
    "推荐型号",
    "到手价",
    "券",
    "店铺",
    "客服",
    "售后",
    "购物",
]

QA_MARKERS = ["问:", "答:", "q:", "a:", "问题:", "回答:"]

DATE_PATTERN = re.compile(r"\d{4}年|\d{1,2}月|\d{1,2}日")


def normalize_text(text: str, max_chars: int = 4000) -> str:
    if not text:
        return ""
    return text[:max_chars]


def contains_any(sample: str, keywords: List[str]) -> bool:
    return any(key in sample for key in keywords)


def count_hits(sample: str, keywords: List[str]) -> int:
    return sum(1 for key in keywords if key in sample)


def detect_flags(sample: str) -> Tuple[bool, bool]:
    has_transaction = contains_any(sample, TRANSACTION_INTENT)
    is_event = contains_any(sample, NEWS_STRONG + NEWS_EVENT) or bool(DATE_PATTERN.search(sample))
    return has_transaction, is_event


def detect_scenario(text: str, text_len: int) -> Tuple[str, str, str, bool, bool]:
    sample = normalize_text(text)
    lower = sample.lower()
    has_transaction, is_event = detect_flags(sample)

    # Style/structure strong rules
    if contains_any(sample, WORK_STYLE) and not contains_any(sample, NEWS_STRONG):
        return "workplace", "B", "high", has_transaction, is_event

    if contains_any(sample, COMMERCE_STRONG) and ("优点" in sample or "缺点" in sample or "性价比" in sample):
        return "commerce", "E", "high", has_transaction, is_event

    if contains_any(sample, NEWS_STRONG):
        return "news", "F", "high", has_transaction, is_event

    # Structure rules
    if contains_any(sample, WORK_STRUCT):
        return "workplace", "B", "high", has_transaction, is_event

    if contains_any(sample, COMMERCE_STRONG) and ("价格" in sample or "参数" in sample or "规格" in sample):
        return "commerce", "E", "high", has_transaction, is_event

    if is_event:
        return "news", "F", "high", has_transaction, is_event

    # Keyword/group rules
    edu_strong_hits = count_hits(sample, EDU_STRONG)
    edu_struct_hits = count_hits(sample, EDU_STRUCT)
    if edu_strong_hits >= 2 or (edu_strong_hits >= 1 and edu_struct_hits >= 1):
        return "education", "A", "medium", has_transaction, is_event

    if contains_any(sample, WORK_WEAK):
        return "workplace", "B", "medium", has_transaction, is_event

    if contains_any(sample, COMMERCE_WEAK) or has_transaction:
        return "commerce", "E", "medium", has_transaction, is_event

    if any(marker in lower for marker in QA_MARKERS) and not has_transaction:
        return "knowledge", "C", "medium", has_transaction, is_event

    if text_len < 200 and contains_any(sample, COMMUNITY_MARKERS):
        return "community", "D", "medium", has_transaction, is_event

    if contains_any(sample, KNOWLEDGE_WEAK) and not has_transaction:
        return "knowledge", "C", "medium", has_transaction, is_event

    if contains_any(sample, NEWS_EVENT):
        return "news", "F", "medium", has_transaction, is_event

    # Fallback
    return "unknown", "U", "low", has_transaction, is_event


def map_style_to_plan(style: str) -> str:
    """Map legacy style labels to scenario plan style buckets."""
    if not style:
        return "mixed"
    if style == "dialogue":
        return "dialogue"
    if style == "explanation":
        return "explanation"
    if style == "list":
        return "list"
    if style == "academic":
        return "report"
    if style in ("technical_doc", "readme"):
        return "guide"
    return "mixed"


def iter_csv_files(input_path: str) -> Iterable[str]:
    if os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            for file in files:
                if file.lower().endswith(".csv"):
                    yield os.path.join(root, file)
    else:
        yield input_path


def parse_int(value: str) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def enrich_csv(
    input_file: str,
    output_file: str,
    stats_counter: Dict[Tuple[str, str, str, str], int],
    scenario_counter: Counter,
    style_counter: Counter,
    length_counter: Counter,
    unknown_counter: Counter,
    sleep_ms: int,
    progress_every: int,
) -> None:
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Missing header in {input_file}")
        fieldnames = list(reader.fieldnames)

        extra_fields = [
            "scenario",
            "scenario_id",
            "scenario_conf",
            "has_transaction_intent",
            "is_event_based",
            "style",
            "style_plan",
            "length_bucket",
        ]
        for col in extra_fields:
            if col not in fieldnames:
                fieldnames.append(col)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8", newline="") as out_f:
            writer = csv.DictWriter(out_f, fieldnames=fieldnames)
            writer.writeheader()

            rows_seen = 0
            for row in reader:
                text = row.get("text", "")
                text_len = parse_int(row.get("length")) or parse_int(row.get("length_chars"))
                if text_len <= 0:
                    text_len = len(text.strip())

                style = row.get("style") or detect_style(text)
                style_plan = row.get("style_plan") or map_style_to_plan(style)
                bucket = length_bucket(text_len)

                scenario, scenario_id, conf, has_tx, is_event = detect_scenario(text, text_len)

                row["scenario"] = scenario
                row["scenario_id"] = scenario_id
                row["scenario_conf"] = conf
                row["has_transaction_intent"] = str(bool(has_tx)).lower()
                row["is_event_based"] = str(bool(is_event)).lower()
                row["style"] = style
                row["style_plan"] = style_plan
                row["length_bucket"] = bucket

                writer.writerow(row)

                stats_key = (scenario, style_plan, bucket, conf)
                stats_counter[stats_key] += 1
                scenario_counter[scenario] += 1
                style_counter[style_plan] += 1
                length_counter[bucket] += 1
                if scenario == "unknown":
                    unknown_counter[conf] += 1

                rows_seen += 1
                if progress_every > 0 and rows_seen % progress_every == 0:
                    print(f"[{os.path.basename(input_file)}] processed {rows_seen:,} rows")
                if sleep_ms > 0 and rows_seen % 1000 == 0:
                    time.sleep(sleep_ms / 1000.0)


def write_stats_csv(stats_path: str, stats_counter: Dict[Tuple[str, str, str, str], int]) -> None:
    with open(stats_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "style", "length_bucket", "scenario_conf", "count"])
        for key in sorted(stats_counter):
            writer.writerow([*key, stats_counter[key]])


def write_report(
    report_path: str,
    dataset_name: str,
    scenario_counter: Counter,
    style_counter: Counter,
    length_counter: Counter,
    unknown_counter: Counter,
    stats_counter: Dict[Tuple[str, str, str, str], int],
) -> None:
    total = sum(scenario_counter.values())
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def format_counter(counter: Counter) -> List[str]:
        return [f"- {key}: {value}" for key, value in counter.most_common()]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 场景标注统计报告\n\n")
        f.write(f"> 数据集: {dataset_name}\n")
        f.write(f"> 生成时间: {timestamp}\n\n")
        f.write("## 基本统计\n\n")
        f.write(f"- 总样本数: {total}\n\n")
        f.write("### 场景分布\n\n")
        f.write("\n".join(format_counter(scenario_counter)) + "\n\n")
        f.write("### Style (Plan) 分布\n\n")
        f.write("\n".join(format_counter(style_counter)) + "\n\n")
        f.write("### Length Bucket 分布\n\n")
        f.write("\n".join(format_counter(length_counter)) + "\n\n")
        if unknown_counter:
            f.write("### Unknown 分布\n\n")
            f.write("\n".join(format_counter(unknown_counter)) + "\n\n")

        f.write("## 场景 × 文风 × 长度（节选）\n\n")
        f.write("| scenario | style_plan | length_bucket | conf | count |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for (scenario, style, bucket, conf), count in sorted(stats_counter.items())[:200]:
            f.write(f"| {scenario} | {style} | {bucket} | {conf} | {count} |\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify dataset by scenario rules.")
    parser.add_argument("--input", required=True, help="Input CSV file or directory.")
    parser.add_argument("--output-dir", required=True, help="Output directory for classified CSVs.")
    parser.add_argument("--stats-csv", required=True, help="Output CSV path for scenario stats.")
    parser.add_argument("--report", required=True, help="Output markdown report path.")
    parser.add_argument("--sleep-ms", type=int, default=0, help="Sleep ms every 1000 rows.")
    parser.add_argument("--progress-every", type=int, default=200000, help="Progress log interval.")
    args = parser.parse_args()

    input_path = args.input
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    stats_counter: Dict[Tuple[str, str, str, str], int] = defaultdict(int)
    scenario_counter: Counter = Counter()
    style_counter: Counter = Counter()
    length_counter: Counter = Counter()
    unknown_counter: Counter = Counter()

    for input_file in iter_csv_files(input_path):
        output_file = os.path.join(output_dir, os.path.basename(input_file))
        enrich_csv(
            input_file=input_file,
            output_file=output_file,
            stats_counter=stats_counter,
            scenario_counter=scenario_counter,
            style_counter=style_counter,
            length_counter=length_counter,
            unknown_counter=unknown_counter,
            sleep_ms=args.sleep_ms,
            progress_every=args.progress_every,
        )

    stats_path = args.stats_csv
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    write_stats_csv(stats_path, stats_counter)

    report_path = args.report
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    dataset_name = os.path.basename(os.path.normpath(input_path))
    write_report(
        report_path=report_path,
        dataset_name=dataset_name,
        scenario_counter=scenario_counter,
        style_counter=style_counter,
        length_counter=length_counter,
        unknown_counter=unknown_counter,
        stats_counter=stats_counter,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
