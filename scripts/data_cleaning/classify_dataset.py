"""Classify dataset samples by style, domain, and length bucket.

This script reads CSV files, appends classification columns, and writes
enriched CSVs to the output directory. It also emits bucket statistics
for downstream gap analysis.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, Iterable, List, Tuple


os.environ.setdefault("PYTHONIOENCODING", "utf-8")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(PROJECT_ROOT)


STYLE_ORDER = [
    "dialogue",
    "academic",
    "readme",
    "list",
    "technical_doc",
    "explanation",
]

LENGTH_BUCKETS = [
    "lt_80",
    "80-200",
    "200-500",
    "500-1000",
    "1000-2000",
    "2000+",
]

DOMAIN_KEYWORDS = {
    "ml_ai": [
        "模型",
        "训练",
        "推理",
        "loss",
        "梯度",
        "embedding",
        "token",
        "dataset",
        "transformer",
        "bert",
        "机器学习",
        "深度学习",
        "cnn",
        "rnn",
        "auc",
        "llm",
    ],
    "software": [
        "代码",
        "函数",
        "变量",
        "库",
        "框架",
        "api",
        "sdk",
        "python",
        "java",
        "javascript",
        "node",
        "react",
        "pytorch",
        "编译",
        "调试",
        "git",
        "linux",
    ],
    "ops": [
        "服务器",
        "运维",
        "部署",
        "docker",
        "kubernetes",
        "k8s",
        "nginx",
        "内存",
        "cpu",
        "端口",
        "日志",
        "故障",
        "容器",
        "集群",
    ],
    "finance": [
        "股票",
        "基金",
        "收益",
        "利率",
        "财报",
        "投资",
        "保险",
        "银行",
        "债券",
        "汇率",
        "金融",
    ],
    "medical": [
        "症状",
        "治疗",
        "疾病",
        "药",
        "患者",
        "临床",
        "诊断",
        "医学",
        "护理",
        "疗效",
    ],
    "education": [
        "课程",
        "考试",
        "学生",
        "教学",
        "作业",
        "课堂",
        "学习",
        "讲义",
    ],
    "law": [
        "法律",
        "法规",
        "合同",
        "条款",
        "法院",
        "律师",
        "诉讼",
        "责任",
    ],
    "general": [],
}

STYLE_KEYWORDS = {
    "academic": [
        "摘要",
        "关键词",
        "引言",
        "方法",
        "实验",
        "结论",
        "参考文献",
        "abstract",
        "introduction",
        "method",
        "results",
    ],
    "readme": [
        "readme",
        "安装",
        "使用",
        "usage",
        "quickstart",
        "许可证",
        "license",
        "贡献",
        "contributing",
        "环境要求",
        "示例",
    ],
    "technical_doc": [
        "参数",
        "配置",
        "接口",
        "返回",
        "异常",
        "请求",
        "响应",
        "字段",
        "路径",
        "示例",
        "endpoint",
        "status code",
        "http",
    ],
    "dialogue": [
        "用户：",
        "助手：",
        "问：",
        "答：",
        "q:",
        "a:",
        "user:",
        "assistant:",
    ],
}


def normalize_text(text: str, max_chars: int = 4000) -> str:
    """Return a normalized text slice for fast rule checks."""
    if not text:
        return ""
    return text[:max_chars]


def length_bucket(length_value: int) -> str:
    """Map raw character length to a bucket label."""
    if length_value < 80:
        return "lt_80"
    if length_value < 200:
        return "80-200"
    if length_value < 500:
        return "200-500"
    if length_value < 1000:
        return "500-1000"
    if length_value < 2000:
        return "1000-2000"
    return "2000+"


def detect_style(text: str) -> str:
    """Detect text style using lightweight rules."""
    sample = normalize_text(text)
    lower = sample.lower()
    lines = [line.strip() for line in sample.splitlines() if line.strip()]

    dialogue_hits = sum(1 for key in STYLE_KEYWORDS["dialogue"] if key in lower)
    if dialogue_hits > 0:
        return "dialogue"

    academic_hits = sum(1 for key in STYLE_KEYWORDS["academic"] if key in lower)
    readme_hits = sum(1 for key in STYLE_KEYWORDS["readme"] if key in lower)
    tech_hits = sum(1 for key in STYLE_KEYWORDS["technical_doc"] if key in lower)

    list_lines = 0
    table_hits = 0
    for line in lines[:200]:
        if line.startswith(("- ", "* ", "+ ")):
            list_lines += 1
        if any(line.startswith(prefix) for prefix in ("1.", "1)", "1、")):
            list_lines += 1
        if "|" in line and "---" in line:
            table_hits += 1

    list_score = list_lines + table_hits * 2
    if readme_hits >= 2 and "```" in sample:
        readme_hits += 2
    if tech_hits >= 2 and list_score >= 2:
        tech_hits += 1

    scores = {
        "academic": academic_hits * 2,
        "readme": readme_hits * 2,
        "list": list_score,
        "technical_doc": tech_hits * 2,
        "explanation": 1,
    }

    best_style = "explanation"
    best_score = scores.get(best_style, 0)
    for style in STYLE_ORDER:
        score = scores.get(style, 0)
        if score > best_score:
            best_style = style
            best_score = score
    return best_style


def detect_domain(text: str) -> str:
    """Detect domain based on keyword counts."""
    sample = normalize_text(text).lower()
    scores: Dict[str, int] = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if not keywords:
            continue
        scores[domain] = sum(1 for kw in keywords if kw in sample)
    if not scores:
        return "general"
    best_domain = max(scores.items(), key=lambda item: item[1])
    if best_domain[1] == 0:
        return "general"
    return best_domain[0]


def map_label_to_main(label_value: str) -> str:
    """Map raw label to HUMAN/AI when possible."""
    label_value = str(label_value).strip().lower()
    if label_value in {"1", "ai", "machine"}:
        return "AI"
    if label_value in {"0", "human"}:
        return "HUMAN"
    return "UNCERTAIN"


def iter_csv_files(input_path: str) -> Iterable[str]:
    """Yield CSV files from a directory or a single file path."""
    if os.path.isfile(input_path):
        if input_path.lower().endswith(".csv"):
            yield input_path
        return
    for name in os.listdir(input_path):
        if name.lower().endswith(".csv"):
            yield os.path.join(input_path, name)


def enrich_csv(
    input_file: str,
    output_file: str,
    stats_counter: Dict[Tuple[str, str, str, str, str], int],
    label_counter: Counter,
    style_counter: Counter,
    domain_counter: Counter,
    length_counter: Counter,
    sleep_ms: int,
    progress_every: int,
) -> None:
    """Enrich a single CSV and update stats counters."""
    rows_seen = 0
    try:
        with open(input_file, "r", encoding="utf-8", errors="replace", newline="") as f_in, open(
            output_file, "w", encoding="utf-8", newline=""
        ) as f_out:
            reader = csv.DictReader(f_in)
            if not reader.fieldnames:
                return
            fieldnames = list(reader.fieldnames)
            extra_fields = [
                "y_main",
                "style",
                "domain",
                "length_bucket",
                "length_chars",
                "source_type",
            ]
            for field in extra_fields:
                if field not in fieldnames:
                    fieldnames.append(field)

            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                rows_seen += 1
                text = row.get("text", "") or ""
                text_str = str(text)
                length_val = len(text_str.strip())
                bucket = length_bucket(length_val)
                style = detect_style(text_str)
                domain = detect_domain(text_str)
                label_raw = row.get("label", "")
                y_main = map_label_to_main(label_raw)
                source_type = "ai" if y_main == "AI" else "human" if y_main == "HUMAN" else ""

                row["y_main"] = y_main
                row["style"] = style
                row["domain"] = domain
                row["length_bucket"] = bucket
                row["length_chars"] = str(length_val)
                if "source_type" not in row or not row.get("source_type"):
                    row["source_type"] = source_type

                writer.writerow(row)

                stats_key = (
                    os.path.basename(input_file),
                    y_main,
                    style,
                    domain,
                    bucket,
                )
                stats_counter[stats_key] += 1
                label_counter[y_main] += 1
                style_counter[style] += 1
                domain_counter[domain] += 1
                length_counter[bucket] += 1

                if progress_every > 0 and rows_seen % progress_every == 0:
                    print(f"[{os.path.basename(input_file)}] processed {rows_seen:,} rows")
                if sleep_ms > 0 and rows_seen % 1000 == 0:
                    time.sleep(sleep_ms / 1000.0)
    except FileNotFoundError as exc:
        print(f"File not found: {input_file} ({exc})")
        raise
    except Exception as exc:
        print(f"Failed to process {input_file}: {exc}")
        raise


def write_stats_csv(stats_path: str, stats_counter: Dict[Tuple[str, str, str, str, str], int]) -> None:
    """Write stats counter to CSV."""
    try:
        with open(stats_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["split_file", "y_main", "style", "domain", "length_bucket", "count"])
            for key in sorted(stats_counter):
                writer.writerow([*key, stats_counter[key]])
    except Exception as exc:
        print(f"Failed to write stats CSV: {exc}")
        raise


def write_report(
    report_path: str,
    dataset_name: str,
    label_counter: Counter,
    style_counter: Counter,
    domain_counter: Counter,
    length_counter: Counter,
    stats_counter: Dict[Tuple[str, str, str, str, str], int],
) -> None:
    """Write a simple markdown report for bucket coverage."""
    total = sum(label_counter.values())
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def format_counter(counter: Counter) -> List[str]:
        return [f"- {key}: {value}" for key, value in counter.most_common()]

    missing_combos = []
    for style in STYLE_ORDER:
        for domain in DOMAIN_KEYWORDS:
            for bucket in LENGTH_BUCKETS:
                total_count = 0
                for label in ("HUMAN", "AI"):
                    key = ("ALL", label, style, domain, bucket)
                    total_count += stats_counter.get(key, 0)
                if total_count == 0:
                    missing_combos.append((style, domain, bucket))

    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# 数据集分类与桶统计报告\n\n")
            f.write(f"> 数据集: {dataset_name}\n")
            f.write(f"> 生成时间: {timestamp}\n\n")
            f.write("## 基本统计\n\n")
            f.write(f"- 总样本数: {total}\n")
            f.write("\n### 标签分布\n\n")
            f.write("\n".join(format_counter(label_counter)) + "\n\n")
            f.write("### Style 分布\n\n")
            f.write("\n".join(format_counter(style_counter)) + "\n\n")
            f.write("### Domain 分布\n\n")
            f.write("\n".join(format_counter(domain_counter)) + "\n\n")
            f.write("### Length Bucket 分布\n\n")
            f.write("\n".join(format_counter(length_counter)) + "\n\n")
            f.write("## 缺失桶提示 (style × domain × length)\n\n")
            if missing_combos:
                for style, domain, bucket in missing_combos[:50]:
                    f.write(f"- {style} / {domain} / {bucket}\n")
                if len(missing_combos) > 50:
                    f.write(f"- ... (总计 {len(missing_combos)} 个缺失桶)\n")
            else:
                f.write("- 未发现完全缺失桶\n")
    except Exception as exc:
        print(f"Failed to write report: {exc}")
        raise


def normalize_stats_keys(
    stats_counter: Dict[Tuple[str, str, str, str, str], int]
) -> Dict[Tuple[str, str, str, str, str], int]:
    """Aggregate split-level stats into ALL split for report convenience."""
    aggregated: Dict[Tuple[str, str, str, str, str], int] = defaultdict(int)
    for (split_file, label, style, domain, bucket), count in stats_counter.items():
        aggregated[("ALL", label, style, domain, bucket)] += count
    return aggregated


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify dataset by style/domain/length buckets.")
    parser.add_argument("--input", required=True, help="Input CSV file or directory.")
    parser.add_argument("--output-dir", required=True, help="Output directory for classified CSVs.")
    parser.add_argument("--stats-csv", required=True, help="Output CSV path for bucket stats.")
    parser.add_argument("--report", required=True, help="Output markdown report path.")
    parser.add_argument("--sleep-ms", type=int, default=0, help="Sleep ms every 1000 rows.")
    parser.add_argument("--progress-every", type=int, default=200000, help="Progress log interval.")
    args = parser.parse_args()

    input_path = args.input
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    stats_counter: Dict[Tuple[str, str, str, str, str], int] = defaultdict(int)
    label_counter: Counter = Counter()
    style_counter: Counter = Counter()
    domain_counter: Counter = Counter()
    length_counter: Counter = Counter()

    for input_file in iter_csv_files(input_path):
        output_file = os.path.join(output_dir, os.path.basename(input_file))
        enrich_csv(
            input_file=input_file,
            output_file=output_file,
            stats_counter=stats_counter,
            label_counter=label_counter,
            style_counter=style_counter,
            domain_counter=domain_counter,
            length_counter=length_counter,
            sleep_ms=args.sleep_ms,
            progress_every=args.progress_every,
        )

    stats_path = args.stats_csv
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    write_stats_csv(stats_path, stats_counter)

    aggregated = normalize_stats_keys(stats_counter)
    report_path = args.report
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    dataset_name = os.path.basename(os.path.normpath(input_path))
    write_report(
        report_path=report_path,
        dataset_name=dataset_name,
        label_counter=label_counter,
        style_counter=style_counter,
        domain_counter=domain_counter,
        length_counter=length_counter,
        stats_counter=aggregated,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
