"""Convert existing dataset CSV into unified schema format.

This script is intended for schema alignment and sample export.
It does not generate new data.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime
from typing import Dict, List


os.environ.setdefault("PYTHONIOENCODING", "utf-8")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(PROJECT_ROOT)


OUTPUT_FIELDS = [
    "id",
    "text",
    "y_main",
    "label",
    "style",
    "domain",
    "length_bucket",
    "length_chars",
    "source_type",
    "source_id",
    "collected_at",
    "model_family",
    "model_name",
    "prompt_id",
    "decoding",
    "seed",
    "q_score",
    "d_score",
    "y_conf",
    "q_flags",
    "y_evidence",
    "routed_pool",
    "segment_annotations",
    "boundary_metrics",
]

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


def normalize_text(text: str, max_chars: int = 4000) -> str:
    """Normalize text for rule checks."""
    if not text:
        return ""
    return text[:max_chars]


def length_bucket(length_value: int) -> str:
    """Map raw character length to bucket."""
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
    """Detect style using simple rules."""
    sample = normalize_text(text)
    lower = sample.lower()
    lines = [line.strip() for line in sample.splitlines() if line.strip()]

    if any(key in lower for key in STYLE_KEYWORDS["dialogue"]):
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
    for style, score in scores.items():
        if score > best_score:
            best_style = style
            best_score = score
    return best_style


def detect_domain(text: str) -> str:
    """Detect domain based on keywords."""
    sample = normalize_text(text).lower()
    scores: Dict[str, int] = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if not keywords:
            continue
        scores[domain] = sum(1 for kw in keywords if kw in sample)
    if not scores:
        return "general"
    best_domain, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score == 0:
        return "general"
    return best_domain


def map_label_to_main(label_value: str) -> str:
    """Map raw label to HUMAN/AI/UNCERTAIN."""
    label_value = str(label_value).strip().lower()
    if label_value in {"1", "ai", "machine"}:
        return "AI"
    if label_value in {"0", "human"}:
        return "HUMAN"
    return "UNCERTAIN"


def default_source_type(y_main: str) -> str:
    """Infer source type from label when missing."""
    if y_main == "AI":
        return "ai_generated"
    if y_main == "HUMAN":
        return "unknown"
    return "unknown"


def build_row(
    row: Dict[str, str],
    dataset_name: str,
    split_name: str,
    row_index: int,
    collected_at: str,
) -> Dict[str, str]:
    """Build schema-compliant row."""
    text = row.get("text", "") or ""
    label_raw = row.get("label", "")

    y_main = row.get("y_main") or map_label_to_main(label_raw)
    length_chars = row.get("length_chars")
    if not length_chars:
        length_chars = str(len(str(text).strip()))

    length_bucket_value = row.get("length_bucket")
    if not length_bucket_value:
        length_bucket_value = length_bucket(int(length_chars))

    style_value = row.get("style") or detect_style(text)
    domain_value = row.get("domain") or detect_domain(text)

    source_id = row.get("source_id") or row.get("source", "") or ""
    source_type = row.get("source_type") or default_source_type(y_main)

    output = {
        "id": row.get("id") or f"{dataset_name}:{split_name}:{row_index}",
        "text": text,
        "y_main": y_main,
        "label": label_raw,
        "style": style_value,
        "domain": domain_value,
        "length_bucket": length_bucket_value,
        "length_chars": length_chars,
        "source_type": source_type,
        "source_id": source_id,
        "collected_at": row.get("collected_at") or collected_at,
        "model_family": row.get("model_family", ""),
        "model_name": row.get("model_name", ""),
        "prompt_id": row.get("prompt_id", ""),
        "decoding": row.get("decoding", ""),
        "seed": row.get("seed", ""),
        "q_score": row.get("q_score", ""),
        "d_score": row.get("d_score", ""),
        "y_conf": row.get("y_conf", ""),
        "q_flags": row.get("q_flags", ""),
        "y_evidence": row.get("y_evidence", ""),
        "routed_pool": row.get("routed_pool", ""),
        "segment_annotations": row.get("segment_annotations", ""),
        "boundary_metrics": row.get("boundary_metrics", ""),
    }
    return output


def convert_file(
    input_path: str,
    output_path: str,
    dataset_name: str,
    split_name: str,
    max_rows: int,
    collected_at: str,
) -> int:
    """Convert a single CSV file into schema format."""
    rows_written = 0
    with open(input_path, "r", encoding="utf-8", errors="replace", newline="") as f_in:
        reader = csv.DictReader(f_in)
        if not reader.fieldnames:
            return 0
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            for idx, row in enumerate(reader):
                output_row = build_row(row, dataset_name, split_name, idx, collected_at)
                writer.writerow(output_row)
                rows_written += 1
                if max_rows and rows_written >= max_rows:
                    break
    return rows_written


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert CSV to unified schema format.")
    parser.add_argument("--input", required=True, help="Input CSV file.")
    parser.add_argument("--output", required=True, help="Output CSV file.")
    parser.add_argument("--dataset-name", required=True, help="Dataset name for ID.")
    parser.add_argument("--split-name", required=True, help="Split name for ID.")
    parser.add_argument("--max-rows", type=int, default=0, help="Max rows to write (0=all).")
    parser.add_argument("--collected-at", default="unknown", help="Default collected_at value.")
    args = parser.parse_args()

    max_rows = args.max_rows if args.max_rows > 0 else 0
    rows_written = convert_file(
        input_path=args.input,
        output_path=args.output,
        dataset_name=args.dataset_name,
        split_name=args.split_name,
        max_rows=max_rows,
        collected_at=args.collected_at,
    )
    print(f"{args.output} rows={rows_written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
