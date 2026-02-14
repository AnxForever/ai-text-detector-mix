#!/usr/bin/env python3
"""
清洗 AI 生成数据中的"分析过程"残留。

部分模型（如 glm-4.7, gpt-4）会先输出"分析请求/需求"再输出实际内容。
此脚本尝试：
1. 检测包含分析过程的样本
2. 尝试提取分析过程之后的实际内容
3. 如果无法提取有效内容，则标记为 rejected

Usage:
    python clean_analysis_residue.py <input.jsonl> [--output cleaned.jsonl] [--rejected rejected.jsonl]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 分析过程的开始标记
ANALYSIS_START_PATTERNS = [
    r"^\d*\.*\s*\*{0,2}分析(请求|需求|要求)[：:]*\*{0,2}",
    r"^用户想要",
    r"^\*{0,2}角色[：:]",
    r"^限制条件[：:]",
    r"^内容(必须)?要求[：:]",
]

# 实际内容的开始标记（分析过程之后可能出现）
CONTENT_START_PATTERNS = [
    r"^\d*\.*\s*\*{0,2}(起草|撰写|输出|正文|公告)[：:]*\*{0,2}",
    r"^[「『【]",  # 中文引号开头
    r"^[a-zA-Z\u4e00-\u9fff]+\s*v\d+\.\d+",  # 产品名 + 版本号
    r"^(本次更新|此次更新|版本更新)",
]

# 需要清理的分析残留模式
ANALYSIS_RESIDUE_PATTERNS = [
    r"\*{1,2}分析(请求|需求|要求)[：:]*\*{0,2}.*?(?=\n\n|\Z)",
    r"\*{1,2}角色[：:].*?(?=\n)",
    r"\*{1,2}任务[：:].*?(?=\n)",
    r"\*{1,2}产品[：:].*?(?=\n)",
    r"\*{1,2}版本[：:].*?(?=\n)",
    r"\*{1,2}日期[：:].*?(?=\n)",
    r"\*{1,2}限制条件[：:].*?(?=\n\n|\Z)",
    r"\*{1,2}内容要求[：:].*?(?=\n\n|\Z)",
    r"\*{1,2}输出规则[：:].*?(?=\n)",
    r"\*{1,2}长度[：:].*?(?=\n)",
    r"\*{1,2}格式[：:].*?(?=\n)",
    r"用户想要.*?(?=\n\n)",
    r"限制条件[：:][\s\S]*?(?=\n\n|\d+\.\s*\*{0,2}(起草|撰写|输出))",
]

# 中文字符正则
CHINESE_CHAR = re.compile(r"[\u4e00-\u9fff]")


def chinese_ratio(text: str) -> float:
    """计算中文字符比例"""
    if not text:
        return 0.0
    chinese = len(CHINESE_CHAR.findall(text))
    return chinese / max(len(text), 1)


def has_analysis_pattern(text: str) -> bool:
    """检测文本是否包含分析过程模式"""
    for pattern in ANALYSIS_START_PATTERNS:
        if re.search(pattern, text, re.MULTILINE):
            return True
    return False


def extract_content_after_analysis(text: str) -> Optional[str]:
    """尝试从分析过程后提取实际内容"""
    # 方法1: 查找"起草内容"等标记后的内容
    for pattern in CONTENT_START_PATTERNS:
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            content = text[match.start():]
            # 清理开头的标记
            content = re.sub(r"^\d*\.*\s*\*{0,2}(起草|撰写|输出|正文)[：:]*\*{0,2}\s*", "", content)
            content = content.strip()
            if content and len(content) >= 50 and chinese_ratio(content) >= 0.3:
                return content

    # 方法2: 查找引号包裹的内容
    quote_match = re.search(r"[「『【](.+?)[」』】]", text, re.DOTALL)
    if quote_match:
        content = quote_match.group(1).strip()
        if content and len(content) >= 50 and chinese_ratio(content) >= 0.3:
            return content

    # 方法3: 查找双换行后的段落（可能是实际内容）
    parts = re.split(r"\n\n+", text)
    for part in parts[1:]:  # 跳过第一段（通常是分析）
        part = part.strip()
        # 检查是否像实际内容（不以分析标记开头）
        if part and len(part) >= 50:
            if not has_analysis_pattern(part) and chinese_ratio(part) >= 0.4:
                return part

    return None


def clean_analysis_residue(text: str) -> Tuple[str, bool]:
    """
    清洗文本中的分析残留。

    Returns:
        Tuple[str, bool]: (清洗后的文本, 是否进行了清洗)
    """
    original_len = len(text)
    cleaned = text

    # 移除分析残留模式
    for pattern in ANALYSIS_RESIDUE_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.MULTILINE | re.DOTALL)

    # 移除多余的空行和 Markdown 标记
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"^\s*[-*]\s*$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*\d+\.\s*$", "", cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()

    was_cleaned = len(cleaned) < original_len * 0.9  # 如果长度减少超过10%，认为进行了清洗
    return cleaned, was_cleaned


def process_record(record: Dict) -> Tuple[Optional[Dict], Optional[Dict], str]:
    """
    处理单条记录。

    Returns:
        Tuple[cleaned_record, rejected_record, action]
        action: "kept" | "cleaned" | "extracted" | "rejected"
    """
    text = record.get("text", "")

    # 检测是否包含分析过程
    if not has_analysis_pattern(text):
        return record, None, "kept"

    # 尝试提取分析后的实际内容
    extracted = extract_content_after_analysis(text)
    if extracted and len(extracted) >= 80:
        new_record = record.copy()
        new_record["text"] = extracted
        new_record["cleaned_analysis"] = 1
        new_record["original_text"] = text
        return new_record, None, "extracted"

    # 尝试清洗分析残留
    cleaned, was_cleaned = clean_analysis_residue(text)
    if was_cleaned and len(cleaned) >= 80 and chinese_ratio(cleaned) >= 0.3:
        if not has_analysis_pattern(cleaned):
            new_record = record.copy()
            new_record["text"] = cleaned
            new_record["cleaned_analysis"] = 1
            new_record["original_text"] = text
            return new_record, None, "cleaned"

    # 无法清洗，标记为 rejected
    rejected = record.copy()
    rejected["reject_reason"] = "analysis_content_only"
    return None, rejected, "rejected"


def main():
    parser = argparse.ArgumentParser(description="清洗 AI 生成数据中的分析过程残留")
    parser.add_argument("input", help="输入 JSONL 文件")
    parser.add_argument("--output", "-o", help="输出清洗后的 JSONL 文件")
    parser.add_argument("--rejected", "-r", help="输出被拒绝的 JSONL 文件")
    parser.add_argument("--dry-run", action="store_true", help="仅统计，不写入文件")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    # 默认输出路径
    output_path = Path(args.output) if args.output else input_path.with_suffix(".cleaned.jsonl")
    rejected_path = Path(args.rejected) if args.rejected else input_path.with_suffix(".analysis_rejected.jsonl")

    # 统计
    stats = {"kept": 0, "cleaned": 0, "extracted": 0, "rejected": 0, "total": 0}
    cleaned_records: List[Dict] = []
    rejected_records: List[Dict] = []

    # 处理
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            stats["total"] += 1

            cleaned, rejected, action = process_record(record)
            stats[action] += 1

            if cleaned:
                cleaned_records.append(cleaned)
            if rejected:
                rejected_records.append(rejected)

    # 输出统计
    print(f"\n=== 分析残留清洗统计 ===")
    print(f"总记录数: {stats['total']}")
    print(f"  - 保持不变: {stats['kept']} ({stats['kept']/max(stats['total'],1)*100:.1f}%)")
    print(f"  - 提取内容: {stats['extracted']} ({stats['extracted']/max(stats['total'],1)*100:.1f}%)")
    print(f"  - 清洗残留: {stats['cleaned']} ({stats['cleaned']/max(stats['total'],1)*100:.1f}%)")
    print(f"  - 无法恢复: {stats['rejected']} ({stats['rejected']/max(stats['total'],1)*100:.1f}%)")
    print(f"清洗后保留: {len(cleaned_records)}")

    if args.dry_run:
        print("\n[Dry run] 未写入文件")
        return 0

    # 写入文件
    if cleaned_records:
        with output_path.open("w", encoding="utf-8") as f:
            for record in cleaned_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"\n清洗后文件: {output_path}")

    if rejected_records:
        with rejected_path.open("w", encoding="utf-8") as f:
            for record in rejected_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"被拒绝文件: {rejected_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
