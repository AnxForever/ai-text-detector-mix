#!/usr/bin/env python3
"""
统一清洗生成数据：分析残留检测 + 长度修复 + 质量验证。

此脚本对生成的 JSONL 数据进行统一清洗：
1. 检测并移除"分析过程"内容
2. 尝试截断过长文本到合适长度
3. 验证最终内容质量

Usage:
    python unified_clean.py <input_dir_or_file> --output <output_dir>

Examples:
    # 清洗单个文件
    python unified_clean.py data.jsonl --output cleaned/

    # 清洗目录下所有 part*.jsonl 文件
    python unified_clean.py datasets/generated/scenario_fill/2026-01-27_10h_multi_proxies/ --output cleaned/
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ============== 配置 ==============

# 分析过程的标记模式（更严格）
# 注意：只匹配分析残留的典型格式，避免误伤正文中的合理词汇
ANALYSIS_PATTERNS = [
    # Markdown 格式的分析标记
    "**分析请求",
    "**分析需求",
    "**分析要求",
    "**限制与要求",
    "**限制条件",
    "**内容要求",
    "**输出规则",
    "* **角色：**",
    "* **任务：**",
    "* **产品：**",
    "*   **角色：**",
    "*   **任务：**",
    # 纯文本格式的分析标记（必须是句首或特定格式）
    "用户想要一",
    "用户想要一篇",
    "用户想要一份",
    "限制条件：",      # 带冒号的才是分析残留
    "限制条件如下",    # 明确的分析格式
    "限制与要求：",
    "内容要求：",
    "内容必须包含：",
    "约束条件如下",    # 只匹配"如下"格式，避免误伤正文中的"约束条件"
    "约束条件：",      # 带冒号的才是分析残留
    # 指令复述
    "仅输出最终文本",
    "仅输出最终正文",
    "仅输出正文",
    "不包含分析",
    "不要分析",
    "不使用Markdown标题",
    "无Markdown标题",
]

# 长度范围（放宽版）
LENGTH_BUCKET_RANGES = {
    "80-200": (80, 350),
    "200-500": (160, 720),
    "500-1000": (360, 1320),
    "1000-2000": (720, 2640),
    "2000+": (1440, 4080),
}

# 截断目标长度
LENGTH_BUCKET_TRUNCATE = {
    "80-200": 300,
    "200-500": 650,
    "500-1000": 1200,
    "1000-2000": 2400,
    "2000+": 3800,
}

# 中文字符
CHINESE_CHAR = re.compile(r"[\u4e00-\u9fff]")
SENTENCE_END = re.compile(r"([。！？!?])")


# ============== 工具函数 ==============

def chinese_ratio(text: str) -> float:
    """计算中文字符比例"""
    if not text:
        return 0.0
    chinese = len(CHINESE_CHAR.findall(text))
    return chinese / max(len(text), 1)


def has_analysis_pattern(text: str) -> bool:
    """检测文本是否包含分析过程模式"""
    return any(pattern in text for pattern in ANALYSIS_PATTERNS)


def smart_truncate(text: str, target_len: int, min_len: int = 50) -> str:
    """智能截断：在句子边界处截断"""
    if len(text) <= target_len:
        return text

    # 按句子拆分
    parts = SENTENCE_END.split(text)
    sentences = []
    for i in range(0, len(parts), 2):
        sent = parts[i]
        punct = parts[i + 1] if i + 1 < len(parts) else ""
        if sent.strip():
            sentences.append(sent + punct)

    # 累积到目标长度
    result = []
    current_len = 0
    for sent in sentences:
        if current_len + len(sent) <= target_len:
            result.append(sent)
            current_len += len(sent)
        else:
            break

    if result:
        return "".join(result).strip()

    # 回退：硬截断
    for i in range(min(len(text), target_len) - 1, min_len, -1):
        if text[i] in "。！？!?.;；，,":
            return text[: i + 1]
    return text[:target_len]


def validate_text(text: str, length_bucket: str) -> Tuple[bool, str]:
    """验证文本质量"""
    if not text:
        return False, "empty"
    if has_analysis_pattern(text):
        return False, "analysis_residue"
    if chinese_ratio(text) < 0.3:
        return False, "low_chinese_ratio"
    min_len, max_len = LENGTH_BUCKET_RANGES.get(length_bucket, (0, 10_000))
    if len(text) < min_len:
        return False, "too_short"
    if len(text) > max_len:
        return False, "too_long"
    return True, "ok"


# ============== 清洗逻辑 ==============

def clean_record(record: Dict) -> Tuple[Optional[Dict], Optional[Dict], str]:
    """
    清洗单条记录。

    Returns:
        (cleaned_record, rejected_record, action)
        action: kept | truncated | rejected_analysis | rejected_quality
    """
    text = record.get("text", "")
    length_bucket = record.get("length_bucket", "200-500")

    # 1. 检测分析残留
    if has_analysis_pattern(text):
        rejected = record.copy()
        rejected["reject_reason"] = "analysis_residue"
        return None, rejected, "rejected_analysis"

    # 2. 检查长度，尝试截断
    min_len, max_len = LENGTH_BUCKET_RANGES.get(length_bucket, (0, 10_000))
    target_len = LENGTH_BUCKET_TRUNCATE.get(length_bucket, max_len)

    cleaned_text = text
    was_truncated = False

    if len(text) > max_len:
        cleaned_text = smart_truncate(text, target_len)
        was_truncated = True

    # 3. 最终验证
    valid, reason = validate_text(cleaned_text, length_bucket)
    if not valid:
        rejected = record.copy()
        rejected["reject_reason"] = reason
        return None, rejected, f"rejected_{reason}"

    # 4. 返回清洗后的记录
    new_record = record.copy()
    if was_truncated:
        new_record["text"] = cleaned_text
        new_record["truncated"] = 1
        new_record["original_length"] = len(text)
        return new_record, None, "truncated"

    return new_record, None, "kept"


def process_file(input_path: Path, output_dir: Path) -> Dict:
    """处理单个文件"""
    stats = {
        "total": 0,
        "kept": 0,
        "truncated": 0,
        "rejected_analysis": 0,
        "rejected_quality": 0,
    }

    cleaned_records = []
    rejected_records = []

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            stats["total"] += 1

            cleaned, rejected, action = clean_record(record)

            if "rejected" in action:
                if "analysis" in action:
                    stats["rejected_analysis"] += 1
                else:
                    stats["rejected_quality"] += 1
                if rejected:
                    rejected_records.append(rejected)
            elif action == "truncated":
                stats["truncated"] += 1
                if cleaned:
                    cleaned_records.append(cleaned)
            else:
                stats["kept"] += 1
                if cleaned:
                    cleaned_records.append(cleaned)

    # 写入文件
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    if cleaned_records:
        out_path = output_dir / f"{stem}_cleaned.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for r in cleaned_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats["output_file"] = str(out_path)

    if rejected_records:
        rej_path = output_dir / f"{stem}_rejected.jsonl"
        with rej_path.open("w", encoding="utf-8") as f:
            for r in rejected_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats["rejected_file"] = str(rej_path)

    stats["cleaned_count"] = len(cleaned_records)
    return stats


def main():
    parser = argparse.ArgumentParser(description="统一清洗生成数据")
    parser.add_argument("input", help="输入文件或目录")
    parser.add_argument("--output", "-o", required=True, help="输出目录")
    parser.add_argument("--pattern", default="*part*.jsonl", help="文件匹配模式（目录模式）")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    # 收集文件
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = sorted(input_path.glob(args.pattern))
    else:
        print(f"Error: Input not found: {input_path}")
        return 1

    if not files:
        print(f"Error: No files found matching pattern: {args.pattern}")
        return 1

    print(f"=== 统一数据清洗 ===")
    print(f"输入: {input_path}")
    print(f"文件数: {len(files)}")
    print(f"输出目录: {output_dir}")
    print()

    # 处理文件
    total_stats = {
        "total": 0,
        "kept": 0,
        "truncated": 0,
        "rejected_analysis": 0,
        "rejected_quality": 0,
        "cleaned_count": 0,
    }

    for file_path in files:
        print(f"处理: {file_path.name}...", end=" ")
        stats = process_file(file_path, output_dir)

        for key in total_stats:
            if key in stats:
                total_stats[key] += stats[key]

        print(f"✓ kept={stats['kept']} trunc={stats['truncated']} "
              f"rej_analysis={stats['rejected_analysis']} rej_quality={stats['rejected_quality']}")

    # 总结
    print()
    print(f"=== 清洗完成 ===")
    print(f"总记录数: {total_stats['total']}")
    print(f"  - 保持不变: {total_stats['kept']} ({total_stats['kept']/max(total_stats['total'],1)*100:.1f}%)")
    print(f"  - 已截断:   {total_stats['truncated']} ({total_stats['truncated']/max(total_stats['total'],1)*100:.1f}%)")
    print(f"  - 分析残留: {total_stats['rejected_analysis']} ({total_stats['rejected_analysis']/max(total_stats['total'],1)*100:.1f}%)")
    print(f"  - 质量问题: {total_stats['rejected_quality']} ({total_stats['rejected_quality']/max(total_stats['total'],1)*100:.1f}%)")
    print(f"清洗后保留: {total_stats['cleaned_count']}")
    print(f"输出目录: {output_dir}")

    # 写入汇总
    summary = {
        "processed_at": datetime.now().isoformat(),
        "input": str(input_path),
        "files_processed": len(files),
        **total_stats,
    }
    summary_path = output_dir / "clean_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
