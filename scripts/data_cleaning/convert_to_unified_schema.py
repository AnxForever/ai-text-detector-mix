#!/usr/bin/env python3
"""
数据集Schema转换与合并工具

功能:
1. 将旧数据(core_v1)转换为统一的v2 schema
2. 将新生成数据转换为v2 schema
3. 合并、去重、平衡

用法:
    # 转换旧数据
    python scripts/data_cleaning/convert_to_unified_schema.py \
        --legacy datasets/active/core_v1/train.csv \
        --output datasets/active/core_v2/legacy_converted.csv

    # 转换生成数据
    python scripts/data_cleaning/convert_to_unified_schema.py \
        --generated datasets/generated/scenario_fill/xxx/cleaned.jsonl \
        --output datasets/active/core_v2/generated_converted.csv

    # 合并
    python scripts/data_cleaning/convert_to_unified_schema.py \
        --merge datasets/active/core_v2/legacy_converted.csv \
                datasets/active/core_v2/generated_converted.csv \
        --output datasets/active/core_v2/merged.csv \
        --balance
"""

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm


# ==================== 映射规则 ====================

# Source → (Scenario_ID, Scenario)
SOURCE_TO_SCENARIO = {
    # HC3数据 → C (知识百科)
    "hc3_human": ("C", "knowledge"),
    "hc3_chatgpt": ("C", "knowledge"),

    # THUCNews → F (新闻资讯)
    "thucnews": ("F", "news"),

    # 默认规则
    "default": ("C", "knowledge"),
}

# Source前缀 → Scenario (用于parallel_*, auto_*等)
SOURCE_PREFIX_TO_SCENARIO = {
    "parallel_": ("A", "education"),    # 学术/专业改写
    "auto_": ("C", "knowledge"),        # 自动生成
    "mgt_": ("C", "knowledge"),         # MGTBench
}

# Source → Model
SOURCE_TO_MODEL = {
    "hc3_chatgpt": "chatgpt-3.5",
    "hc3_human": None,
    "thucnews": None,
    "parallel_gpt-4.1-mini": "gpt-4.1-mini",
    "parallel_deepseek-v3.2": "deepseek-v3.2",
    "parallel_deepseek-v3.2-chat": "deepseek-v3.2",
    "parallel_Kimi-K2": "kimi-k2",
    "parallel_gemini-2.5-flash": "gemini-2.5-flash",
    "parallel_claude-haiku-4-5-20251001": "claude-haiku-4.5",
    "parallel_cursor2-gpt-5": "gpt-5",
    "parallel_claude-sonnet-4-5": "claude-sonnet-4.5",
    "parallel_claude-sonnet-4-5-20250929": "claude-sonnet-4.5",
    "parallel_qwen-max-latest": "qwen-max",
    "auto_deepseek": "deepseek",
    "auto_custom": "unknown",
}


# ==================== 推断函数 ====================

def infer_scenario_from_source(source: str) -> Tuple[str, str]:
    """从source字段推断scenario"""
    # 精确匹配
    if source in SOURCE_TO_SCENARIO:
        return SOURCE_TO_SCENARIO[source]

    # 前缀匹配
    for prefix, scenario in SOURCE_PREFIX_TO_SCENARIO.items():
        if source.startswith(prefix):
            return scenario

    # 默认
    return SOURCE_TO_SCENARIO["default"]


def infer_model_from_source(source: str, label: int) -> Optional[str]:
    """从source字段推断model"""
    if label == 0:  # Human
        return None

    # 精确匹配
    if source in SOURCE_TO_MODEL:
        return SOURCE_TO_MODEL[source]

    # 从source提取模型名
    if source.startswith("parallel_"):
        return source.replace("parallel_", "")
    if source.startswith("auto_"):
        return source.replace("auto_", "") or "unknown"

    return "unknown"


def infer_style_from_text(text: str) -> str:
    """基于文本特征推断style"""
    if not isinstance(text, str) or len(text) < 10:
        return "explanation"

    lines = text.strip().split('\n')

    # 列表特征检测
    list_patterns = [
        r'^\s*[1-9]\d*[\.、]',           # 1. 2. 或 1、2、
        r'^\s*[一二三四五六七八九十]+[、．]',  # 一、二、
        r'^\s*[-•●◆★]\s',               # 无序列表符号
        r'^\s*第[一二三四五六七八九十]',    # 第一、第二
        r'^\s*\([1-9]\)',                # (1) (2)
    ]

    list_line_count = 0
    for line in lines:
        for pattern in list_patterns:
            if re.match(pattern, line):
                list_line_count += 1
                break

    if list_line_count >= 3:
        return "list"

    # 对话特征检测
    dialogue_patterns = [
        r'[""「」『』].*?[""」」』』]',     # 引号对话
        r'^\s*[甲乙丙丁ABCD][:：]\s',     # 角色对话
        r'问[:：]|答[:：]',               # 问答形式
        r'客户[:：]|客服[:：]',           # 客服对话
    ]

    for pattern in dialogue_patterns:
        if re.search(pattern, text, re.MULTILINE):
            return "dialogue"

    # 报告/正式特征
    report_patterns = [
        r'摘要|引言|背景|结论|总结|概述',
        r'研究表明|数据显示|分析发现|调查结果',
        r'综上所述|由此可见|总而言之',
        r'一、.*二、|第一章|第一节',
    ]

    report_matches = sum(1 for p in report_patterns if re.search(p, text))
    if report_matches >= 2:
        return "report"

    # 指南特征
    guide_patterns = [
        r'步骤|方法|技巧|指南|教程',
        r'首先.*然后.*最后',
        r'注意事项|温馨提示|小贴士',
    ]

    if any(re.search(p, text) for p in guide_patterns):
        return "guide"

    # 默认
    return "explanation"


def get_length_bucket(length: int) -> str:
    """获取长度桶"""
    if length < 80:
        return "0-80"
    elif length < 200:
        return "80-200"
    elif length < 500:
        return "200-500"
    elif length < 1000:
        return "500-1000"
    elif length < 2000:
        return "1000-2000"
    else:
        return "2000+"


def generate_text_id(text: str) -> str:
    """生成文本唯一ID"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()[:12]


# ==================== 转换函数 ====================

def convert_legacy_row(row: Dict) -> Dict:
    """将旧数据行转换为v2 schema"""
    text = str(row.get('text', ''))
    source = str(row.get('source', 'unknown'))
    label = int(row.get('label', 0))
    length = int(row.get('length', len(text)))

    scenario_id, scenario = infer_scenario_from_source(source)
    style = infer_style_from_text(text)
    model = infer_model_from_source(source, label)

    return {
        "text_id": generate_text_id(text),
        "text": text,
        "label": label,
        "scenario_id": scenario_id,
        "scenario": scenario,
        "style": style,
        "length": length,
        "length_bucket": get_length_bucket(length),
        "source": source,
        "source_type": "legacy",
        "model": model,
        "created_at": "2026-01-27T00:00:00",
        "schema_version": "v2",
    }


def convert_generated_row(row: Dict) -> Dict:
    """将生成数据行转换为v2 schema"""
    text = str(row.get('text', ''))
    length = len(text)

    return {
        "text_id": row.get('text_id', generate_text_id(text)),
        "text": text,
        "label": 1,  # 生成数据都是AI
        "scenario_id": row.get('scenario_id', 'C'),
        "scenario": row.get('scenario', 'knowledge'),
        "style": row.get('style_plan', row.get('style', 'explanation')),
        "length": length,
        "length_bucket": row.get('length_bucket', get_length_bucket(length)),
        "source": f"generated_{row.get('model', 'unknown')}",
        "source_type": "generated",
        "model": row.get('model', 'unknown'),
        "created_at": row.get('created_at', datetime.now().isoformat()),
        "schema_version": "v2",
    }


# ==================== 主功能函数 ====================

def convert_legacy_data(input_path: Path, output_path: Path) -> pd.DataFrame:
    """转换旧数据"""
    print(f"📂 加载旧数据: {input_path}")
    df = pd.read_csv(input_path, encoding='utf-8-sig')
    print(f"   总行数: {len(df)}")

    print("🔄 转换中...")
    converted = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="转换"):
        converted.append(convert_legacy_row(row.to_dict()))

    result_df = pd.DataFrame(converted)

    # 统计
    print(f"\n📊 转换结果统计:")
    print(f"   场景分布: {result_df['scenario_id'].value_counts().to_dict()}")
    print(f"   样式分布: {result_df['style'].value_counts().to_dict()}")
    print(f"   标签分布: {result_df['label'].value_counts().to_dict()}")

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 已保存: {output_path}")

    return result_df


def convert_generated_data(input_path: Path, output_path: Path) -> pd.DataFrame:
    """转换生成数据"""
    print(f"📂 加载生成数据: {input_path}")

    records = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except:
                    pass

    print(f"   总行数: {len(records)}")

    print("🔄 转换中...")
    converted = []
    for row in tqdm(records, desc="转换"):
        converted.append(convert_generated_row(row))

    result_df = pd.DataFrame(converted)

    # 统计
    print(f"\n📊 转换结果统计:")
    print(f"   场景分布: {result_df['scenario_id'].value_counts().to_dict()}")
    print(f"   样式分布: {result_df['style'].value_counts().to_dict()}")
    print(f"   模型分布: {result_df['model'].value_counts().head(5).to_dict()}")

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 已保存: {output_path}")

    return result_df


def merge_datasets(
    input_paths: List[Path],
    output_path: Path,
    balance: bool = False,
    dedupe: bool = True
) -> pd.DataFrame:
    """合并多个数据集"""
    print("📂 加载数据集...")
    dfs = []
    for path in input_paths:
        print(f"   加载: {path}")
        df = pd.read_csv(path, encoding='utf-8-sig')
        print(f"      行数: {len(df)}")
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    print(f"\n📊 合并后总数: {len(merged)}")

    # 去重
    if dedupe:
        before = len(merged)
        merged = merged.drop_duplicates(subset=['text_id'], keep='last')  # 保留后来的（新数据）
        after = len(merged)
        print(f"🔄 去重: {before} → {after} (移除 {before - after})")

    # 平衡
    if balance:
        ai_count = (merged['label'] == 1).sum()
        human_count = (merged['label'] == 0).sum()
        print(f"\n⚖️ 平衡前: AI={ai_count}, Human={human_count}")

        target = min(ai_count, human_count)
        ai_df = merged[merged['label'] == 1].sample(n=target, random_state=42)
        human_df = merged[merged['label'] == 0].sample(n=target, random_state=42)
        merged = pd.concat([ai_df, human_df]).sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"⚖️ 平衡后: AI={target}, Human={target}, 总计={len(merged)}")

    # 最终统计
    print(f"\n📊 最终数据集统计:")
    print(f"   总数: {len(merged)}")
    print(f"   标签: {merged['label'].value_counts().to_dict()}")
    print(f"   场景: {merged['scenario_id'].value_counts().to_dict()}")
    print(f"   样式: {merged['style'].value_counts().to_dict()}")
    print(f"   来源类型: {merged['source_type'].value_counts().to_dict()}")

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 已保存: {output_path}")

    return merged


def main():
    parser = argparse.ArgumentParser(description="数据集Schema转换与合并")

    # 转换模式
    parser.add_argument("--legacy", type=Path, help="旧数据CSV路径")
    parser.add_argument("--generated", type=Path, help="生成数据JSONL路径")

    # 合并模式
    parser.add_argument("--merge", nargs="+", type=Path, help="要合并的CSV文件列表")

    # 输出
    parser.add_argument("--output", "-o", type=Path, required=True, help="输出路径")

    # 选项
    parser.add_argument("--balance", action="store_true", help="平衡AI/Human比例")
    parser.add_argument("--no-dedupe", action="store_true", help="不去重")

    args = parser.parse_args()

    print("=" * 60)
    print("📦 数据集Schema转换与合并工具")
    print("=" * 60)

    if args.legacy:
        convert_legacy_data(args.legacy, args.output)

    elif args.generated:
        convert_generated_data(args.generated, args.output)

    elif args.merge:
        merge_datasets(
            args.merge,
            args.output,
            balance=args.balance,
            dedupe=not args.no_dedupe
        )

    else:
        print("❌ 请指定 --legacy, --generated 或 --merge")
        sys.exit(1)

    print("\n✅ 完成!")


if __name__ == "__main__":
    main()
