#!/usr/bin/env python3
"""
数据集最终合并脚本

功能: 将旧数据(core_v1)和新生成数据合并为最终数据集(core_v2)
用法: python scripts/data_cleaning/final_merge.py

流程:
1. 转换旧数据到v2 schema
2. 清洗新生成数据
3. 合并并去重
4. 平衡标签
5. 划分train/val/test
6. 输出最终数据集
"""

import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm


# ==================== 配置 ====================

# 输入路径
LEGACY_DIR = Path("datasets/active/core_v1")
GENERATED_DIR = Path("datasets/generated/scenario_fill/2026-01-27_10h_multi_proxies")

# 输出路径
OUTPUT_DIR = Path("datasets/active/core_v2")

# 目标配置
TARGET_TOTAL = 60000
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


# ==================== 映射规则 ====================

SOURCE_TO_SCENARIO = {
    "hc3_human": ("C", "knowledge"),
    "hc3_chatgpt": ("C", "knowledge"),
    "thucnews": ("F", "news"),
    "default": ("C", "knowledge"),
}

SOURCE_PREFIX_TO_SCENARIO = {
    "parallel_": ("A", "education"),
    "auto_": ("C", "knowledge"),
    "mgt_": ("C", "knowledge"),
    "generated_": ("C", "knowledge"),
}

SOURCE_TO_MODEL = {
    "hc3_chatgpt": "chatgpt-3.5",
    "parallel_gpt-4.1-mini": "gpt-4.1-mini",
    "parallel_deepseek-v3.2": "deepseek-v3.2",
    "parallel_Kimi-K2": "kimi-k2",
    "parallel_gemini-2.5-flash": "gemini-2.5-flash",
    "parallel_claude-haiku-4-5-20251001": "claude-haiku-4.5",
    "parallel_cursor2-gpt-5": "gpt-5",
    "parallel_claude-sonnet-4-5": "claude-sonnet-4.5",
    "parallel_qwen-max-latest": "qwen-max",
    "auto_deepseek": "deepseek",
}


# ==================== 工具函数 ====================

def generate_text_id(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()[:12]


def get_length_bucket(length: int) -> str:
    if length < 80: return "0-80"
    elif length < 200: return "80-200"
    elif length < 500: return "200-500"
    elif length < 1000: return "500-1000"
    elif length < 2000: return "1000-2000"
    else: return "2000+"


def infer_scenario_from_source(source: str) -> Tuple[str, str]:
    if source in SOURCE_TO_SCENARIO:
        return SOURCE_TO_SCENARIO[source]
    for prefix, scenario in SOURCE_PREFIX_TO_SCENARIO.items():
        if source.startswith(prefix):
            return scenario
    return SOURCE_TO_SCENARIO["default"]


def infer_style_from_text(text: str) -> str:
    if not isinstance(text, str) or len(text) < 10:
        return "explanation"

    lines = text.strip().split('\n')

    # 列表检测
    list_patterns = [
        r'^\s*[1-9]\d*[\.、]',
        r'^\s*[一二三四五六七八九十]+[、．]',
        r'^\s*[-•●◆★]\s',
    ]
    list_count = sum(1 for line in lines for p in list_patterns if re.match(p, line))
    if list_count >= 3:
        return "list"

    # 对话检测
    if re.search(r'问[:：]|答[:：]|客户[:：]|客服[:：]', text):
        return "dialogue"

    # 报告检测
    report_patterns = [r'摘要|引言|结论|总结', r'研究表明|数据显示', r'综上所述|由此可见']
    if sum(1 for p in report_patterns if re.search(p, text)) >= 2:
        return "report"

    # 指南检测
    if re.search(r'步骤|方法|指南|教程|首先.*然后.*最后', text):
        return "guide"

    return "explanation"


def infer_model_from_source(source: str, label: int) -> str:
    if label == 0:
        return None
    if source in SOURCE_TO_MODEL:
        return SOURCE_TO_MODEL[source]
    if source.startswith("parallel_"):
        return source.replace("parallel_", "")
    if source.startswith("generated_"):
        return source.replace("generated_", "")
    return "unknown"


# ==================== 转换函数 ====================

def convert_legacy_row(row: Dict) -> Dict:
    """转换旧数据行"""
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
    """转换生成数据行"""
    text = str(row.get('text', ''))
    length = len(text)

    return {
        "text_id": row.get('text_id', generate_text_id(text)),
        "text": text,
        "label": 1,
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


# ==================== 主流程 ====================

def load_legacy_data() -> pd.DataFrame:
    """加载并转换旧数据"""
    print("\n📂 加载旧数据 (core_v1)...")

    dfs = []
    for split in ['train', 'val', 'test']:
        path = LEGACY_DIR / f"{split}.csv"
        if path.exists():
            df = pd.read_csv(path, encoding='utf-8-sig')
            df['split'] = split
            dfs.append(df)
            print(f"   {split}: {len(df):,}条")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"   合计: {len(combined):,}条")

    print("🔄 转换到v2 schema...")
    converted = []
    for _, row in tqdm(combined.iterrows(), total=len(combined), desc="转换"):
        conv = convert_legacy_row(row.to_dict())
        conv['original_split'] = row.get('split', 'train')
        converted.append(conv)

    return pd.DataFrame(converted)


def load_generated_data() -> pd.DataFrame:
    """加载并转换生成数据"""
    print("\n📂 加载生成数据...")

    files = list(GENERATED_DIR.glob("ai_scenario_fill_*.jsonl"))
    files = [f for f in files if 'rejected' not in f.name]

    records = []
    for f in tqdm(files, desc="读取文件"):
        with open(f, 'r', encoding='utf-8') as fp:
            for line in fp:
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except:
                        pass

    print(f"   读取: {len(records):,}条")

    print("🔄 转换到v2 schema...")
    converted = [convert_generated_row(r) for r in tqdm(records, desc="转换")]

    return pd.DataFrame(converted)


def merge_and_dedupe(legacy_df: pd.DataFrame, gen_df: pd.DataFrame) -> pd.DataFrame:
    """合并并去重"""
    print("\n🔗 合并数据...")
    print(f"   旧数据: {len(legacy_df):,}")
    print(f"   新数据: {len(gen_df):,}")

    merged = pd.concat([legacy_df, gen_df], ignore_index=True)
    print(f"   合并后: {len(merged):,}")

    print("🔄 去重...")
    before = len(merged)
    merged = merged.drop_duplicates(subset=['text_id'], keep='last')
    after = len(merged)
    print(f"   去重后: {after:,} (移除 {before - after:,})")

    return merged


def balance_dataset(df: pd.DataFrame, target: int = None) -> pd.DataFrame:
    """平衡数据集"""
    print("\n⚖️ 平衡数据集...")

    ai_count = (df['label'] == 1).sum()
    human_count = (df['label'] == 0).sum()
    print(f"   平衡前: AI={ai_count:,}, Human={human_count:,}")

    if target:
        each = target // 2
    else:
        each = min(ai_count, human_count)

    ai_df = df[df['label'] == 1].sample(n=min(each, ai_count), random_state=42)
    human_df = df[df['label'] == 0].sample(n=min(each, human_count), random_state=42)

    balanced = pd.concat([ai_df, human_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"   平衡后: {len(balanced):,} (AI={len(ai_df):,}, Human={len(human_df):,})")

    return balanced


def split_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """划分数据集"""
    print("\n✂️ 划分数据集...")

    # 打乱
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    print(f"   训练集: {len(train_df):,} ({len(train_df)/n*100:.1f}%)")
    print(f"   验证集: {len(val_df):,} ({len(val_df)/n*100:.1f}%)")
    print(f"   测试集: {len(test_df):,} ({len(test_df)/n*100:.1f}%)")

    return train_df, val_df, test_df


def save_datasets(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """保存数据集"""
    print("\n💾 保存数据集...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False, encoding='utf-8-sig')
    val_df.to_csv(OUTPUT_DIR / "val.csv", index=False, encoding='utf-8-sig')
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False, encoding='utf-8-sig')

    # 合并版本
    all_df = pd.concat([train_df, val_df, test_df])
    all_df.to_csv(OUTPUT_DIR / "all.csv", index=False, encoding='utf-8-sig')

    print(f"   保存到: {OUTPUT_DIR}")


def print_stats(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """打印最终统计"""
    print("\n" + "=" * 65)
    print("📊 最终数据集统计")
    print("=" * 65)

    all_df = pd.concat([train_df, val_df, test_df])

    print(f"\n总样本数: {len(all_df):,}")
    print(f"训练/验证/测试: {len(train_df):,} / {len(val_df):,} / {len(test_df):,}")

    print(f"\n标签分布:")
    print(f"  Human: {(all_df['label']==0).sum():,} ({(all_df['label']==0).sum()/len(all_df)*100:.1f}%)")
    print(f"  AI:    {(all_df['label']==1).sum():,} ({(all_df['label']==1).sum()/len(all_df)*100:.1f}%)")

    print(f"\n场景分布:")
    for sid in ['A', 'B', 'C', 'D', 'E', 'F']:
        cnt = (all_df['scenario_id'] == sid).sum()
        print(f"  {sid}: {cnt:,} ({cnt/len(all_df)*100:.1f}%)")

    print(f"\n来源类型:")
    for stype, cnt in all_df['source_type'].value_counts().items():
        print(f"  {stype}: {cnt:,} ({cnt/len(all_df)*100:.1f}%)")


def main():
    print("=" * 65)
    print("📦 数据集最终合并工具")
    print("=" * 65)

    # 1. 加载数据
    legacy_df = load_legacy_data()
    gen_df = load_generated_data()

    # 2. 合并去重
    merged_df = merge_and_dedupe(legacy_df, gen_df)

    # 3. 平衡
    balanced_df = balance_dataset(merged_df, TARGET_TOTAL)

    # 4. 划分
    train_df, val_df, test_df = split_dataset(balanced_df)

    # 5. 保存
    save_datasets(train_df, val_df, test_df)

    # 6. 统计
    print_stats(train_df, val_df, test_df)

    print("\n" + "=" * 65)
    print("✅ 数据集构建完成！")
    print("=" * 65)


if __name__ == "__main__":
    main()
