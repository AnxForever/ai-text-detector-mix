"""
V10 数据准备脚本 - 方案 α 纯数据增强

目标：解决两个核心问题
1. FN (AI→Human 漏检): 技术/教育类 AI 文本被误判为人类 → +500 education AI
2. FP (Human→AI 误判): 短文本 Human 被误判为 AI → +300 diverse short Human + +500 Human technical (平衡)

数据来源：
- Education AI: my_generated_ai 未用于训练的 education 场景 (3,475 条可用)
- Human Technical: HC3 baike(≥128) + CSL academic + HC3 medicine(≥128)
- Short Human: HC3 各类别 <128 chars (多源采样)

输出：datasets/merged_v2/train_v10.csv
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

np.random.seed(42)

BASE = Path("/mnt/c/datacollection")
TRAIN_PATH = BASE / "datasets/merged_v2/train.csv"
OUTPUT_PATH = BASE / "datasets/merged_v2/train_v10.csv"

print("=" * 60)
print("V10 数据准备 - 方案 α 纯数据增强")
print("=" * 60)

# ── 加载现有训练集 ──
train = pd.read_csv(TRAIN_PATH)
train_prefixes_200 = set(str(t)[:200] for t in train["text"].values)

# 也加载评估集，确保零重叠
eval_df = pd.read_csv(BASE / "datasets/eval/fair_test/independent_data.csv")
eval_prefixes_200 = set(str(t)[:200] for t in eval_df["text"].values)

all_exclude = train_prefixes_200 | eval_prefixes_200

print(f"\n现有训练集: {len(train)} 样本 (AI: {(train['label']==1).sum()}, Human: {(train['label']==0).sum()})")
print(f"评估集: {len(eval_df)} 样本")


# ════════════════════════════════════════════════════════════
# Part 1: +500 Education AI (from my_generated_ai unused)
# ════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("Part 1: Education AI 样本 (目标 500)")
print("─" * 60)

education_unused = []
with open(BASE / "datasets/my_generated_ai/all_generated.jsonl", "r") as f:
    for line in f:
        d = json.loads(line)
        if d.get("scenario") == "education" and str(d.get("text", ""))[:200] not in all_exclude:
            education_unused.append(d)

print(f"可用 education 样本: {len(education_unused)}")

# 按模型分层采样，优先弱模型（GPT-5, DeepSeek-v3.2, LLaMA-405B, GPT-OSS-120B）
model_groups = {}
for d in education_unused:
    model = d.get("model", "unknown")
    # 归一化模型名
    if "gpt-5" in model.lower():
        key = "gpt-5"
    elif "deepseek" in model.lower():
        key = "deepseek-v3.2"
    elif "llama" in model.lower():
        key = "llama-405b"
    elif "gpt-oss" in model.lower():
        key = "gpt-oss-120b"
    elif "gpt-4" in model.lower():
        key = "gpt-4"
    elif "glm" in model.lower():
        key = "glm-4.7"
    elif "gemini" in model.lower():
        key = "gemini"
    else:
        key = model
    model_groups.setdefault(key, []).append(d)

# 采样策略：弱模型优先（评估集检出率低的模型多采）
priority_models = {
    "gpt-5": 80,          # 评估集 75%, 最弱
    "deepseek-v3.2": 100, # 评估集 75%, 最弱 + 样本充足
    "llama-405b": 80,     # 评估集 88.89%
    "gpt-oss-120b": 80,   # 评估集 87.50%
    "gpt-4": 50,          # 评估集 100% 但增加覆盖
    "glm-4.7": 50,        # 评估集 100% 但增加覆盖
    "gemini": 60,         # 评估集 87.50-100%
}

education_ai_samples = []
for model_key, target_n in priority_models.items():
    available = model_groups.get(model_key, [])
    n = min(target_n, len(available))
    if n > 0:
        selected = np.random.choice(len(available), n, replace=False)
        for idx in selected:
            d = available[idx]
            education_ai_samples.append({
                "text": d["text"],
                "label": 1,
                "source": f"v10_edu_ai_{model_key}",
                "category": "AI",
                "type": d.get("scenario", "education"),
            })
    print(f"  {model_key}: {n}/{len(available)} sampled")

print(f"Education AI 总计: {len(education_ai_samples)}")


# ════════════════════════════════════════════════════════════
# Part 2: +500 Human Technical (HC3 baike + CSL + HC3 medicine)
# ════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("Part 2: Human Technical 样本 (目标 500)")
print("─" * 60)

human_tech_pool = []

# Source 1: HC3 baike (百科知识) ≥128 chars
with open(BASE / "datasets/external/HC3-Chinese/baike.jsonl", "r") as f:
    for line in f:
        d = json.loads(line)
        for ans in d.get("human_answers", []):
            if len(str(ans)) >= 128 and str(ans)[:200] not in all_exclude:
                human_tech_pool.append({
                    "text": str(ans),
                    "source_tag": "hc3_baike_tech",
                    "length": len(str(ans)),
                })

print(f"  HC3 baike (≥128): {len(human_tech_pool)} available")

# Source 2: CSL academic (学术摘要)
csl = pd.read_csv(BASE / "datasets/external/processed/csl_academic_human.csv")
csl_count = 0
for _, row in csl.iterrows():
    text = str(row["text"])
    if text[:200] not in all_exclude and len(text) >= 100:
        human_tech_pool.append({
            "text": text,
            "source_tag": "csl_academic",
            "length": len(text),
        })
        csl_count += 1

print(f"  CSL academic (≥100): {csl_count} available")

# Source 3: HC3 medicine (医学) ≥128 chars
with open(BASE / "datasets/external/HC3-Chinese/medicine.jsonl", "r") as f:
    for line in f:
        d = json.loads(line)
        for ans in d.get("human_answers", []):
            if len(str(ans)) >= 128 and str(ans)[:200] not in all_exclude:
                human_tech_pool.append({
                    "text": str(ans),
                    "source_tag": "hc3_medicine_tech",
                    "length": len(str(ans)),
                })

# Source 4: HC3 psychology (心理学，也属专业领域) ≥128 chars
with open(BASE / "datasets/external/HC3-Chinese/psychology.jsonl", "r") as f:
    for line in f:
        d = json.loads(line)
        for ans in d.get("human_answers", []):
            if len(str(ans)) >= 128 and str(ans)[:200] not in all_exclude:
                human_tech_pool.append({
                    "text": str(ans),
                    "source_tag": "hc3_psychology_tech",
                    "length": len(str(ans)),
                })

print(f"  Total human tech pool: {len(human_tech_pool)}")

# 采样 500，按来源分层
source_counts = Counter(d["source_tag"] for d in human_tech_pool)
print(f"  Pool distribution: {dict(source_counts)}")

# 分层采样目标
tech_targets = {
    "hc3_baike_tech": 200,      # 百科知识 → 最接近 education
    "csl_academic": min(136, csl_count),  # 学术摘要 → 全取
    "hc3_medicine_tech": 80,    # 医学
    "hc3_psychology_tech": 80,  # 心理学
}

# 剩余从 baike 补
remaining = 500 - sum(min(v, source_counts.get(k, 0)) for k, v in tech_targets.items())

human_tech_samples = []
by_source = {}
for d in human_tech_pool:
    by_source.setdefault(d["source_tag"], []).append(d)

for source_tag, target_n in tech_targets.items():
    available = by_source.get(source_tag, [])
    n = min(target_n, len(available))
    if n > 0:
        selected_indices = np.random.choice(len(available), n, replace=False)
        for idx in selected_indices:
            d = available[idx]
            human_tech_samples.append({
                "text": d["text"],
                "label": 0,
                "source": f"v10_{d['source_tag']}",
                "category": "Human",
                "type": "technical",
            })
    print(f"  Sampled {source_tag}: {n}")

# 如果不够 500，从 baike 补充
if len(human_tech_samples) < 500:
    used_texts = set(d["text"][:200] for d in human_tech_samples)
    remaining_baike = [d for d in by_source.get("hc3_baike_tech", []) if d["text"][:200] not in used_texts]
    need = 500 - len(human_tech_samples)
    if remaining_baike:
        extra_n = min(need, len(remaining_baike))
        selected = np.random.choice(len(remaining_baike), extra_n, replace=False)
        for idx in selected:
            d = remaining_baike[idx]
            human_tech_samples.append({
                "text": d["text"],
                "label": 0,
                "source": "v10_hc3_baike_tech_extra",
                "category": "Human",
                "type": "technical",
            })
        print(f"  Extra baike: {extra_n}")

print(f"Human Technical 总计: {len(human_tech_samples)}")


# ════════════════════════════════════════════════════════════
# Part 3: +300 Diverse Short Human (<128 chars, multi-source)
# ════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("Part 3: Diverse Short Human (目标 300)")
print("─" * 60)

short_human_pool = []

# Collect from ALL HC3 categories, <128 chars, non-overlapping
hc3_categories = ["baike", "finance", "law", "medicine", "nlpcc_dbqa", "open_qa", "psychology"]
for cat in hc3_categories:
    count = 0
    with open(BASE / f"datasets/external/HC3-Chinese/{cat}.jsonl", "r") as f:
        for line in f:
            d = json.loads(line)
            for ans in d.get("human_answers", []):
                text = str(ans).strip()
                if 20 < len(text) < 128 and text[:200] not in all_exclude:
                    short_human_pool.append({
                        "text": text,
                        "source_tag": f"hc3_{cat}_short",
                        "length": len(text),
                    })
                    count += 1
    print(f"  HC3 {cat} short: {count}")

# LCSTS news summaries (real human, diverse topics)
lcsts_count = 0
with open(BASE / "datasets/external/LCSTS/train.json", "r") as f:
    for line in f:
        d = json.loads(line)
        # Use the 'input' field (source text, usually longer) or 'output' (summary, short)
        for field in ["output"]:  # summaries are typically short
            text = str(d.get(field, "")).strip()
            if 20 < len(text) < 128 and text[:200] not in all_exclude:
                short_human_pool.append({
                    "text": text,
                    "source_tag": "lcsts_summary",
                    "length": len(text),
                })
                lcsts_count += 1
                break

print(f"  LCSTS summaries: {lcsts_count}")
print(f"  Total short pool: {len(short_human_pool)}")

# 多源分层采样 300 条
short_by_source = {}
for d in short_human_pool:
    short_by_source.setdefault(d["source_tag"], []).append(d)

# 均匀分配，每个来源最多 50 条
short_targets = {}
sources_list = list(short_by_source.keys())
per_source = max(1, 300 // len(sources_list))

for src in sources_list:
    short_targets[src] = min(per_source, len(short_by_source[src]))

# 调整到 300
total_short = sum(short_targets.values())
if total_short < 300:
    # 从大池子补
    for src in sorted(short_by_source, key=lambda s: len(short_by_source[s]), reverse=True):
        if total_short >= 300:
            break
        extra = min(300 - total_short, len(short_by_source[src]) - short_targets.get(src, 0))
        if extra > 0:
            short_targets[src] = short_targets.get(src, 0) + extra
            total_short += extra

short_human_samples = []
for source_tag, target_n in short_targets.items():
    available = short_by_source[source_tag]
    n = min(target_n, len(available))
    if n > 0:
        selected = np.random.choice(len(available), n, replace=False)
        for idx in selected:
            d = available[idx]
            short_human_samples.append({
                "text": d["text"],
                "label": 0,
                "source": f"v10_{d['source_tag']}",
                "category": "Human",
                "type": "short_diverse",
            })
    print(f"  Sampled {source_tag}: {n}")

print(f"Short Human 总计: {len(short_human_samples)}")


# ════════════════════════════════════════════════════════════
# Merge & Save
# ════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("合并数据")
print("=" * 60)

# 合并所有新样本
new_samples = education_ai_samples + human_tech_samples + short_human_samples
new_df = pd.DataFrame(new_samples)

# 确保 schema 匹配 (train has: text, label, source, category, type)
# 检查 train 的列
print(f"训练集列: {list(train.columns)}")
print(f"新数据列: {list(new_df.columns)}")

# 对齐列
for col in train.columns:
    if col not in new_df.columns:
        new_df[col] = ""
new_df = new_df[train.columns]

# 合并
train_v10 = pd.concat([train, new_df], ignore_index=True)

# 最终去重 (prefix[:200])
before_dedup = len(train_v10)
train_v10 = train_v10.drop_duplicates(subset=["text"], keep="first")
print(f"去重: {before_dedup} → {len(train_v10)} (移除 {before_dedup - len(train_v10)})")

# 统计
print(f"\n最终训练集: {len(train_v10)} 样本")
print(f"  AI: {(train_v10['label']==1).sum()}")
print(f"  Human: {(train_v10['label']==0).sum()}")
print(f"  新增 AI: {len(education_ai_samples)}")
print(f"  新增 Human: {len(human_tech_samples) + len(short_human_samples)}")

# 新增来源分布
print(f"\n新增样本来源:")
new_sources = Counter(d.get("source", "unknown") for d in new_samples)
for src, cnt in new_sources.most_common():
    print(f"  {src}: {cnt}")

# 保存
train_v10.to_csv(OUTPUT_PATH, index=False)
print(f"\n已保存到: {OUTPUT_PATH}")
print(f"文件大小: {OUTPUT_PATH.stat().st_size / 1024 / 1024:.1f} MB")

# 验证：与评估集零重叠
v10_prefixes = set(str(t)[:200] for t in train_v10["text"].values)
overlap_with_eval = len(v10_prefixes & eval_prefixes_200)
print(f"\n与评估集重叠: {overlap_with_eval} 条")
if overlap_with_eval > 0:
    print("⚠️ 警告：存在评估集重叠！")
else:
    print("✓ 与评估集零重叠")

# 长度分布
print(f"\n新增样本长度分布:")
new_lens = new_df["text"].str.len()
for bucket, (lo, hi) in {"0-64": (0, 64), "64-128": (64, 128), "128-256": (128, 256), "256-512": (256, 512), "512+": (512, 99999)}.items():
    cnt = ((new_lens >= lo) & (new_lens < hi)).sum()
    print(f"  {bucket}: {cnt}")
