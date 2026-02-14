#!/usr/bin/env python3
"""
公平模型对比评估 - 所有模型在相同条件下的真实表现

评估策略：
1. 构建"干净测试集"：排除所有模型训练数据中出现过的样本
2. 多维度评估：标准指标 + 分类别 + 问题样本
3. 所有模型用相同测试集、相同阈值、相同指标
"""

import os
import sys
import json
import time
import hashlib
import warnings
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent

# ─── 模型配置 ───────────────────────────────────────────────────
MODELS = {
    "bert_v2_with_sep": {
        "path": "models/bert_v2_with_sep",
        "trained_on": ["datasets/active/core_v1/train.csv"],
        "desc": "基准模型 (SEP标记)",
    },
    "bert_v2_balanced": {
        "path": "models/bert_v2_balanced",
        "trained_on": ["datasets/active/core_v1/train.csv"],
        "desc": "平衡采样版",
    },
    "bert_v3_core_v2": {
        "path": "models/bert_v3_core_v2",
        "trained_on": ["datasets/active/core_v2/train.csv"],
        "desc": "扩展场景版",
    },
    "bert_v3_fresh": {
        "path": "models/bert_v3_fresh",
        "trained_on": ["datasets/active/core_v2/train.csv"],
        "desc": "从零训练对照",
    },
    "bert_v4_core_v3": {
        "path": "models/bert_v4_core_v3",
        "trained_on": ["datasets/active/core_v3/train.csv"],
        "desc": "core_v3版",
    },
    "bert_v4_defense_focused": {
        "path": "models/bert_v4_defense_focused",
        "trained_on": [],  # 小数据集，难以追踪
        "desc": "防御增强版",
    },
    "bert_v5_paired": {
        "path": "models/bert_v5_paired",
        "trained_on": ["datasets/paired/paired_v3_all_train.csv"],
        "desc": "配对数据版",
    },
    "bert_v6_merged": {
        "path": "models/bert_v6_merged",
        "trained_on": ["datasets/merged_v1/train.csv"],
        "desc": "合并版v1",
    },
    "bert_v7_improved": {
        "path": "models/bert_v7_improved",
        "trained_on": ["datasets/merged_v2/train.csv"],
        "desc": "改进版(length penalty)",
    },
}

# ─── 问题样本（所有模型都没训练过）───────────────────────────────
PROBLEM_SAMPLES = [
    # Human 样本 (label=0) - 正式文本，容易被误判为AI
    {"text": "温馨提示：明天上午9点将进行消防演练，请各位同事提前做好准备，保持通道畅通。",
     "label": 0, "category": "正式通知"},
    {"text": "今天天气真好，中午和同事一起去公园散步了，心情舒畅。",
     "label": 0, "category": "日常分享"},
    {"text": "这家店的牛肉面真的很不错，汤底浓郁，面条筋道，强烈推荐大家去尝尝。",
     "label": 0, "category": "口语推荐"},
    {"text": "今天终于把报告写完了，感觉整个人都轻松了。",
     "label": 0, "category": "工作日常"},
    {"text": "本文研究了深度学习在自然语言处理中的应用，通过实验验证了模型的有效性。",
     "label": 0, "category": "学术摘要"},
    {"text": "实验结果表明，改进后的算法在准确率上提升了5%，验证了本文方法的可行性。",
     "label": 0, "category": "实验结论"},
    {"text": "昨日，我市召开经济工作会议，市长强调要加快产业升级，推动高质量发展。",
     "label": 0, "category": "地方新闻"},
    {"text": "关于2024年期末考试安排的通知：考试时间为1月15日至1月20日，请同学们认真复习。",
     "label": 0, "category": "考试通知"},
    {"text": "尊敬的老师：您好！我是计算机系大三学生张明，想就毕业论文选题向您请教。",
     "label": 0, "category": "学生邮件"},
    {"text": "梯度爆炸（gradient explosion）是指在深度神经网络训练过程中，反向传播时梯度值呈指数级增长的现象。",
     "label": 1, "category": "AI技术文档"},
    # AI 样本 (label=1) - 口语化AI文本，容易被误判为Human
    {"text": "额...这个问题嘛...让我想想啊，好像是这样的...不对不对，应该是那样...",
     "label": 1, "category": "犹豫式AI"},
    {"text": "老铁们这个东西绝绝子！yyds！真的太绝了，建议大家冲冲冲！",
     "label": 1, "category": "网络语AI"},
    {"text": "深度学习是机器学习的一个分支，它使用多层神经网络来学习数据的层次化表示。通过自动提取特征，深度学习在图像识别、自然语言处理等领域取得了显著成果。",
     "label": 1, "category": "AI百科解释"},
    {"text": "人工智能（Artificial Intelligence，简称AI）是计算机科学的一个重要分支。它致力于研究和开发能够模拟、延伸和扩展人类智能的理论、方法和技术。",
     "label": 1, "category": "AI定义文本"},
]


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def text_hash(text):
    """生成文本指纹（前200字符的hash）"""
    return hashlib.md5(str(text)[:200].encode('utf-8')).hexdigest()


def collect_training_fingerprints():
    """收集所有模型训练数据的文本指纹"""
    all_train_hashes = set()
    per_model_hashes = {}

    for model_name, config in MODELS.items():
        model_hashes = set()
        for train_path in config["trained_on"]:
            full_path = PROJECT_ROOT / train_path
            if not full_path.exists():
                continue
            try:
                if train_path.endswith('.csv'):
                    df = pd.read_csv(full_path, encoding='utf-8-sig', usecols=['text'])
                elif train_path.endswith('.jsonl'):
                    df = pd.read_json(full_path, lines=True, columns=['text'])
                else:
                    continue
                for t in df['text']:
                    h = text_hash(t)
                    model_hashes.add(h)
                    all_train_hashes.add(h)
            except Exception as e:
                print(f"  警告: 无法加载 {train_path}: {e}")
        per_model_hashes[model_name] = model_hashes

    return all_train_hashes, per_model_hashes


def load_test_data():
    """加载并构建干净测试集"""
    print("=" * 80)
    print("第一步：收集训练数据指纹")
    print("=" * 80)

    all_train_hashes, per_model_hashes = collect_training_fingerprints()
    print(f"所有模型训练数据合计 {len(all_train_hashes)} 个唯一文本指纹")
    for name, hashes in per_model_hashes.items():
        print(f"  {name}: {len(hashes)} 条")

    print("\n" + "=" * 80)
    print("第二步：构建干净测试集")
    print("=" * 80)

    test_sets = {}

    # ── 测试集1: core_v1/test (标准测试集) ──
    path = PROJECT_ROOT / "datasets/active/core_v1/test.csv"
    if path.exists():
        df = pd.read_csv(path, encoding='utf-8-sig')
        df['_hash'] = df['text'].apply(text_hash)
        total = len(df)
        leaked = df['_hash'].isin(all_train_hashes).sum()
        clean = df[~df['_hash'].isin(all_train_hashes)].copy()
        print(f"\ncore_v1/test: {total} 条, 泄露 {leaked} 条 ({leaked/total*100:.1f}%), 干净 {len(clean)} 条")
        print(f"  label分布: {clean['label'].value_counts().to_dict()}")
        if len(clean) > 0:
            test_sets['core_v1_clean'] = clean

        # 也保留完整版用于对比
        test_sets['core_v1_full'] = df

    # ── 测试集2: merged_v2/val (v7的验证集) ──
    path = PROJECT_ROOT / "datasets/merged_v2/val.csv"
    if path.exists():
        df = pd.read_csv(path, encoding='utf-8-sig')
        df['_hash'] = df['text'].apply(text_hash)
        total = len(df)
        leaked_count = 0
        for name, hashes in per_model_hashes.items():
            if name == 'bert_v7_improved':
                continue  # v7用这个做val，不算泄露
            overlap = df['_hash'].isin(hashes).sum()
            if overlap > 0:
                leaked_count = max(leaked_count, overlap)
        # 排除所有训练数据（包括v7的，因为它用这个做了模型选择）
        clean = df[~df['_hash'].isin(all_train_hashes)].copy()
        print(f"\nmerged_v2/val: {total} 条, 与训练数据重叠后剩余干净 {len(clean)} 条")
        if len(clean) > 0:
            test_sets['merged_v2_val_clean'] = clean

    # ── 测试集3: 纯Human独立数据 (从未用于训练) ──
    human_dfs = []

    # human_consolidated (手动收集的正式人类文本)
    p = PROJECT_ROOT / "datasets/human_consolidated/all_human_consolidated.jsonl"
    if p.exists():
        hc = pd.read_json(p, lines=True)
        hc['_hash'] = hc['text'].apply(text_hash)
        clean = hc[~hc['_hash'].isin(all_train_hashes)]
        print(f"\nhuman_consolidated: {len(hc)} 条, 干净 {len(clean)} 条")
        human_dfs.append(clean)

    # defense_patch
    p = PROJECT_ROOT / "datasets/defense_patch/defense_patch_v2.jsonl"
    if p.exists():
        dp = pd.read_json(p, lines=True)
        dp['_hash'] = dp['text'].apply(text_hash)
        clean = dp[~dp['_hash'].isin(all_train_hashes)]
        print(f"defense_patch_v2: {len(dp)} 条, 干净 {len(clean)} 条")
        human_dfs.append(clean)

    # external human
    p = PROJECT_ROOT / "datasets/external/processed/human_opensource.jsonl"
    if p.exists():
        eh = pd.read_json(p, lines=True)
        eh['_hash'] = eh['text'].apply(text_hash)
        clean = eh[~eh['_hash'].isin(all_train_hashes)]
        print(f"external/human_opensource: {len(eh)} 条, 干净 {len(clean)} 条")
        human_dfs.append(clean)

    # toutiao tech
    p = PROJECT_ROOT / "datasets/human_consolidated/toutiao_tech_human.jsonl"
    if p.exists():
        tt = pd.read_json(p, lines=True)
        tt['_hash'] = tt['text'].apply(text_hash)
        clean = tt[~tt['_hash'].isin(all_train_hashes)]
        print(f"toutiao_tech_human: {len(tt)} 条, 干净 {len(clean)} 条")
        human_dfs.append(clean)

    if human_dfs:
        combined = pd.concat(human_dfs, ignore_index=True)
        # 去重
        combined = combined.drop_duplicates(subset=['_hash'])
        print(f"\n独立Human数据合计: {len(combined)} 条 (去重后)")
        test_sets['independent_human'] = combined

    return test_sets


def evaluate_batch(model, dataloader, device):
    """批量评估"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label']

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # AI概率

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def compute_metrics(preds, labels):
    """计算全部指标"""
    acc = accuracy_score(labels, preds)
    p_human = precision_score(labels, preds, pos_label=0, zero_division=0)
    r_human = recall_score(labels, preds, pos_label=0, zero_division=0)
    f1_human = f1_score(labels, preds, pos_label=0, zero_division=0)
    p_ai = precision_score(labels, preds, pos_label=1, zero_division=0)
    r_ai = recall_score(labels, preds, pos_label=1, zero_division=0)
    f1_ai = f1_score(labels, preds, pos_label=1, zero_division=0)

    return {
        'accuracy': acc,
        'human_precision': p_human,
        'human_recall': r_human,
        'human_f1': f1_human,
        'ai_precision': p_ai,
        'ai_recall': r_ai,
        'ai_f1': f1_ai,
    }


def evaluate_problem_samples(tokenizer, model, device):
    """评估问题样本"""
    results = []
    model.eval()

    for sample in PROBLEM_SAMPLES:
        inputs = tokenizer(
            sample['text'], return_tensors='pt',
            truncation=True, max_length=512, padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            ai_prob = probs[0][1].item()

        pred = 1 if ai_prob >= 0.5 else 0
        correct = pred == sample['label']
        results.append({
            'category': sample['category'],
            'label': sample['label'],
            'ai_prob': ai_prob,
            'pred': pred,
            'correct': correct,
        })

    return results


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    os.chdir(PROJECT_ROOT)

    # 加载测试数据
    test_sets = load_test_data()

    print("\n" + "=" * 80)
    print("第三步：逐模型评估")
    print("=" * 80)

    all_results = {}
    batch_size = 32

    for model_name, config in MODELS.items():
        model_path = PROJECT_ROOT / config["path"]
        if not model_path.exists():
            print(f"\n⏭️  {model_name}: 模型不存在，跳过")
            continue

        print(f"\n{'─' * 80}")
        print(f"📊 {model_name} ({config['desc']})")
        print(f"{'─' * 80}")

        try:
            tokenizer = BertTokenizer.from_pretrained(str(model_path))
            model = BertForSequenceClassification.from_pretrained(str(model_path)).to(device)
            model.eval()
        except Exception as e:
            print(f"   ❌ 加载失败: {e}")
            continue

        model_results = {'desc': config['desc']}

        # ── 各测试集评估 ──
        for test_name, test_df in test_sets.items():
            if 'text' not in test_df.columns or 'label' not in test_df.columns:
                continue

            texts = test_df['text'].tolist()
            labels = test_df['label'].tolist()

            dataset = TextDataset(texts, labels, tokenizer, max_len=512)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                num_workers=0, pin_memory=True)

            t0 = time.time()
            preds, true_labels, probs = evaluate_batch(model, loader, device)
            elapsed = time.time() - t0

            metrics = compute_metrics(preds, true_labels)
            model_results[test_name] = metrics
            model_results[f'{test_name}_time'] = elapsed

            label_dist = pd.Series(true_labels).value_counts().to_dict()
            print(f"\n  [{test_name}] ({len(texts)}条, H:{label_dist.get(0,0)} A:{label_dist.get(1,0)}) {elapsed:.1f}s")
            print(f"    准确率: {metrics['accuracy']*100:.2f}%")
            print(f"    Human: P={metrics['human_precision']*100:.1f}% R={metrics['human_recall']*100:.1f}% F1={metrics['human_f1']*100:.1f}%")
            print(f"    AI:    P={metrics['ai_precision']*100:.1f}% R={metrics['ai_recall']*100:.1f}% F1={metrics['ai_f1']*100:.1f}%")

        # ── 问题样本评估 ──
        prob_results = evaluate_problem_samples(tokenizer, model, device)
        correct_count = sum(1 for r in prob_results if r['correct'])
        total = len(prob_results)
        model_results['problem_samples'] = prob_results
        model_results['problem_acc'] = correct_count / total if total > 0 else 0

        human_samples = [r for r in prob_results if r['label'] == 0]
        ai_samples = [r for r in prob_results if r['label'] == 1]
        human_correct = sum(1 for r in human_samples if r['correct'])
        ai_correct = sum(1 for r in ai_samples if r['correct'])

        print(f"\n  [问题样本] ({total}条)")
        print(f"    总正确率: {correct_count}/{total} ({correct_count/total*100:.1f}%)")
        print(f"    Human样本: {human_correct}/{len(human_samples)}")
        print(f"    AI样本: {ai_correct}/{len(ai_samples)}")

        for r in prob_results:
            mark = "✅" if r['correct'] else "❌"
            label_str = "Human" if r['label'] == 0 else "AI"
            print(f"      {mark} [{label_str}] {r['category']}: AI概率={r['ai_prob']*100:.1f}%")

        # 释放显存
        del model
        torch.cuda.empty_cache() if device.type == 'cuda' else None

        all_results[model_name] = model_results

    # ── 汇总表格 ──
    print("\n\n" + "=" * 120)
    print("📋 模型对比汇总")
    print("=" * 120)

    # 找到所有测试集名称
    test_set_names = [k for k in test_sets.keys() if k != 'core_v1_full']

    # 打印表头
    header = f"{'模型':<25}"
    for ts_name in test_set_names:
        header += f" | {ts_name:<20}"
    header += f" | {'问题样本':<12}"
    print(header)
    print("─" * len(header))

    ranked = []

    for model_name, results in all_results.items():
        row = f"{model_name:<25}"
        scores = []
        for ts_name in test_set_names:
            if ts_name in results:
                acc = results[ts_name]['accuracy']
                row += f" | {acc*100:>6.2f}%             "
                scores.append(acc)
            else:
                row += f" | {'N/A':<20}"
        prob_acc = results.get('problem_acc', 0)
        row += f" | {prob_acc*100:>5.1f}%"
        print(row)

        avg_score = np.mean(scores) if scores else 0
        ranked.append((model_name, avg_score, prob_acc, results))

    # ── 综合排名 ──
    print("\n" + "=" * 120)
    print("🏆 综合排名（按干净测试集平均准确率）")
    print("=" * 120)

    ranked.sort(key=lambda x: x[1], reverse=True)
    for i, (name, avg, prob_acc, results) in enumerate(ranked, 1):
        detail_parts = []
        for ts_name in test_set_names:
            if ts_name in results:
                detail_parts.append(f"{ts_name}={results[ts_name]['accuracy']*100:.2f}%")
        detail = ", ".join(detail_parts)
        print(f"  #{i} {name:<25} 平均={avg*100:.2f}%  问题样本={prob_acc*100:.1f}%  ({detail})")

    # ── 保存详细结果 ──
    output_path = PROJECT_ROOT / "scripts/evaluation/fair_comparison_results.json"
    save_data = {}
    for model_name, results in all_results.items():
        save_entry = {}
        for key, val in results.items():
            if key == 'problem_samples':
                save_entry[key] = val
            elif isinstance(val, dict):
                save_entry[key] = {k: float(v) for k, v in val.items()}
            else:
                save_entry[key] = val
        save_data[model_name] = save_entry

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存: {output_path}")


if __name__ == '__main__':
    main()
