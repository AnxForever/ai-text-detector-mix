#!/usr/bin/env python3
"""
V10 对比评估脚本 - V6/V7/V8/V9/V10 五代模型在三个评估集上的全面对比

评估集:
  1. core_v1_test_clean (660 条)
  2. independent_data (910 条, 含 150 条真实 LLM 输出)
  3. merged_v2_val_clean (1,185 条)

重点关注:
  - 三集平均准确率
  - independent_data 中各 AI 模型源检出率 (GPT-5, DeepSeek-v3.2 等)
  - 长度分段准确率
  - V10 新增数据效果验证 (education AI FN, short Human FP)
  - Temperature Scaling 校准

用法:
    python scripts/evaluation/eval_v10_compare.py
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scipy.optimize import minimize_scalar

warnings.filterwarnings('ignore')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = ROOT / "datasets" / "eval" / "fair_test"

# ─── 模型列表 ───
MODELS_TO_EVAL = [
    ("bert_v6_merged", ROOT / "models" / "bert_v6_merged"),
    ("bert_v7_improved", ROOT / "models" / "bert_v7_improved"),
    ("bert_v8_calibrated", ROOT / "models" / "bert_v8_calibrated"),
    ("bert_v9_p0_supplement", ROOT / "models" / "bert_v9_p0_supplement"),
    ("bert_v10_augmented", ROOT / "models" / "bert_v10_augmented"),
]

# ─── 评估集 ───
EVAL_SETS = [
    ("core_v1_test_clean", EVAL_DIR / "core_v1_test_clean.csv"),
    ("independent_data", EVAL_DIR / "independent_data.csv"),
    ("merged_v2_val_clean", EVAL_DIR / "merged_v2_val_clean.csv"),
]


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
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
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
        }


def predict_batch(model, dataloader, device):
    """返回 (logits_all, labels_all)"""
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label']
            out = model(input_ids=ids, attention_mask=mask)
            all_logits.append(out.logits.cpu())
            all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


def find_optimal_temperature(logits, labels, min_t=0.1, max_t=5.0):
    """找到最优 Temperature Scaling 参数"""
    import torch.nn.functional as F

    def nll(t):
        scaled = logits / t
        log_probs = F.log_softmax(scaled, dim=-1)
        return -log_probs[range(len(labels)), labels].mean().item()

    result = minimize_scalar(nll, bounds=(min_t, max_t), method='bounded')
    return result.x


def compute_ece(probs, labels, n_bins=10):
    """Expected Calibration Error"""
    confidences = probs.max(dim=1).values.numpy()
    predictions = probs.argmax(dim=1).numpy()
    labels_np = labels.numpy()

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if in_bin.sum() == 0:
            continue
        avg_conf = confidences[in_bin].mean()
        avg_acc = (predictions[in_bin] == labels_np[in_bin]).mean()
        ece += in_bin.sum() / len(labels_np) * abs(avg_conf - avg_acc)
    return ece


def high_confidence_errors(probs, labels, threshold=0.8):
    """高置信错误数"""
    predictions = probs.argmax(dim=1).numpy()
    confidences = probs.max(dim=1).values.numpy()
    labels_np = labels.numpy()
    wrong = predictions != labels_np
    high_conf = confidences >= threshold
    return int((wrong & high_conf).sum())


def eval_one_model(model_name, model_path, device):
    """评估一个模型在所有评估集上的表现"""
    if not model_path.exists():
        print(f"  {model_name}: 模型不存在，跳过")
        return None

    print(f"\n{'━' * 70}")
    print(f"  {model_name}")
    print(f"{'━' * 70}")

    tokenizer = BertTokenizer.from_pretrained(str(model_path))
    model = BertForSequenceClassification.from_pretrained(
        str(model_path), num_labels=2
    ).to(device)
    model.eval()

    results = {"model": model_name}

    for eval_name, eval_path in EVAL_SETS:
        if not eval_path.exists():
            print(f"  {eval_name}: 文件不存在")
            continue

        df = pd.read_csv(eval_path)
        df = df.dropna(subset=['text', 'label'])
        texts = df['text'].astype(str).tolist()
        labels_list = df['label'].astype(int).tolist()

        ds = TextDataset(texts, labels_list, tokenizer, max_len=256)
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

        t0 = time.time()
        logits, labels = predict_batch(model, loader, device)
        elapsed = time.time() - t0

        preds = logits.argmax(dim=1).numpy()
        labels_np = labels.numpy()

        acc = accuracy_score(labels_np, preds)
        p, r, f1, _ = precision_recall_fscore_support(labels_np, preds, average='binary')

        results[eval_name] = {
            "accuracy": round(acc * 100, 2),
            "precision": round(p * 100, 2),
            "recall": round(r * 100, 2),
            "f1": round(f1 * 100, 2),
            "samples": len(df),
            "time_sec": round(elapsed, 1),
        }

        print(f"\n  [{eval_name}] ({len(df)} 样本, {elapsed:.1f}s)")
        print(f"    Acc: {acc*100:.2f}%  P: {p*100:.2f}%  R: {r*100:.2f}%  F1: {f1*100:.2f}%")

        # ─── independent_data 专项分析 ───
        if eval_name == "independent_data" and 'source' in df.columns:
            # 1. 按来源分组
            source_results = {}
            for src in sorted(df['source'].dropna().unique()):
                mask = df['source'] == src
                if mask.sum() < 3:
                    continue
                src_labels = labels_np[mask.values]
                src_preds = preds[mask.values]
                src_acc = (src_preds == src_labels).mean()
                source_results[src] = {
                    "accuracy": round(src_acc * 100, 2),
                    "count": int(mask.sum()),
                }

            results[f"{eval_name}_by_source"] = source_results

            # 2. 重点: 真实 AI 来源检出率
            print(f"\n    真实 AI 来源检出率:")
            ai_sources = df[df['label'] == 1]['source'].dropna().unique()
            for src in sorted(ai_sources):
                mask = (df['source'] == src) & (df['label'] == 1)
                if mask.sum() == 0:
                    continue
                src_preds_ai = preds[mask.values]
                src_labels_ai = labels_np[mask.values]
                detected = (src_preds_ai == 1).sum()
                total = len(src_preds_ai)
                rate = detected / total * 100
                mark = "●" if rate >= 95 else ("◐" if rate >= 80 else "○")
                print(f"      {mark} {src}: {detected}/{total} ({rate:.1f}%)")

            # 3. 长度分段
            char_lens = df['text'].str.len().values
            print(f"\n    长度分段准确率:")
            for name, lo, hi in [("0-64", 0, 64), ("64-128", 64, 128),
                                  ("128-256", 128, 256), ("256-512", 256, 512),
                                  ("512+", 512, 99999)]:
                m = (char_lens >= lo) & (char_lens < hi)
                if m.sum() > 0:
                    seg_acc = (preds[m] == labels_np[m]).mean() * 100
                    print(f"      {name}: {seg_acc:.2f}% ({m.sum()} 样本)")

            # 4. 错误样本分析
            errors = df.iloc[preds != labels_np]
            results[f"{eval_name}_errors"] = len(errors)
            if len(errors) > 0:
                fn_mask = (preds == 0) & (labels_np == 1)  # AI→Human
                fp_mask = (preds == 1) & (labels_np == 0)  # Human→AI
                print(f"\n    错误分析: FN(AI→Human)={fn_mask.sum()}, FP(Human→AI)={fp_mask.sum()}")

                if fn_mask.sum() > 0:
                    fn_df = df.iloc[fn_mask]
                    fn_sources = fn_df['source'].value_counts()
                    print(f"    FN 来源: {dict(fn_sources.head(10))}")

            # 5. Temperature Scaling 校准
            opt_t = find_optimal_temperature(logits, labels)
            raw_probs = torch.softmax(logits, dim=-1)
            scaled_probs = torch.softmax(logits / opt_t, dim=-1)
            ece_before = compute_ece(raw_probs, labels)
            ece_after = compute_ece(scaled_probs, labels)
            hce_before = high_confidence_errors(raw_probs, labels)
            hce_after = high_confidence_errors(scaled_probs, labels)

            results[f"{eval_name}_calibration"] = {
                "optimal_T": round(opt_t, 4),
                "ECE_before": round(ece_before, 4),
                "ECE_after": round(ece_after, 4),
                "high_conf_errors_before": hce_before,
                "high_conf_errors_after": hce_after,
            }

            print(f"\n    Temperature Scaling:")
            print(f"      最优 T: {opt_t:.4f}")
            print(f"      ECE: {ece_before:.4f} → {ece_after:.4f}")
            print(f"      高置信错误: {hce_before} → {hce_after}")

    # 三集平均
    accs = []
    for eval_name, _ in EVAL_SETS:
        if eval_name in results:
            accs.append(results[eval_name]["accuracy"])
    if accs:
        results["three_set_avg"] = round(sum(accs) / len(accs), 2)
        print(f"\n  三集平均: {results['three_set_avg']:.2f}%")

    del model
    torch.cuda.empty_cache()
    return results


def main():
    print("=" * 70)
    print("  V10 五代模型对比评估")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 评估集信息
    print(f"\n评估集:")
    for name, path in EVAL_SETS:
        if path.exists():
            df = pd.read_csv(path)
            print(f"  {name}: {len(df)} 样本 (AI: {(df['label']==1).sum()}, Human: {(df['label']==0).sum()})")
        else:
            print(f"  {name}: 不存在!")

    # 逐模型评估
    all_results = {}
    for model_name, model_path in MODELS_TO_EVAL:
        result = eval_one_model(model_name, model_path, device)
        if result:
            all_results[model_name] = result

    # ═══ 汇总对比表 ═══
    print("\n\n" + "=" * 90)
    print("  五代模型对比汇总")
    print("=" * 90)

    header = f"{'模型':<28}"
    for name, _ in EVAL_SETS:
        short = name.replace("_", " ")[:18]
        header += f" | {short:>18}"
    header += f" | {'三集平均':>10}"
    print(header)
    print("─" * len(header))

    for model_name in [m[0] for m in MODELS_TO_EVAL]:
        if model_name not in all_results:
            continue
        r = all_results[model_name]
        row = f"{model_name:<28}"
        for eval_name, _ in EVAL_SETS:
            if eval_name in r:
                row += f" | {r[eval_name]['accuracy']:>17.2f}%"
            else:
                row += f" |       {'N/A':>11}"
        avg = r.get("three_set_avg", 0)
        row += f" | {avg:>9.2f}%"
        print(row)

    # ═══ independent_data AI 来源对比 ═══
    print("\n\n" + "=" * 90)
    print("  independent_data AI 来源检出率对比")
    print("=" * 90)

    # 收集所有 AI 来源
    ai_sources_all = set()
    for model_name, r in all_results.items():
        by_source = r.get("independent_data_by_source", {})
        for src in by_source:
            if "real_ai" in src or "m4" in src.lower():
                ai_sources_all.add(src)

    if ai_sources_all:
        header = f"{'AI 来源':<35}"
        for model_name in [m[0] for m in MODELS_TO_EVAL]:
            if model_name in all_results:
                short = model_name.replace("bert_", "")[:10]
                header += f" | {short:>10}"
        print(header)
        print("─" * len(header))

        for src in sorted(ai_sources_all):
            row = f"{src:<35}"
            for model_name in [m[0] for m in MODELS_TO_EVAL]:
                if model_name not in all_results:
                    continue
                by_source = all_results[model_name].get("independent_data_by_source", {})
                if src in by_source:
                    row += f" | {by_source[src]['accuracy']:>9.1f}%"
                else:
                    row += f" |       N/A"
            print(row)

    # ═══ Temperature 校准对比 ═══
    print("\n\n" + "=" * 70)
    print("  Temperature Scaling 校准对比 (independent_data)")
    print("=" * 70)

    header = f"{'模型':<28} | {'T':>6} | {'ECE前':>8} | {'ECE后':>8} | {'HCE前':>6} | {'HCE后':>6}"
    print(header)
    print("─" * len(header))

    for model_name in [m[0] for m in MODELS_TO_EVAL]:
        if model_name not in all_results:
            continue
        cal = all_results[model_name].get("independent_data_calibration", {})
        if cal:
            print(f"{model_name:<28} | {cal['optimal_T']:>6.4f} | {cal['ECE_before']:>8.4f} | {cal['ECE_after']:>8.4f} | {cal['high_conf_errors_before']:>6} | {cal['high_conf_errors_after']:>6}")

    # 保存结果
    output_path = ROOT / "models" / "bert_v10_augmented" / "eval_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 序列化
    save_data = {}
    for k, v in all_results.items():
        save_data[k] = {}
        for kk, vv in v.items():
            if isinstance(vv, (dict, list, str, int, float, bool)):
                save_data[k][kk] = vv
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {output_path}")


if __name__ == '__main__':
    main()
