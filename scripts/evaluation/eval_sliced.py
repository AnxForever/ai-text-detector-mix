#!/usr/bin/env python3
"""
Sliced Evaluation - 多维度切片评估 BERT 模型

对每个模型在 fair_test 数据集上进行如下切片分析：
1. Per-dataset metrics (Accuracy, Precision, Recall, F1)
2. Length-bucket metrics (按字符长度分桶)
3. Source-based metrics (按 source 列分组)
4. Confidence calibration (置信度校准)
5. Error analysis (误分类样本分析)

用法:
    python scripts/evaluation/eval_sliced.py
    python scripts/evaluation/eval_sliced.py --models bert_v6_merged bert_v7_improved
"""

import argparse
import json
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ─── 模型配置 ───────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "bert_v6_merged": {
        "path": PROJECT_ROOT / "models" / "bert_v6_merged",
        "max_length": 256,
        "desc": "合并版 v1 (max_len=256)",
    },
    "bert_v7_improved": {
        "path": PROJECT_ROOT / "models" / "bert_v7_improved",
        "max_length": 256,
        "desc": "改进版 v7 (length penalty, max_len=256)",
    },
    "bert_improved": {
        "path": PROJECT_ROOT / "models" / "bert_improved" / "best_model",
        "max_length": 512,
        "desc": "改进版 baseline (max_len=512)",
    },
    "bert_v8_calibrated": {
        "path": PROJECT_ROOT / "models" / "bert_v8_calibrated",
        "max_length": 256,
        "desc": "校准版 v8 (label smoothing=0.05, lr=1e-5)",
    },
    "bert_v9_p0_supplement": {
        "path": PROJECT_ROOT / "models" / "bert_v9_p0_supplement",
        "max_length": 256,
        "desc": "P0 supplement v9 (label smoothing=0.05, lr=1e-5)",
    },
}

# ─── 测试集配置 ─────────────────────────────────────────────────────
DATASET_CONFIGS = {
    "core_v1_test_clean": {
        "path": PROJECT_ROOT / "datasets" / "eval" / "fair_test" / "core_v1_test_clean.csv",
        "desc": "core_v1 测试集 (干净版)",
    },
    "independent_data": {
        "path": PROJECT_ROOT / "datasets" / "eval" / "fair_test" / "independent_data.csv",
        "desc": "独立测试数据",
    },
    "merged_v2_val_clean": {
        "path": PROJECT_ROOT / "datasets" / "eval" / "fair_test" / "merged_v2_val_clean.csv",
        "desc": "merged_v2 验证集 (干净版)",
    },
}

# ─── 长度分桶配置 ────────────────────────────────────────────────────
LENGTH_BUCKETS = [
    (0, 64, "0-64"),
    (64, 128, "64-128"),
    (128, 256, "128-256"),
    (256, 512, "256-512"),
    (512, float("inf"), "512+"),
]

BATCH_SIZE = 32


# ─── Dataset 类 ──────────────────────────────────────────────────────
class TextDataset(Dataset):
    """BERT 文本分类数据集"""

    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ─── 批量推理 ────────────────────────────────────────────────────────
def batch_inference(model, dataloader, device):
    """批量推理，返回预测标签、真实标签、各类别 softmax 概率"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []  # shape: (N, 2)

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return (
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs),
    )


# ─── 指标计算 ────────────────────────────────────────────────────────
def compute_classification_metrics(preds, labels):
    """计算 Accuracy, Precision, Recall, F1 (macro + per-class)"""
    acc = accuracy_score(labels, preds)
    precision_macro = precision_score(labels, preds, average="macro", zero_division=0)
    recall_macro = recall_score(labels, preds, average="macro", zero_division=0)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)

    precision_human = precision_score(labels, preds, pos_label=0, zero_division=0)
    recall_human = recall_score(labels, preds, pos_label=0, zero_division=0)
    f1_human = f1_score(labels, preds, pos_label=0, zero_division=0)

    precision_ai = precision_score(labels, preds, pos_label=1, zero_division=0)
    recall_ai = recall_score(labels, preds, pos_label=1, zero_division=0)
    f1_ai = f1_score(labels, preds, pos_label=1, zero_division=0)

    return {
        "accuracy": float(acc),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "human_precision": float(precision_human),
        "human_recall": float(recall_human),
        "human_f1": float(f1_human),
        "ai_precision": float(precision_ai),
        "ai_recall": float(recall_ai),
        "ai_f1": float(f1_ai),
    }


def compute_length_bucket_metrics(preds, labels, char_lengths):
    """按字符长度分桶计算 accuracy 和 count"""
    bucket_results = {}
    for low, high, name in LENGTH_BUCKETS:
        mask = (char_lengths >= low) & (char_lengths < high)
        count = int(mask.sum())
        if count == 0:
            bucket_results[name] = {"count": 0, "accuracy": None}
            continue
        bucket_preds = preds[mask]
        bucket_labels = labels[mask]
        acc = float(accuracy_score(bucket_labels, bucket_preds))
        bucket_results[name] = {"count": count, "accuracy": acc}
    return bucket_results


def compute_source_metrics(preds, labels, sources):
    """按 source 列分组计算 accuracy"""
    source_results = {}
    unique_sources = sorted(set(sources))
    for src in unique_sources:
        mask = np.array([s == src for s in sources])
        count = int(mask.sum())
        if count == 0:
            continue
        src_preds = preds[mask]
        src_labels = labels[mask]
        acc = float(accuracy_score(src_labels, src_preds))
        source_results[src] = {"count": count, "accuracy": acc}
    return source_results


def compute_confidence_calibration(preds, labels, probs):
    """置信度校准分析"""
    # 预测类别的 softmax 概率 = confidence
    pred_confidences = np.array([probs[i, preds[i]] for i in range(len(preds))])

    correct_mask = preds == labels
    wrong_mask = ~correct_mask

    avg_confidence = float(np.mean(pred_confidences))
    avg_conf_correct = (
        float(np.mean(pred_confidences[correct_mask]))
        if correct_mask.sum() > 0
        else None
    )
    avg_conf_wrong = (
        float(np.mean(pred_confidences[wrong_mask]))
        if wrong_mask.sum() > 0
        else None
    )

    # 高置信度错误：confidence > 0.9 但预测错误
    high_conf_wrong_mask = wrong_mask & (pred_confidences > 0.9)
    high_conf_wrong_count = int(high_conf_wrong_mask.sum())

    return {
        "avg_confidence": avg_confidence,
        "avg_confidence_correct": avg_conf_correct,
        "avg_confidence_wrong": avg_conf_wrong,
        "high_confidence_wrong_count": high_conf_wrong_count,
        "total_correct": int(correct_mask.sum()),
        "total_wrong": int(wrong_mask.sum()),
    }


def collect_error_samples(texts, preds, labels, probs, max_samples=10):
    """收集前 N 个误分类样本"""
    errors = []
    pred_confidences = np.array([probs[i, preds[i]] for i in range(len(preds))])

    for i in range(len(preds)):
        if preds[i] != labels[i]:
            snippet = str(texts[i])[:100]
            errors.append(
                {
                    "text_snippet": snippet,
                    "true_label": int(labels[i]),
                    "predicted_label": int(preds[i]),
                    "confidence": float(pred_confidences[i]),
                }
            )
            if len(errors) >= max_samples:
                break
    return errors


# ─── 单数据集评估 ────────────────────────────────────────────────────
def evaluate_single_dataset(
    model, tokenizer, max_length, df, dataset_name, device
):
    """对单个数据集执行全部切片评估"""
    texts = df["text"].tolist()
    labels_list = df["label"].astype(int).tolist()
    char_lengths = np.array([len(str(t)) for t in texts])

    dataset = TextDataset(texts, labels_list, tokenizer, max_length=max_length)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    t0 = time.time()
    preds, true_labels, probs = batch_inference(model, loader, device)
    elapsed = time.time() - t0

    # 1) Per-dataset metrics
    metrics = compute_classification_metrics(preds, true_labels)
    metrics["sample_count"] = len(texts)
    metrics["human_count"] = int((true_labels == 0).sum())
    metrics["ai_count"] = int((true_labels == 1).sum())
    metrics["inference_time_sec"] = round(elapsed, 2)

    # 2) Length-bucket metrics
    length_buckets = compute_length_bucket_metrics(preds, true_labels, char_lengths)

    # 3) Source-based metrics
    source_metrics = {}
    if "source" in df.columns:
        sources = df["source"].fillna("unknown").tolist()
        source_metrics = compute_source_metrics(preds, true_labels, sources)

    # 4) Confidence calibration
    calibration = compute_confidence_calibration(preds, true_labels, probs)

    # 5) Error analysis
    error_samples = collect_error_samples(texts, preds, true_labels, probs, max_samples=10)

    return {
        "metrics": metrics,
        "length_buckets": length_buckets,
        "source_metrics": source_metrics,
        "confidence_calibration": calibration,
        "error_samples": error_samples,
    }


# ─── 报告打印 ────────────────────────────────────────────────────────
LABEL_MAP = {0: "Human", 1: "AI"}


def print_dataset_report(dataset_name, result):
    """打印单个数据集的切片报告"""
    m = result["metrics"]
    print(f"\n  [{dataset_name}] ({m['sample_count']} samples, "
          f"Human: {m['human_count']}, AI: {m['ai_count']}, "
          f"time: {m['inference_time_sec']}s)")
    print(f"    Accuracy:  {m['accuracy']*100:.2f}%")
    print(f"    Precision: {m['precision_macro']*100:.2f}% (macro) "
          f"| Human={m['human_precision']*100:.1f}% AI={m['ai_precision']*100:.1f}%")
    print(f"    Recall:    {m['recall_macro']*100:.2f}% (macro) "
          f"| Human={m['human_recall']*100:.1f}% AI={m['ai_recall']*100:.1f}%")
    print(f"    F1:        {m['f1_macro']*100:.2f}% (macro) "
          f"| Human={m['human_f1']*100:.1f}% AI={m['ai_f1']*100:.1f}%")

    # Length buckets
    lb = result["length_buckets"]
    print(f"\n    Length Buckets:")
    print(f"      {'Bucket':<12} {'Count':>8} {'Accuracy':>10}")
    print(f"      {'------':<12} {'-----':>8} {'--------':>10}")
    for bucket_name in ["0-64", "64-128", "128-256", "256-512", "512+"]:
        info = lb.get(bucket_name, {})
        count = info.get("count", 0)
        acc = info.get("accuracy")
        acc_str = f"{acc*100:.2f}%" if acc is not None else "N/A"
        print(f"      {bucket_name:<12} {count:>8} {acc_str:>10}")

    # Source metrics
    sm = result["source_metrics"]
    if sm:
        print(f"\n    Source Breakdown:")
        print(f"      {'Source':<30} {'Count':>8} {'Accuracy':>10}")
        print(f"      {'------':<30} {'-----':>8} {'--------':>10}")
        for src, info in sorted(sm.items(), key=lambda x: -x[1]["count"]):
            print(f"      {src:<30} {info['count']:>8} {info['accuracy']*100:.2f}%")

    # Confidence calibration
    cal = result["confidence_calibration"]
    print(f"\n    Confidence Calibration:")
    print(f"      Avg confidence (all):     {cal['avg_confidence']*100:.2f}%")
    if cal["avg_confidence_correct"] is not None:
        print(f"      Avg confidence (correct): {cal['avg_confidence_correct']*100:.2f}%  ({cal['total_correct']} samples)")
    if cal["avg_confidence_wrong"] is not None:
        print(f"      Avg confidence (wrong):   {cal['avg_confidence_wrong']*100:.2f}%  ({cal['total_wrong']} samples)")
    print(f"      High-conf wrong (>0.9):   {cal['high_confidence_wrong_count']}")

    # Error samples
    errors = result["error_samples"]
    if errors:
        print(f"\n    Error Analysis (first {len(errors)} misclassified):")
        for i, e in enumerate(errors, 1):
            true_str = LABEL_MAP.get(e["true_label"], str(e["true_label"]))
            pred_str = LABEL_MAP.get(e["predicted_label"], str(e["predicted_label"]))
            print(f"      {i:>2}. true={true_str}, pred={pred_str}, "
                  f"conf={e['confidence']*100:.1f}%")
            print(f"          \"{e['text_snippet']}\"")
    else:
        print(f"\n    Error Analysis: No misclassified samples!")


def print_cross_model_summary(all_results):
    """打印跨模型对比汇总"""
    print("\n" + "=" * 100)
    print("CROSS-MODEL SUMMARY")
    print("=" * 100)

    dataset_names = list(DATASET_CONFIGS.keys())

    # Header
    header = f"  {'Model':<22}"
    for ds in dataset_names:
        short_name = ds[:18]
        header += f" | {short_name:>18}"
    header += f" | {'Average':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for model_name, model_data in all_results.items():
        row = f"  {model_name:<22}"
        accs = []
        for ds in dataset_names:
            ds_result = model_data.get("datasets", {}).get(ds)
            if ds_result and ds_result["metrics"]["accuracy"] is not None:
                acc = ds_result["metrics"]["accuracy"]
                accs.append(acc)
                row += f" | {acc*100:>17.2f}%"
            else:
                row += f" | {'N/A':>18}"
        avg = np.mean(accs) if accs else 0
        row += f" | {avg*100:>9.2f}%"
        print(row)

    # High-confidence wrong summary
    print(f"\n  High-Confidence Wrong Predictions (conf > 0.9):")
    header2 = f"  {'Model':<22}"
    for ds in dataset_names:
        short_name = ds[:18]
        header2 += f" | {short_name:>18}"
    print(header2)
    print("  " + "-" * (len(header2) - 2))

    for model_name, model_data in all_results.items():
        row = f"  {model_name:<22}"
        for ds in dataset_names:
            ds_result = model_data.get("datasets", {}).get(ds)
            if ds_result:
                hcw = ds_result["confidence_calibration"]["high_confidence_wrong_count"]
                total = ds_result["metrics"]["sample_count"]
                row += f" | {hcw:>8} / {total:<7}"
            else:
                row += f" | {'N/A':>18}"
        print(row)


# ─── 主函数 ──────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Sliced evaluation of BERT models on fair_test datasets"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_CONFIGS.keys()),
        default=list(MODEL_CONFIGS.keys()),
        help="Models to evaluate (default: all three)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 100)
    print("SLICED EVALUATION - BERT Models on fair_test Datasets")
    print("=" * 100)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Datasets: {len(DATASET_CONFIGS)}")
    print()

    # 预加载所有数据集
    datasets_loaded = {}
    for ds_name, ds_cfg in DATASET_CONFIGS.items():
        path = ds_cfg["path"]
        if not path.exists():
            print(f"WARNING: Dataset not found: {path}")
            continue
        df = pd.read_csv(path, encoding="utf-8-sig")
        if "text" not in df.columns or "label" not in df.columns:
            print(f"WARNING: Dataset {ds_name} missing 'text' or 'label' column, skipping")
            continue
        # 清理: 移除 NaN text
        df = df.dropna(subset=["text"])
        df["label"] = df["label"].astype(int)
        datasets_loaded[ds_name] = df
        print(f"Loaded {ds_name}: {len(df)} samples "
              f"(Human: {(df['label']==0).sum()}, AI: {(df['label']==1).sum()})")

    if not datasets_loaded:
        print("ERROR: No datasets loaded. Exiting.")
        sys.exit(1)

    # 逐模型评估
    all_results = {}

    for model_name in args.models:
        cfg = MODEL_CONFIGS[model_name]
        model_path = cfg["path"]
        max_length = cfg["max_length"]

        print(f"\n{'=' * 100}")
        print(f"MODEL: {model_name} ({cfg['desc']})")
        print(f"  Path: {model_path}")
        print(f"  Max length: {max_length}")
        print(f"{'=' * 100}")

        if not model_path.exists():
            print(f"  ERROR: Model path does not exist, skipping.")
            continue

        try:
            tokenizer = BertTokenizer.from_pretrained(str(model_path))
            model = BertForSequenceClassification.from_pretrained(str(model_path)).to(device)
            model.eval()
        except Exception as e:
            print(f"  ERROR: Failed to load model: {e}")
            continue

        model_results = {
            "desc": cfg["desc"],
            "max_length": max_length,
            "datasets": {},
        }

        for ds_name, df in datasets_loaded.items():
            result = evaluate_single_dataset(
                model, tokenizer, max_length, df, ds_name, device
            )
            model_results["datasets"][ds_name] = result
            print_dataset_report(ds_name, result)

        all_results[model_name] = model_results

        # 释放显存
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 跨模型汇总
    if len(all_results) > 1:
        print_cross_model_summary(all_results)

    # 保存 JSON 结果
    output_path = PROJECT_ROOT / "datasets" / "eval" / "fair_test" / "sliced_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_path}")
    print("Done.")


if __name__ == "__main__":
    main()
