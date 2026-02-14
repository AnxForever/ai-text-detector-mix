#!/usr/bin/env python3
"""
Temperature Scaling 校准 - 对已有模型进行后置校准

原理：
  - 在验证集上学习一个标量温度参数 T
  - 推理时 logits / T → softmax → 置信度更合理
  - 不改变准确率（argmax 不受 T 影响），只改善置信度校准

用法:
    # 对 V7 模型校准并评估
    python scripts/evaluation/calibrate_temperature.py

    # 指定模型
    python scripts/evaluation/calibrate_temperature.py --model bert_v7_improved
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

MODEL_CONFIGS = {
    "bert_v7_improved": {
        "path": PROJECT_ROOT / "models" / "bert_v7_improved",
        "max_length": 256,
    },
    "bert_v6_merged": {
        "path": PROJECT_ROOT / "models" / "bert_v6_merged",
        "max_length": 256,
    },
    "bert_v8_calibrated": {
        "path": PROJECT_ROOT / "models" / "bert_v8_calibrated",
        "max_length": 256,
    },
    "bert_v9_p0_supplement": {
        "path": PROJECT_ROOT / "models" / "bert_v9_p0_supplement",
        "max_length": 256,
    },
    "bert_v10_augmented": {
        "path": PROJECT_ROOT / "models" / "bert_v10_augmented",
        "max_length": 256,
    },
    "bert_v11_candidate": {
        "path": PROJECT_ROOT / "models" / "bert_v11_candidate",
        "max_length": 256,
    },
    "bert_v11a_clean": {
        "path": PROJECT_ROOT / "models" / "bert_v11a_clean",
        "max_length": 256,
    },
    "bert_v11b_augmented": {
        "path": PROJECT_ROOT / "models" / "bert_v11b_augmented",
        "max_length": 256,
    },
    "bert_v11c_boundary_fix": {
        "path": PROJECT_ROOT / "models" / "bert_v11c_boundary_fix",
        "max_length": 256,
    },
}

# 校准用验证集（必须是模型没训练过的干净数据）
# 使用 independent_data 因为它是唯一零污染的评估集
CALIBRATION_SET = PROJECT_ROOT / "datasets" / "eval" / "fair_test" / "independent_data.csv"

FAIR_TEST_SETS = {
    "core_v1_test_clean": PROJECT_ROOT / "datasets" / "eval" / "fair_test" / "core_v1_test_clean.csv",
    "independent_data": PROJECT_ROOT / "datasets" / "eval" / "fair_test" / "independent_data.csv",
    "merged_v2_val_clean": PROJECT_ROOT / "datasets" / "eval" / "fair_test" / "merged_v2_val_clean.csv",
}

BATCH_SIZE = 32


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
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


class TemperatureScaler(nn.Module):
    """学习一个标量温度参数 T，使 logits / T 后的置信度更准确"""

    def __init__(self):
        super().__init__()
        # 初始化 T=1.0（log(1.0)=0），用 log 空间确保 T > 0
        self.log_temperature = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self):
        return self.log_temperature.exp()

    def forward(self, logits):
        return logits / self.temperature


def collect_logits(model, dataloader, device):
    """收集模型在数据集上的所有 logits 和标签"""
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(outputs.logits.cpu())
            all_labels.append(labels)

    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def learn_temperature(logits, labels, max_iter=200, lr=0.01):
    """在验证集上优化温度参数 T，最小化 NLL"""
    scaler = TemperatureScaler()
    nll_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([scaler.log_temperature], lr=lr, max_iter=max_iter)

    def eval_closure():
        optimizer.zero_grad()
        scaled_logits = scaler(logits)
        loss = nll_criterion(scaled_logits, labels)
        loss.backward()
        return loss

    optimizer.step(eval_closure)

    return scaler.temperature.item()


def compute_calibration_metrics(probs, labels, n_bins=10):
    """计算 ECE (Expected Calibration Error) 和分桶校准指标"""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_stats = []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        count = mask.sum()

        if count == 0:
            bin_stats.append({
                "range": f"{lo:.1f}-{hi:.1f}",
                "count": 0,
                "avg_confidence": None,
                "avg_accuracy": None,
                "gap": None,
            })
            continue

        avg_conf = confidences[mask].mean()
        avg_acc = accuracies[mask].mean()
        gap = abs(avg_conf - avg_acc)
        ece += gap * (count / len(labels))

        bin_stats.append({
            "range": f"{lo:.1f}-{hi:.1f}",
            "count": int(count),
            "avg_confidence": float(avg_conf),
            "avg_accuracy": float(avg_acc),
            "gap": float(gap),
        })

    return {
        "ece": float(ece),
        "bins": bin_stats,
        "avg_confidence": float(confidences.mean()),
        "avg_confidence_correct": float(confidences[accuracies == 1].mean()) if (accuracies == 1).any() else None,
        "avg_confidence_wrong": float(confidences[accuracies == 0].mean()) if (accuracies == 0).any() else None,
        "high_conf_wrong": int(((confidences > 0.9) & (accuracies == 0)).sum()),
        "total_wrong": int((accuracies == 0).sum()),
        "accuracy": float(accuracies.mean()),
    }


def main():
    parser = argparse.ArgumentParser(description="Temperature Scaling 校准")
    parser.add_argument(
        "--model",
        default="bert_v10_augmented",
        choices=list(MODEL_CONFIGS.keys()),
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional explicit model path (overrides --model preset).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Optional max token length override.",
    )
    args = parser.parse_args()

    model_name = args.model
    config = MODEL_CONFIGS[model_name]
    model_path = Path(args.model_path) if args.model_path else config["path"]
    max_length = args.max_length if args.max_length else config["max_length"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print(f"模型: {model_name} ({model_path})")
    print(f"max_length: {max_length}")
    print()

    # 加载模型和分词器
    print("加载模型...")
    tokenizer = BertTokenizer.from_pretrained(str(model_path))
    model = BertForSequenceClassification.from_pretrained(str(model_path))
    model.to(device)
    model.eval()

    # ═══════════════════════════════════════════════════════════════════
    # 第一步：在校准集上学习最优温度
    # ═══════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("第一步：学习最优温度 T")
    print(f"校准集: {CALIBRATION_SET.name}")
    print("=" * 70)

    cal_df = pd.read_csv(CALIBRATION_SET, encoding="utf-8-sig")
    cal_df = cal_df.dropna(subset=["text", "label"])
    cal_texts = cal_df["text"].tolist()
    cal_labels = cal_df["label"].astype(int).tolist()
    print(f"校准集样本数: {len(cal_texts)}")

    cal_dataset = TextDataset(cal_texts, cal_labels, tokenizer, max_length)
    cal_loader = DataLoader(cal_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("收集 logits...")
    cal_logits, cal_labels_tensor = collect_logits(model, cal_loader, device)

    # 校准前的 NLL 和 ECE
    before_probs = torch.softmax(cal_logits, dim=1).numpy()
    before_metrics = compute_calibration_metrics(before_probs, cal_labels_tensor.numpy())
    print(f"\n校准前 (T=1.0):")
    print(f"  ECE = {before_metrics['ece']:.4f}")
    print(f"  准确率 = {before_metrics['accuracy']:.2%}")
    print(f"  平均置信度 = {before_metrics['avg_confidence']:.2%}")
    print(f"  正确预测置信度 = {before_metrics['avg_confidence_correct']:.2%}")
    print(f"  错误预测置信度 = {before_metrics['avg_confidence_wrong']:.2%}")
    print(f"  高置信错判(>0.9) = {before_metrics['high_conf_wrong']}/{before_metrics['total_wrong']}")

    # 学习温度
    print("\n优化温度参数...")
    optimal_T = learn_temperature(cal_logits, cal_labels_tensor)
    print(f"  最优温度 T = {optimal_T:.4f}")

    # 校准后
    after_probs = torch.softmax(cal_logits / optimal_T, dim=1).numpy()
    after_metrics = compute_calibration_metrics(after_probs, cal_labels_tensor.numpy())
    print(f"\n校准后 (T={optimal_T:.4f}):")
    print(f"  ECE = {after_metrics['ece']:.4f} (变化: {after_metrics['ece'] - before_metrics['ece']:+.4f})")
    print(f"  准确率 = {after_metrics['accuracy']:.2%} (不变)")
    print(f"  平均置信度 = {after_metrics['avg_confidence']:.2%}")
    print(f"  正确预测置信度 = {after_metrics['avg_confidence_correct']:.2%}")
    print(f"  错误预测置信度 = {after_metrics['avg_confidence_wrong']:.2%}")
    print(f"  高置信错判(>0.9) = {after_metrics['high_conf_wrong']}/{after_metrics['total_wrong']}")

    # ═══════════════════════════════════════════════════════════════════
    # 第二步：用最优 T 在所有 fair_test 上评估
    # ═══════════════════════════════════════════════════════════════════
    print()
    print("=" * 70)
    print(f"第二步：用 T={optimal_T:.4f} 在 fair_test 上评估")
    print("=" * 70)

    results = {"model": model_name, "optimal_temperature": optimal_T, "datasets": {}}

    for ds_name, ds_path in FAIR_TEST_SETS.items():
        print(f"\n  [{ds_name}]")
        df = pd.read_csv(ds_path, encoding="utf-8-sig")
        df = df.dropna(subset=["text", "label"])
        texts = df["text"].tolist()
        labels_list = df["label"].astype(int).tolist()
        char_lengths = np.array([len(str(t)) for t in texts])
        print(f"    样本数: {len(texts)}")

        dataset = TextDataset(texts, labels_list, tokenizer, max_length)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

        logits, labels_tensor = collect_logits(model, loader, device)
        labels_np = labels_tensor.numpy()

        # 校准前
        probs_before = torch.softmax(logits, dim=1).numpy()
        preds_before = np.argmax(probs_before, axis=1)
        metrics_before = compute_calibration_metrics(probs_before, labels_np)

        # 校准后
        probs_after = torch.softmax(logits / optimal_T, dim=1).numpy()
        preds_after = np.argmax(probs_after, axis=1)
        metrics_after = compute_calibration_metrics(probs_after, labels_np)

        # 准确率验证（应该不变）
        acc_before = accuracy_score(labels_np, preds_before)
        acc_after = accuracy_score(labels_np, preds_after)

        print(f"    准确率: {acc_before:.2%} → {acc_after:.2%}")
        print(f"    ECE:    {metrics_before['ece']:.4f} → {metrics_after['ece']:.4f} ({metrics_after['ece'] - metrics_before['ece']:+.4f})")
        print(f"    平均置信度: {metrics_before['avg_confidence']:.2%} → {metrics_after['avg_confidence']:.2%}")
        print(f"    错判置信度: {metrics_before['avg_confidence_wrong']:.2%} → {metrics_after['avg_confidence_wrong']:.2%}")
        print(f"    高置信错判: {metrics_before['high_conf_wrong']} → {metrics_after['high_conf_wrong']}")

        # 长度桶分析（校准后）
        print(f"\n    长度桶 (校准后):")
        print(f"    {'桶':<12} {'数量':>6} {'准确率':>8} {'平均置信度':>12} {'ECE':>8}")
        print(f"    {'-'*12} {'-'*6} {'-'*8} {'-'*12} {'-'*8}")

        for lo, hi, bucket_name in [(0, 64, "0-64"), (64, 128, "64-128"),
                                     (128, 256, "128-256"), (256, 512, "256-512"),
                                     (512, float("inf"), "512+")]:
            mask = (char_lengths >= lo) & (char_lengths < hi)
            count = mask.sum()
            if count == 0:
                continue
            bucket_acc = accuracy_score(labels_np[mask], preds_after[mask])
            bucket_conf = probs_after[mask].max(axis=1).mean()
            bucket_preds_correct = (preds_after[mask] == labels_np[mask]).astype(float)
            bucket_ece = abs(bucket_conf - bucket_acc)
            print(f"    {bucket_name:<12} {count:>6} {bucket_acc:>8.2%} {bucket_conf:>12.2%} {bucket_ece:>8.4f}")

        results["datasets"][ds_name] = {
            "before": {
                "accuracy": float(acc_before),
                "ece": metrics_before["ece"],
                "avg_confidence": metrics_before["avg_confidence"],
                "avg_confidence_wrong": metrics_before["avg_confidence_wrong"],
                "high_conf_wrong": metrics_before["high_conf_wrong"],
                "total_wrong": metrics_before["total_wrong"],
            },
            "after": {
                "accuracy": float(acc_after),
                "ece": metrics_after["ece"],
                "avg_confidence": metrics_after["avg_confidence"],
                "avg_confidence_wrong": metrics_after["avg_confidence_wrong"],
                "high_conf_wrong": metrics_after["high_conf_wrong"],
                "total_wrong": metrics_after["total_wrong"],
            },
        }

    # ═══════════════════════════════════════════════════════════════════
    # 第三步：总结
    # ═══════════════════════════════════════════════════════════════════
    print()
    print("=" * 70)
    print("总结")
    print("=" * 70)
    print(f"  最优温度 T = {optimal_T:.4f}")
    print()
    print(f"  {'数据集':<25} {'ECE前':>8} {'ECE后':>8} {'变化':>8} {'高置信错判前':>12} {'后':>4}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*12} {'-'*4}")

    for ds_name, ds_result in results["datasets"].items():
        b, a = ds_result["before"], ds_result["after"]
        print(f"  {ds_name:<25} {b['ece']:>8.4f} {a['ece']:>8.4f} {a['ece']-b['ece']:>+8.4f} "
              f"{b['high_conf_wrong']:>12} {a['high_conf_wrong']:>4}")

    print()
    print("  使用方式：推理时将 logits 除以 T 后再 softmax")
    print(f"  scaled_probs = softmax(logits / {optimal_T:.4f})")

    # 保存结果
    output_path = PROJECT_ROOT / "datasets" / "eval" / "fair_test" / "calibration_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已保存: {output_path}")


if __name__ == "__main__":
    main()
