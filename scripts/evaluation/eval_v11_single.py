#!/usr/bin/env python3
"""
V11 单模型评估脚本 - 在三个公平评估集上评估指定模型

输出兼容 check_v11_regression_gate.py 的 eval_comparison.json 格式。

用法:
    python scripts/evaluation/eval_v11_single.py --model bert_v11a_clean
    python scripts/evaluation/eval_v11_single.py --model bert_v11b_augmented
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scipy.optimize import minimize_scalar

warnings.filterwarnings('ignore')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = ROOT / "datasets" / "eval" / "fair_test"

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
    import torch.nn.functional as F

    def nll(t):
        scaled = logits / t
        log_probs = F.log_softmax(scaled, dim=-1)
        return -log_probs[range(len(labels)), labels].mean().item()

    result = minimize_scalar(nll, bounds=(min_t, max_t), method='bounded')
    return result.x


def compute_ece(probs, labels, n_bins=10):
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
    predictions = probs.argmax(dim=1).numpy()
    confidences = probs.max(dim=1).values.numpy()
    labels_np = labels.numpy()
    wrong = predictions != labels_np
    high_conf = confidences >= threshold
    return int((wrong & high_conf).sum())


def eval_model(model_name, model_path, device):
    print(f"\n{'=' * 70}")
    print(f"  Evaluating: {model_name}")
    print(f"  Path: {model_path}")
    print(f"{'=' * 70}")

    tokenizer = BertTokenizer.from_pretrained(str(model_path))
    model = BertForSequenceClassification.from_pretrained(
        str(model_path), num_labels=2
    ).to(device)
    model.eval()

    results = {"model": model_name}

    for eval_name, eval_path in EVAL_SETS:
        if not eval_path.exists():
            print(f"  {eval_name}: file not found, skipping")
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

        print(f"\n  [{eval_name}] ({len(df)} samples, {elapsed:.1f}s)")
        print(f"    Acc: {acc*100:.2f}%  P: {p*100:.2f}%  R: {r*100:.2f}%  F1: {f1*100:.2f}%")

        # independent_data detailed analysis
        if eval_name == "independent_data" and 'source' in df.columns:
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

            print(f"\n    AI source detection rates:")
            ai_sources = df[df['label'] == 1]['source'].dropna().unique()
            for src in sorted(ai_sources):
                mask = (df['source'] == src) & (df['label'] == 1)
                if mask.sum() == 0:
                    continue
                src_preds_ai = preds[mask.values]
                detected = (src_preds_ai == 1).sum()
                total = len(src_preds_ai)
                rate = detected / total * 100
                print(f"      {src}: {detected}/{total} ({rate:.1f}%)")

            # Error analysis
            errors = df.iloc[preds != labels_np]
            results[f"{eval_name}_errors"] = len(errors)
            fn_mask = (preds == 0) & (labels_np == 1)
            fp_mask = (preds == 1) & (labels_np == 0)
            print(f"\n    Errors: FN={fn_mask.sum()}, FP={fp_mask.sum()}")

            # Temperature Scaling
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

            print(f"\n    Temperature Scaling: T={opt_t:.4f}")
            print(f"      ECE: {ece_before:.4f} -> {ece_after:.4f}")
            print(f"      High-conf errors: {hce_before} -> {hce_after}")

    # Three-set average
    accs = []
    for eval_name, _ in EVAL_SETS:
        if eval_name in results:
            accs.append(results[eval_name]["accuracy"])
    if accs:
        results["three_set_avg"] = round(sum(accs) / len(accs), 2)
        print(f"\n  Three-set average: {results['three_set_avg']:.2f}%")

    del model
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a single model on 3 fair test sets")
    parser.add_argument("--model", required=True, help="Model directory name (e.g. bert_v11a_clean)")
    parser.add_argument("--model-path", default=None, help="Full model path (overrides --model)")
    args = parser.parse_args()

    model_name = args.model
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        model_path = ROOT / "models" / model_name

    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    result = eval_model(model_name, model_path, device)

    # Save eval_comparison.json in model directory
    output = {model_name: result}
    output_path = model_path / "eval_comparison.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {output_path}")


if __name__ == '__main__':
    main()
