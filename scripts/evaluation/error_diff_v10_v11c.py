#!/usr/bin/env python3
"""
V11c Error Diff Analysis - 对比 V10 vs V11c 的逐样本预测差异

输出:
  - 新增错误 (V10 correct, V11c wrong)
  - 修复错误 (V10 wrong, V11c correct)
  - 按长度桶和标签分组统计
"""

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import accuracy_score

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

LENGTH_BINS = [(0, 64), (64, 128), (128, 256), (256, 512), (512, 99999)]
BIN_NAMES = ["0-64", "64-128", "128-256", "256-512", "512+"]


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


def predict(model, tokenizer, texts, labels, device):
    ds = TextDataset(texts, labels, tokenizer, max_len=256)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    model.eval()
    all_preds, all_confs = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            logits = model(input_ids=ids, attention_mask=mask).logits
            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1).cpu().numpy()
            confs = probs.max(dim=-1).values.cpu().numpy()
            all_preds.extend(preds)
            all_confs.extend(confs)
    return np.array(all_preds), np.array(all_confs)


def get_length_bin(length):
    for i, (lo, hi) in enumerate(LENGTH_BINS):
        if lo <= length < hi:
            return BIN_NAMES[i]
    return BIN_NAMES[-1]


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    models_info = {
        "V10": ROOT / "models" / "bert_v10_augmented",
        "V11c": ROOT / "models" / "bert_v11c_boundary_fix",
    }

    loaded = {}
    for name, path in models_info.items():
        print(f"Loading {name} from {path}...")
        tok = BertTokenizer.from_pretrained(str(path))
        mdl = BertForSequenceClassification.from_pretrained(str(path), num_labels=2).to(device)
        mdl.eval()
        loaded[name] = (tok, mdl)

    report = {}

    for eval_name, eval_path in EVAL_SETS:
        print(f"\n{'=' * 70}")
        print(f"  {eval_name}")
        print(f"{'=' * 70}")

        df = pd.read_csv(eval_path)
        df = df.dropna(subset=['text', 'label'])
        texts = df['text'].astype(str).tolist()
        labels = df['label'].astype(int).values
        lengths = [len(t) for t in texts]

        preds_v10, confs_v10 = predict(loaded["V10"][1], loaded["V10"][0], texts, labels, device)
        preds_v11c, confs_v11c = predict(loaded["V11c"][1], loaded["V11c"][0], texts, labels, device)

        v10_correct = preds_v10 == labels
        v11c_correct = preds_v11c == labels

        # New errors: V10 correct, V11c wrong
        new_errors = v10_correct & ~v11c_correct
        # Fixed: V10 wrong, V11c correct
        fixed = ~v10_correct & v11c_correct
        # Both wrong
        both_wrong = ~v10_correct & ~v11c_correct

        print(f"\n  V10 accuracy:  {v10_correct.mean()*100:.2f}% ({v10_correct.sum()}/{len(labels)})")
        print(f"  V11c accuracy: {v11c_correct.mean()*100:.2f}% ({v11c_correct.sum()}/{len(labels)})")
        print(f"\n  New errors (V10 ok -> V11c wrong): {new_errors.sum()}")
        print(f"  Fixed errors (V10 wrong -> V11c ok): {fixed.sum()}")
        print(f"  Both wrong: {both_wrong.sum()}")
        print(f"  Net change: {fixed.sum() - new_errors.sum():+d}")

        set_report = {
            "v10_accuracy": round(v10_correct.mean() * 100, 2),
            "v11c_accuracy": round(v11c_correct.mean() * 100, 2),
            "new_errors": int(new_errors.sum()),
            "fixed_errors": int(fixed.sum()),
            "both_wrong": int(both_wrong.sum()),
            "net_change": int(fixed.sum()) - int(new_errors.sum()),
        }

        # Detail new errors
        if new_errors.sum() > 0:
            print(f"\n  --- New Errors Detail ---")
            new_err_details = []
            for idx in np.where(new_errors)[0]:
                label_str = "AI" if labels[idx] == 1 else "Human"
                pred_str = "AI" if preds_v11c[idx] == 1 else "Human"
                text_preview = texts[idx][:100].replace('\n', ' ')
                lb = get_length_bin(lengths[idx])
                source = df.iloc[idx].get('source', 'N/A') if 'source' in df.columns else 'N/A'
                conf_v11c = confs_v11c[idx]
                conf_v10 = confs_v10[idx]
                print(f"    [{idx}] len={lengths[idx]} bin={lb} label={label_str} pred={pred_str} v11c_conf={conf_v11c:.4f} v10_conf={conf_v10:.4f} source={source}")
                print(f"         {text_preview}")
                new_err_details.append({
                    "index": int(idx),
                    "length": lengths[idx],
                    "bin": lb,
                    "label": label_str,
                    "v11c_pred": pred_str,
                    "v11c_conf": round(float(conf_v11c), 4),
                    "v10_conf": round(float(conf_v10), 4),
                    "source": str(source),
                    "text_preview": text_preview,
                })
            set_report["new_error_details"] = new_err_details

        # Detail fixed errors
        if fixed.sum() > 0:
            print(f"\n  --- Fixed Errors Detail ---")
            fixed_details = []
            for idx in np.where(fixed)[0]:
                label_str = "AI" if labels[idx] == 1 else "Human"
                pred_str = "AI" if preds_v11c[idx] == 1 else "Human"
                text_preview = texts[idx][:100].replace('\n', ' ')
                lb = get_length_bin(lengths[idx])
                source = df.iloc[idx].get('source', 'N/A') if 'source' in df.columns else 'N/A'
                conf_v11c = confs_v11c[idx]
                conf_v10 = confs_v10[idx]
                print(f"    [{idx}] len={lengths[idx]} bin={lb} label={label_str} pred={pred_str} v11c_conf={conf_v11c:.4f} v10_conf={conf_v10:.4f} source={source}")
                print(f"         {text_preview}")
                fixed_details.append({
                    "index": int(idx),
                    "length": lengths[idx],
                    "bin": lb,
                    "label": label_str,
                    "v11c_pred": pred_str,
                    "v11c_conf": round(float(conf_v11c), 4),
                    "v10_conf": round(float(conf_v10), 4),
                    "source": str(source),
                    "text_preview": text_preview,
                })
            set_report["fixed_error_details"] = fixed_details

        # Length bin summary
        print(f"\n  --- Length Bin Accuracy ---")
        print(f"  {'Bin':<12} {'Count':>6} {'V10 Acc':>8} {'V11c Acc':>9} {'Delta':>7}")
        print(f"  {'-'*12} {'-'*6} {'-'*8} {'-'*9} {'-'*7}")
        bin_stats = []
        for bn, (lo, hi) in zip(BIN_NAMES, LENGTH_BINS):
            mask = np.array([lo <= l < hi for l in lengths])
            if mask.sum() == 0:
                continue
            v10_acc_bin = v10_correct[mask].mean() * 100
            v11c_acc_bin = v11c_correct[mask].mean() * 100
            delta = v11c_acc_bin - v10_acc_bin
            print(f"  {bn:<12} {mask.sum():>6} {v10_acc_bin:>7.2f}% {v11c_acc_bin:>8.2f}% {delta:>+6.2f}")
            bin_stats.append({
                "bin": bn,
                "count": int(mask.sum()),
                "v10_acc": round(v10_acc_bin, 2),
                "v11c_acc": round(v11c_acc_bin, 2),
                "delta": round(delta, 2),
            })
        set_report["length_bins"] = bin_stats

        # Label split
        for lab, lab_name in [(0, "Human"), (1, "AI")]:
            mask = labels == lab
            if mask.sum() == 0:
                continue
            v10_a = v10_correct[mask].mean() * 100
            v11c_a = v11c_correct[mask].mean() * 100
            print(f"\n  {lab_name} (n={mask.sum()}): V10={v10_a:.2f}% V11c={v11c_a:.2f}% delta={v11c_a-v10_a:+.2f}")

        report[eval_name] = set_report

    # Save
    out_path = ROOT / "docs" / "plans" / "v11c_error_diff_analysis.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved: {out_path}")

    # Cleanup
    for name, (tok, mdl) in loaded.items():
        del mdl
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
