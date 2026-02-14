#!/usr/bin/env python3
"""
V11c 训练脚本 - 清洗数据 + 弱域补充 + 长AI边界修复

基于 V10 完全相同配置，唯一变量为训练数据。
  - V10 数据: train_v10.csv (62,980 行)
  - V11a 数据: train_v11_candidate.csv (60,456 行, 过滤 hard pattern + 未审核 unknown)
  - V11b 数据: train_v11b_candidate.csv (61,056 行, V11a + 600 弱域补充)
  - V11c 数据: train_v11c_candidate.csv (63,187 行, V11b + 2131 长AI边界修复)

控制变量设计：
  - Base model: bert_v7_improved (与 V10/V11a/V11b 相同)
  - 所有超参、损失函数、采样策略均沿用 V10
  - 目的: 在 V11b 基础上恢复 256+ 长度桶 AI 覆盖至 V10 水平，修复决策边界偏移

用法:
    python scripts/training/train_v11c.py
"""
import os
import sys
import time
import json
import gc
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

ROOT = Path(__file__).resolve().parent.parent.parent


# ──────────────────────────────────────────────
#  Config (与 V10 完全一致，仅更换训练数据和输出路径)
# ──────────────────────────────────────────────

CONFIG = {
    "base_model": str(ROOT / "models" / "bert_v7_improved"),
    "train_data": str(ROOT / "datasets" / "merged_v2" / "train_v11c_candidate.csv"),
    "val_data": str(ROOT / "datasets" / "merged_v2" / "val.csv"),
    "output": str(ROOT / "models" / "bert_v11c_boundary_fix"),
    "batch_size": 8,
    "max_length": 256,
    "epochs": 5,
    "learning_rate": 1e-5,
    "label_smoothing": 0.05,
    "length_penalty_weight": 0.1,
    "accum_steps": 4,
    "patience": 2,
    "min_text_length": 10,
    "max_text_length": 5000,
}


# ──────────────────────────────────────────────
#  Dataset
# ──────────────────────────────────────────────

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.lengths = [len(str(t)) for t in texts]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
            'length': torch.tensor(self.lengths[idx], dtype=torch.float),
        }


# ──────────────────────────────────────────────
#  Loss with Label Smoothing + Length Awareness
# ──────────────────────────────────────────────

class LengthAwareLabelSmoothingLoss(nn.Module):
    def __init__(self, label_smoothing=0.05, length_penalty_weight=0.1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(
            reduction='none',
            label_smoothing=label_smoothing
        )
        self.length_penalty_weight = length_penalty_weight

    def forward(self, logits, labels, lengths, mean_length=500):
        ce = self.ce(logits, labels)
        length_ratio = lengths / mean_length
        weight = 1.0 / (1.0 + self.length_penalty_weight * torch.abs(
            torch.log(length_ratio + 1e-6)
        ))
        return (ce * weight).mean()


# ──────────────────────────────────────────────
#  Data loading & cleaning
# ──────────────────────────────────────────────

def load_and_clean(path, min_len, max_len):
    df = pd.read_csv(path)
    before = len(df)
    df = df.dropna(subset=['text', 'label'])
    df['text'] = df['text'].astype(str)
    df['char_len'] = df['text'].str.len()
    df = df[(df['char_len'] >= min_len) & (df['char_len'] <= max_len)]
    df = df.drop_duplicates(subset=['text'], keep='first')
    df['label'] = df['label'].astype(int)

    ai_phrases = ["作为一个AI", "作为AI", "AI语言模型", "As an AI"]
    for phrase in ai_phrases:
        df = df[~df['text'].str.contains(phrase, na=False)]

    print(f"  {path}: {before} -> {len(df)} (cleaned {before - len(df)})")
    return df


def build_sampler(df):
    bins = [0, 100, 200, 500, 1000, 2000, float('inf')]
    df = df.copy()
    df['len_bin'] = pd.cut(df['char_len'], bins=bins, labels=False)

    bin_weights = 1.0 / df['len_bin'].value_counts()
    label_weights = 1.0 / df['label'].value_counts()

    w = df['len_bin'].map(bin_weights).values.astype(np.float64)
    w *= df['label'].map(label_weights).values.astype(np.float64)

    return WeightedRandomSampler(w, num_samples=len(df), replacement=True)


# ──────────────────────────────────────────────
#  Train / Evaluate
# ──────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scheduler, device, loss_fn, accum):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc="  Train")
    for i, batch in enumerate(pbar):
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        lengths = batch['length'].to(device)

        outputs = model(input_ids=ids, attention_mask=mask)
        loss = loss_fn(outputs.logits, labels, lengths) / accum
        loss.backward()

        if (i + 1) % accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accum
        preds = outputs.logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if (i + 1) % 500 == 0:
            torch.cuda.empty_cache()
            gc.collect()

        pbar.set_postfix(loss=f'{total_loss/(i+1):.4f}', acc=f'{correct/total:.4f}')

    return total_loss / len(loader), correct / total


def evaluate(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss = 0
    all_preds, all_labels, all_lengths = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Eval"):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            out = model(input_ids=ids, attention_mask=mask)
            total_loss += ce(out.logits, labels).item()

            all_preds.extend(out.logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_lengths.extend(batch['length'].numpy())

    preds = np.array(all_preds)
    labels = np.array(all_labels)
    lengths = np.array(all_lengths)

    loss = total_loss / len(loader)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')

    print("\n  Length-segmented accuracy:")
    for name, lo, hi in [("Short <128", 0, 128), ("128-256", 128, 256),
                          ("256-512", 256, 512), ("512+", 512, 99999)]:
        m = (lengths >= lo) & (lengths < hi)
        if m.sum() > 0:
            print(f"    {name}: {(preds[m]==labels[m]).mean():.4f} ({m.sum()} samples)")

    return loss, acc, p, r, f1


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def main():
    cfg = CONFIG

    print("=" * 70)
    print("  V11c BERT Training - Cleaned + weak-domain + long-AI boundary fix")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    eff_batch = cfg['batch_size'] * cfg['accum_steps']
    print(f"\nConfig:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    print(f"  effective_batch: {eff_batch}")

    # Load model
    print(f"\n>>> Loading base model: {cfg['base_model']}")
    tokenizer = BertTokenizer.from_pretrained(cfg['base_model'])
    model = BertForSequenceClassification.from_pretrained(
        cfg['base_model'], num_labels=2
    ).to(device)

    # Load data
    print("\n>>> Loading data")
    train_df = load_and_clean(cfg['train_data'], cfg['min_text_length'], cfg['max_text_length'])
    val_df = load_and_clean(cfg['val_data'], cfg['min_text_length'], cfg['max_text_length'])

    print(f"\nData stats:")
    print(f"  Train: {len(train_df)} (AI: {(train_df['label']==1).sum()}, Human: {(train_df['label']==0).sum()})")
    print(f"  Val:   {len(val_df)} (AI: {(val_df['label']==1).sum()}, Human: {(val_df['label']==0).sum()})")

    # Source distribution
    if 'source' in train_df.columns:
        print(f"\n  Top-10 sources:")
        for src, cnt in train_df['source'].value_counts().head(10).items():
            print(f"    {src}: {cnt}")

    # DataLoaders
    train_ds = TextDataset(train_df['text'].tolist(), train_df['label'].tolist(),
                           tokenizer, cfg['max_length'])
    val_ds = TextDataset(val_df['text'].tolist(), val_df['label'].tolist(),
                         tokenizer, cfg['max_length'])

    sampler = build_sampler(train_df)
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'],
                              sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'],
                            shuffle=False, num_workers=0)

    # Optimizer
    loss_fn = LengthAwareLabelSmoothingLoss(
        label_smoothing=cfg['label_smoothing'],
        length_penalty_weight=cfg['length_penalty_weight'],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=0.01)

    total_steps = len(train_loader) * cfg['epochs'] // cfg['accum_steps']
    warmup_steps = total_steps // 10
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    output_dir = Path(cfg['output'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training
    print("\n" + "=" * 70)
    print("  Start training")
    print("=" * 70)

    best_acc = 0
    patience_counter = 0
    results = []

    for epoch in range(cfg['epochs']):
        t0 = time.time()
        print(f"\n{'─' * 70}")
        print(f"  Epoch {epoch + 1}/{cfg['epochs']}")
        print(f"{'─' * 70}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            device, loss_fn, cfg['accum_steps']
        )

        val_loss, val_acc, val_p, val_r, val_f1 = evaluate(model, val_loader, device)

        elapsed = time.time() - t0

        print(f"\n  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        print(f"          P: {val_p:.4f}, R: {val_r:.4f}, F1: {val_f1:.4f}")
        print(f"  Time: {elapsed:.1f}s")

        results.append({
            'epoch': epoch + 1,
            'train_loss': round(train_loss, 4),
            'train_acc': round(train_acc, 4),
            'val_loss': round(val_loss, 4),
            'val_acc': round(val_acc, 4),
            'val_precision': round(val_p, 4),
            'val_recall': round(val_r, 4),
            'val_f1': round(val_f1, 4),
            'time_sec': round(elapsed, 1),
        })

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            model.save_pretrained(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))
            print(f"  >>> Saved best model (Val Acc={val_acc:.4f})")
        else:
            patience_counter += 1
            print(f"  Val Acc not improved ({patience_counter}/{cfg['patience']})")
            if patience_counter >= cfg['patience']:
                print(f"\n  Early stopping at epoch {epoch + 1}")
                break

        torch.cuda.empty_cache()
        gc.collect()

    # Save log
    log = {
        'version': 'v11c',
        'strategy': 'data_cleaning_plus_weak_domain_plus_long_ai_boundary_fix',
        'description': 'V11b config + 2131 long-AI boundary supplement (restore 256+ bucket to V10 levels)',
        'base_model': cfg['base_model'],
        'config': cfg,
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'data_changes_vs_v10': {
            'removed_hard_patterns': 750,
            'removed_unapproved_unknown': 1767,
            'removed_length_violations': 7,
            'added_weak_domain_formal_collected': 300,
            'added_weak_domain_llama_405b': 300,
            'added_long_ai_boundary_fix': 2131,
            'v10_rows': 62980,
            'v11a_rows': 60456,
            'v11b_rows': 61056,
            'v11c_rows': 63187,
            'net_change_vs_v10': 207,
        },
        'best_val_acc': round(best_acc, 5),
        'results': results,
    }
    log_path = output_dir / 'training_log.json'
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"  Training complete! Best Val Acc: {best_acc:.4f}")
    print(f"  Model: {output_dir}")
    print(f"  Log: {log_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
