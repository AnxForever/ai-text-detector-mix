#!/usr/bin/env python3
"""
V8 训练脚本 - 使用 merged_v3 扩充数据集训练 BERT 分类器

目标：解决 V5/V6 的技术文档漏检退化问题。
数据源：datasets/merged_v3（由 merge_for_v8.py 生成）

改进策略：
1. 数据清洗：移除过短/重复/AI模板短语
2. 长度感知损失：减轻"长文本=AI"偏差
3. 长度平衡采样：确保各长度区间均匀训练
4. 梯度累积：支持小显存 (batch_size=8, accum=4 ≈ effective 32)
5. Early stopping：防止过拟合
6. 按长度区间/来源分组评估：方便诊断

用法:
    python scripts/training/train_v8.py
    python scripts/training/train_v8.py --data datasets/merged_v3 --epochs 5
    python scripts/training/train_v8.py --model-path models/bert_v4_defense_focused
"""
import os
import sys
import time
import json
import gc
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from tqdm import tqdm

# 离线模式
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

ROOT = Path(__file__).resolve().parent.parent.parent


# ──────────────────────────────────────────────
#  Dataset
# ──────────────────────────────────────────────

class TextDataset(Dataset):
    """支持长度信息的文本数据集"""

    def __init__(self, texts, labels, tokenizer, max_len=256, sources=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.lengths = [len(str(t)) for t in texts]
        self.sources = sources or ["unknown"] * len(texts)

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
#  Loss
# ──────────────────────────────────────────────

class LengthAwareLoss(nn.Module):
    """长度感知损失：对长度极端的样本降低权重，减轻长度偏差"""

    def __init__(self, length_penalty_weight=0.1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.length_penalty_weight = length_penalty_weight

    def forward(self, logits, labels, lengths, mean_length=500):
        ce = self.ce(logits, labels)
        length_ratio = lengths / mean_length
        weight = 1.0 / (1.0 + self.length_penalty_weight * torch.abs(
            torch.log(length_ratio + 1e-6)
        ))
        return (ce * weight).mean()


# ──────────────────────────────────────────────
#  Data cleaning & sampling
# ──────────────────────────────────────────────

def clean_data(df, min_length=50, max_length=5000):
    """清洗数据"""
    before = len(df)
    df = df.dropna(subset=['text', 'label'])
    df['text'] = df['text'].astype(str)
    df['char_len'] = df['text'].str.len()
    df = df[(df['char_len'] >= min_length) & (df['char_len'] <= max_length)]
    df = df.drop_duplicates(subset=['text'], keep='first')

    # 移除明显的 AI 自我揭露短语
    ai_phrases = ["作为一个AI", "作为AI", "AI语言模型", "As an AI", "作为一个人工智能"]
    for phrase in ai_phrases:
        df = df[~df['text'].str.contains(phrase, na=False)]

    print(f"  清洗: {before} -> {len(df)} (移除 {before - len(df)})")
    return df


def build_length_balanced_sampler(df):
    """创建长度 + 标签联合平衡采样器"""
    bins = [0, 100, 200, 500, 1000, 2000, float('inf')]
    df = df.copy()
    df['len_bin'] = pd.cut(df['char_len'], bins=bins, labels=False)

    # 长度桶权重
    bin_counts = df['len_bin'].value_counts()
    bin_weights = 1.0 / bin_counts
    sample_weights = df['len_bin'].map(bin_weights).values.astype(np.float64)

    # 标签权重
    label_counts = df['label'].value_counts()
    label_weights = 1.0 / label_counts
    sample_weights *= df['label'].map(label_weights).values.astype(np.float64)

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(df),
        replacement=True
    )


# ──────────────────────────────────────────────
#  Model resolution
# ──────────────────────────────────────────────

def resolve_model_path(preferred: str) -> str:
    """按优先级查找本地可用的 BERT 模型"""
    weight_files = [
        "pytorch_model.bin", "model.safetensors",
        "tf_model.h5", "model.ckpt.index", "flax_model.msgpack",
    ]

    def has_weights(p):
        d = Path(p)
        return d.exists() and any((d / f).exists() for f in weight_files)

    candidates = [preferred]
    # HuggingFace cache
    cache = Path.home() / ".cache/huggingface/hub/models--bert-base-chinese"
    if cache.exists():
        snapshots = cache / "snapshots"
        if snapshots.exists():
            for snap in sorted(snapshots.iterdir(), reverse=True):
                candidates.append(str(snap))
    # 项目内已有模型作为 fallback
    candidates.extend([
        str(ROOT / "models" / "bert_v4_defense_focused"),
        str(ROOT / "models" / "bert_v6_merged"),
    ])

    for path in candidates:
        if has_weights(path):
            return path

    raise FileNotFoundError(
        "未找到本地 BERT 模型权重。请先下载 bert-base-chinese 或"
        "通过 --model-path 指定一个本地模型目录。"
    )


# ──────────────────────────────────────────────
#  Training / evaluation
# ──────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scheduler, device,
                    loss_fn, accum_steps=4):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc="  训练")
    for i, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        lengths = batch['length'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels, lengths) / accum_steps
        loss.backward()

        if (i + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps
        preds = torch.argmax(outputs.logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if (i + 1) % 500 == 0:
            torch.cuda.empty_cache()
            gc.collect()

        pbar.set_postfix({
            'loss': f'{total_loss / (i + 1):.4f}',
            'acc': f'{correct / total:.4f}'
        })

    return total_loss / len(loader), correct / total


def evaluate(model, loader, device, df_meta=None):
    """评估模型，返回 (loss, acc, precision, recall, f1)。
    若传入 df_meta 则额外打印按长度区间的细分指标。
    """
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss = 0
    all_preds, all_labels, all_lengths = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  评估"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            lengths = batch['length']

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            total_loss += ce(outputs.logits, labels).item()

            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_lengths.extend(lengths.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_lengths = np.array(all_lengths)

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary'
    )

    # 按长度区间细分
    print("\n  按长度区间准确率:")
    bins = [(0, 200, "短"), (200, 500, "中"), (500, 1000, "长"), (1000, 10000, "超长")]
    for lo, hi, name in bins:
        mask = (all_lengths >= lo) & (all_lengths < hi)
        if mask.sum() > 0:
            seg_acc = (all_preds[mask] == all_labels[mask]).mean()
            print(f"    {name} ({lo}-{hi}): {seg_acc:.4f} ({mask.sum()} 样本)")

    return avg_loss, acc, precision, recall, f1


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="V8 BERT 训练")
    parser.add_argument('--data', default='datasets/merged_v3', help='数据目录')
    parser.add_argument('--output', default='models/bert_v8_merged', help='输出目录')
    parser.add_argument('--model-path', default='bert-base-chinese', help='基础模型路径')
    parser.add_argument('--batch-size', type=int, default=8, help='批次大小')
    parser.add_argument('--max-len', type=int, default=256, help='最大序列长度')
    parser.add_argument('--epochs', type=int, default=5, help='训练轮数')
    parser.add_argument('--lr', type=float, default=2e-5, help='学习率')
    parser.add_argument('--accum-steps', type=int, default=4,
                        help='梯度累积步数 (effective batch = batch_size * accum)')
    parser.add_argument('--length-penalty', type=float, default=0.1,
                        help='长度惩罚权重')
    parser.add_argument('--patience', type=int, default=2,
                        help='Early stopping 容忍轮数')
    parser.add_argument('--no-balanced-sampler', action='store_true',
                        help='禁用长度平衡采样（默认启用）')
    args = parser.parse_args()

    data_dir = ROOT / args.data
    output_dir = ROOT / args.output

    print("=" * 70)
    print("  V8 BERT 训练 — 修复技术文档漏检")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    effective_batch = args.batch_size * args.accum_steps
    print(f"\n训练参数:")
    print(f"  data:              {data_dir}")
    print(f"  output:            {output_dir}")
    print(f"  batch_size:        {args.batch_size} (effective {effective_batch})")
    print(f"  max_len:           {args.max_len}")
    print(f"  epochs:            {args.epochs}")
    print(f"  lr:                {args.lr}")
    print(f"  length_penalty:    {args.length_penalty}")
    print(f"  balanced_sampler:  {not args.no_balanced_sampler}")
    print(f"  patience:          {args.patience}")

    # ── 加载模型 ──
    print("\n>>> 加载模型")
    model_path = resolve_model_path(args.model_path)
    print(f"  使用: {model_path}")
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(
        model_path, num_labels=2
    ).to(device)

    # ── 加载数据 ──
    print(f"\n>>> 加载数据: {data_dir}")
    train_path = data_dir / "train.csv"
    val_path = data_dir / "val.csv"

    if not train_path.exists():
        print(f"[ERROR] {train_path} 不存在，请先运行 merge_for_v8.py")
        sys.exit(1)

    train_df = pd.read_csv(train_path, encoding='utf-8-sig')
    val_df = pd.read_csv(val_path, encoding='utf-8-sig')

    print(f"  原始 train: {len(train_df)}, val: {len(val_df)}")

    # ── 清洗 ──
    print("\n清洗训练数据:")
    train_df = clean_data(train_df)
    print("清洗验证数据:")
    val_df = clean_data(val_df)

    train_df['label'] = train_df['label'].astype(int)
    val_df['label'] = val_df['label'].astype(int)

    print(f"\n最终数据:")
    print(f"  Train: {len(train_df)}")
    print(f"  Val:   {len(val_df)}")
    print(f"  Train 标签: {train_df['label'].value_counts().to_dict()}")

    for label, name in [(0, "Human"), (1, "AI")]:
        subset = train_df[train_df['label'] == label]['char_len']
        print(f"  {name} 长度: 均值={subset.mean():.0f}, 中位数={subset.median():.0f}")

    # ── Dataset & DataLoader ──
    train_dataset = TextDataset(
        train_df['text'].tolist(),
        train_df['label'].tolist(),
        tokenizer,
        max_len=args.max_len,
        sources=train_df['source'].tolist() if 'source' in train_df.columns else None,
    )
    val_dataset = TextDataset(
        val_df['text'].tolist(),
        val_df['label'].tolist(),
        tokenizer,
        max_len=args.max_len,
    )

    use_balanced = not args.no_balanced_sampler
    if use_balanced:
        print("\n使用长度+标签联合平衡采样器")
        sampler = build_length_balanced_sampler(train_df)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size,
            sampler=sampler, num_workers=0
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=0
        )

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0
    )

    # ── Optimizer & Scheduler ──
    loss_fn = LengthAwareLoss(length_penalty_weight=args.length_penalty)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01
    )

    total_steps = len(train_loader) * args.epochs // args.accum_steps
    warmup_steps = total_steps // 10
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ──
    print("\n" + "=" * 70)
    print("  开始训练")
    print("=" * 70)

    best_f1 = 0
    patience_counter = 0
    results = []

    for epoch in range(args.epochs):
        t0 = time.time()
        print(f"\n{'─' * 70}")
        print(f"  Epoch {epoch + 1}/{args.epochs}")
        print(f"{'─' * 70}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            device, loss_fn, args.accum_steps
        )

        val_loss, val_acc, val_prec, val_recall, val_f1 = evaluate(
            model, val_loader, device
        )

        elapsed = time.time() - t0

        print(f"\n  Train  — Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val    — Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        print(f"           P: {val_prec:.4f}, R: {val_recall:.4f}, F1: {val_f1:.4f}")
        print(f"  Time: {elapsed:.1f}s")

        results.append({
            'epoch': epoch + 1,
            'train_loss': round(train_loss, 5),
            'train_acc': round(train_acc, 5),
            'val_loss': round(val_loss, 5),
            'val_acc': round(val_acc, 5),
            'val_precision': round(val_prec, 5),
            'val_recall': round(val_recall, 5),
            'val_f1': round(val_f1, 5),
            'time_sec': round(elapsed, 1),
        })

        # Best model by F1
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            model.save_pretrained(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))
            print(f"  >>> 保存最佳模型 (F1={val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  F1 未提升 ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print(f"\n  Early stopping at epoch {epoch + 1}")
                break

        torch.cuda.empty_cache()
        gc.collect()

    # ── 保存训练日志 ──
    log = {
        'version': 'v8',
        'data': str(data_dir),
        'base_model': model_path,
        'params': {
            'batch_size': args.batch_size,
            'effective_batch': effective_batch,
            'max_len': args.max_len,
            'epochs_planned': args.epochs,
            'epochs_ran': len(results),
            'lr': args.lr,
            'length_penalty': args.length_penalty,
            'balanced_sampler': use_balanced,
            'patience': args.patience,
        },
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'best_val_f1': round(best_f1, 5),
        'results': results,
    }
    log_path = output_dir / 'training_log.json'
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"  训练完成! Best Val F1: {best_f1:.4f}")
    print(f"  模型: {output_dir}")
    print(f"  日志: {log_path}")
    print("=" * 70)

    print("\n下一步:")
    print(f"  1. 运行评估:")
    print(f"     python scripts/evaluation/eval_complete.py --model {output_dir}")
    print(f"  2. 技术文档专项测试:")
    print(f"     python scripts/evaluation/test_single_text.py --interactive --model {output_dir}")


if __name__ == '__main__':
    main()
