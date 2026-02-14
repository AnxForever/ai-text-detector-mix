#!/usr/bin/env python3
"""
改进的BERT训练脚本 - 解决长度偏差问题

改进点:
1. 数据清洗: 移除过短文本和重复
2. 长度平衡采样: 确保各长度区间平衡
3. 长度加权损失: 减轻长度偏差影响
4. 对比学习预热 (可选): 借鉴DeTeCtive方法
5. RoBERTa支持 (可选): 比BERT更强

参考:
- DeTeCtive (NeurIPS 2024): Multi-Level Contrastive Learning
- https://github.com/heyongxin233/DeTeCtive
"""
import os
import time
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import gc
import argparse
from collections import Counter

# 离线模式
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'


class ImprovedTextDataset(Dataset):
    """改进的数据集类，支持长度信息"""

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
            'length': torch.tensor(self.lengths[idx], dtype=torch.float)
        }


class LengthAwareLoss(nn.Module):
    """长度感知的损失函数，减轻长度偏差，支持 Label Smoothing"""

    def __init__(self, base_loss='ce', length_penalty_weight=0.1, label_smoothing=0.0):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none', label_smoothing=label_smoothing)
        self.length_penalty_weight = length_penalty_weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, labels, lengths, mean_length=500):
        # 基础交叉熵损失（含 label smoothing）
        ce = self.ce_loss(logits, labels)

        # 长度惩罚: 对长度极端的样本降低权重
        # 避免模型学习"长文本=AI"的捷径
        length_ratio = lengths / mean_length
        # 长度权重: 接近平均长度的样本权重更高
        length_weight = 1.0 / (1.0 + self.length_penalty_weight * torch.abs(torch.log(length_ratio + 1e-6)))

        weighted_loss = ce * length_weight
        return weighted_loss.mean()


def clean_data(df, min_length=50, max_length=5000):
    """清洗数据"""
    original_len = len(df)

    # 1. 移除空值
    df = df.dropna(subset=['text', 'label'])

    # 2. 计算长度
    df['length'] = df['text'].str.len()

    # 3. 过滤极端长度
    df = df[(df['length'] >= min_length) & (df['length'] <= max_length)]

    # 4. 移除重复
    df = df.drop_duplicates(subset=['text'], keep='first')

    # 5. 移除AI模板短语
    ai_phrases = ["作为一个AI", "作为AI", "AI语言模型", "As an AI"]
    for phrase in ai_phrases:
        df = df[~df['text'].str.contains(phrase, na=False)]

    print(f"  数据清洗: {original_len} -> {len(df)} (移除 {original_len - len(df)})")

    return df


def get_length_balanced_sampler(df):
    """创建长度平衡采样器"""
    # 按长度分桶
    bins = [0, 100, 200, 500, 1000, 2000, float('inf')]
    df['length_bin'] = pd.cut(df['length'], bins=bins, labels=False)

    # 计算每个桶的权重
    bin_counts = df['length_bin'].value_counts()
    bin_weights = 1.0 / bin_counts

    # 为每个样本分配权重
    sample_weights = df['length_bin'].map(bin_weights).values

    # 同时考虑标签平衡
    label_counts = df['label'].value_counts()
    label_weights = 1.0 / label_counts
    sample_weights *= df['label'].map(label_weights).values

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(df),
        replacement=True
    )


def train_epoch(model, loader, optimizer, scheduler, device, loss_fn, accumulation_steps=4):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc="Training")
    for i, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        lengths = batch['length'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # 使用长度感知损失
        loss = loss_fn(outputs.logits, labels, lengths) / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

        preds = torch.argmax(outputs.logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if (i + 1) % 500 == 0:
            torch.cuda.empty_cache()
            gc.collect()

        pbar.set_postfix({'loss': f'{total_loss/(i+1):.4f}', 'acc': f'{correct/total:.4f}'})

    return total_loss / len(loader), correct / total


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_lengths = []

    ce_loss = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            lengths = batch['length']

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            loss = ce_loss(outputs.logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_lengths.extend(lengths.numpy())

    # 按长度区间分析准确率
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_lengths = np.array(all_lengths)

    print("\n  按长度区间的准确率:")
    bins = [(0, 200, "短"), (200, 500, "中"), (500, 1000, "长"), (1000, 10000, "超长")]
    for low, high, name in bins:
        mask = (all_lengths >= low) & (all_lengths < high)
        if mask.sum() > 0:
            acc = (all_preds[mask] == all_labels[mask]).mean()
            print(f"    {name} ({low}-{high}): {acc:.4f} ({mask.sum()} 样本)")

    return total_loss / len(loader), correct / total


def _has_model_weights(model_dir: str) -> bool:
    """妫€鏌ョ洰褰曚腑鏄惁鏈夋ā鍨嬫潈閲?"""
    try:
        path = Path(model_dir)
        if not path.exists():
            return False
        weight_files = [
            "pytorch_model.bin",
            "model.safetensors",
            "tf_model.h5",
            "model.ckpt.index",
            "flax_model.msgpack"
        ]
        return any((path / name).exists() for name in weight_files)
    except Exception:
        return False


def _resolve_model_path(preferred_path: str) -> str:
    """浼樺厛浣跨敤鏈湴鍙敤妯″瀷锛屽け璐ュ啀閫€鍥?"""
    candidates = []
    if preferred_path:
        candidates.append(preferred_path)

    cache_path = os.path.expanduser(
        "~/.cache/huggingface/hub/models--bert-base-chinese/"
        "snapshots/8f23c25b06e129b6c986331a13d8d025a92cf0ea"
    )
    candidates.append(cache_path)
    candidates.append("models/bert_v6_merged")

    for path in candidates:
        if _has_model_weights(path):
            return path

    raise FileNotFoundError(
        "No local model weights found. Please download bert-base-chinese or "
        "provide --model-path to a local model directory."
    )


def main():
    parser = argparse.ArgumentParser(description='改进的BERT训练')
    parser.add_argument('--data', default='datasets/merged_v2', help='数据目录')
    parser.add_argument('--output', default='models/bert_v7_improved', help='输出目录')
    parser.add_argument('--model-path', default='bert-base-chinese', help='基础模型路径')
    parser.add_argument('--batch-size', type=int, default=8, help='批次大小')
    parser.add_argument('--max-len', type=int, default=256, help='最大序列长度')
    parser.add_argument('--epochs', type=int, default=3, help='训练轮数')
    parser.add_argument('--lr', type=float, default=2e-5, help='学习率')
    parser.add_argument('--length-penalty', type=float, default=0.1, help='长度惩罚权重')
    parser.add_argument('--use-balanced-sampler', action='store_true', help='使用长度平衡采样')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label Smoothing 系数 (推荐 0.05)')
    parser.add_argument('--early-stopping', type=int, default=0, help='早停 patience (0=不启用)')
    args = parser.parse_args()

    print("=" * 70)
    print("改进的BERT训练 - 解决长度偏差问题")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\n训练参数:")
    print(f"  data: {args.data}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  max_length: {args.max_len}")
    print(f"  epochs: {args.epochs}")
    print(f"  length_penalty: {args.length_penalty}")
    print(f"  balanced_sampler: {args.use_balanced_sampler}")
    print(f"  label_smoothing: {args.label_smoothing}")
    print(f"  early_stopping: {args.early_stopping}")
    print(f"  model_path: {args.model_path}")

    # 加载模型
    print("\n>>> 加载模型")
    try:
        model_path = _resolve_model_path(args.model_path)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        return

    print(f"  使用模型: {model_path}")
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(
        model_path,
        num_labels=2
    ).to(device)

    # 加载和清洗数据
    print(f"\n>>> 加载数据: {args.data}")
    train_df = pd.read_csv(f'{args.data}/train.csv', encoding='utf-8-sig')
    val_df = pd.read_csv(f'{args.data}/val.csv', encoding='utf-8-sig')

    print("\n清洗训练数据:")
    train_df = clean_data(train_df)
    print("清洗验证数据:")
    val_df = clean_data(val_df)

    train_df['label'] = train_df['label'].astype(int)
    val_df['label'] = val_df['label'].astype(int)

    print(f"\n最终数据:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val: {len(val_df)} samples")
    print(f"  Train labels: {train_df['label'].value_counts().to_dict()}")

    # 长度统计
    for label, name in [(0, "Human"), (1, "AI")]:
        subset = train_df[train_df['label'] == label]['length']
        print(f"  {name} 长度: 均值={subset.mean():.0f}, 中位数={subset.median():.0f}")

    # 创建数据集
    train_dataset = ImprovedTextDataset(
        train_df['text'].tolist(),
        train_df['label'].tolist(),
        tokenizer,
        max_len=args.max_len
    )
    val_dataset = ImprovedTextDataset(
        val_df['text'].tolist(),
        val_df['label'].tolist(),
        tokenizer,
        max_len=args.max_len
    )

    # 数据加载器
    if args.use_balanced_sampler:
        print("\n使用长度平衡采样器")
        sampler = get_length_balanced_sampler(train_df)
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

    # 损失函数和优化器
    loss_fn = LengthAwareLoss(length_penalty_weight=args.length_penalty, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    accumulation_steps = 4
    total_steps = len(train_loader) * args.epochs // accumulation_steps
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=total_steps // 3, T_mult=2)

    os.makedirs(args.output, exist_ok=True)

    print("\n" + "=" * 70)
    print("开始训练...")
    print("=" * 70)

    best_acc = 0
    best_val_loss = float('inf')
    no_improve_count = 0
    results = []

    for epoch in range(args.epochs):
        epoch_start = time.time()
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*70}")

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler,
            device, loss_fn, accumulation_steps
        )

        val_loss, val_acc = evaluate(model, val_loader, device)

        epoch_time = time.time() - epoch_start

        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        print(f"  Time: {epoch_time:.1f}s")

        results.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'time': epoch_time
        })

        if val_acc > best_acc:
            best_acc = val_acc
            best_val_loss = val_loss
            no_improve_count = 0
            print(f"  >>> New best! Saving model...")
            model.save_pretrained(args.output)
            tokenizer.save_pretrained(args.output)
        else:
            no_improve_count += 1
            print(f"  No improvement ({no_improve_count}/{args.early_stopping or 'disabled'})")

        # Early stopping
        if args.early_stopping > 0 and no_improve_count >= args.early_stopping:
            print(f"\n  >>> Early stopping triggered after {epoch+1} epochs")
            break

        torch.cuda.empty_cache()
        gc.collect()

    # 保存训练日志
    with open(f'{args.output}/training_log.json', 'w') as f:
        json.dump({
            'params': vars(args),
            'results': results,
            'best_val_acc': best_acc,
            'improvements': [
                'Length-aware loss to reduce length bias',
                'Data cleaning (remove short/duplicate/template)',
                'Optional length-balanced sampling',
                f'Label Smoothing = {args.label_smoothing}' if args.label_smoothing > 0 else None,
                f'Early stopping patience = {args.early_stopping}' if args.early_stopping > 0 else None,
            ]
        }, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("训练完成!")
    print(f"Best Val Acc: {best_acc:.4f}")
    print(f"Model saved to: {args.output}")
    print("=" * 70)


if __name__ == '__main__':
    main()
