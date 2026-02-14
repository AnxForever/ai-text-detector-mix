#!/usr/bin/env python3
"""
训练脚本 - 使用合并数据集 (core_v1 + paired_v2)
保守资源设置
"""
import os
import time
import json
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import gc

# 离线模式
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# === 保守参数 ===
BATCH_SIZE = 8
MAX_LEN = 256
EPOCHS = 3
LR = 2e-5
ACCUMULATION_STEPS = 4


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
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def train_epoch(model, loader, optimizer, scheduler, device, accumulation_steps=4):
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

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss / accumulation_steps
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

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            total_loss += outputs.loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total


def main():
    print("=" * 60)
    print("BERT Training - Merged Dataset (core_v1 + paired_v2)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\n训练参数:")
    print(f"  batch_size: {BATCH_SIZE}")
    print(f"  max_length: {MAX_LEN}")
    print(f"  epochs: {EPOCHS}")
    print(f"  effective_batch_size: {BATCH_SIZE * ACCUMULATION_STEPS}")

    # 加载模型
    print("\n>>> 加载 bert-base-chinese")
    model_path = os.path.expanduser(
        '~/.cache/huggingface/hub/models--bert-base-chinese/'
        'snapshots/8f23c25b06e129b6c986331a13d8d025a92cf0ea'
    )
    if not os.path.exists(model_path):
        model_path = 'bert-base-chinese'

    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(
        model_path,
        num_labels=2
    ).to(device)

    # 加载数据
    print("\n>>> 加载 merged_v1 数据集")
    train_df = pd.read_csv('datasets/merged_v1/train.csv')
    val_df = pd.read_csv('datasets/merged_v1/val.csv')

    train_df = train_df.dropna(subset=['text', 'label'])
    val_df = val_df.dropna(subset=['text', 'label'])
    train_df['label'] = train_df['label'].astype(int)
    val_df['label'] = val_df['label'].astype(int)

    print(f"  Train: {len(train_df)} samples")
    print(f"  Val: {len(val_df)} samples")
    print(f"  Train labels: {train_df['label'].value_counts().to_dict()}")

    train_dataset = TextDataset(
        train_df['text'].tolist(),
        train_df['label'].tolist(),
        tokenizer,
        max_len=MAX_LEN
    )
    val_dataset = TextDataset(
        val_df['text'].tolist(),
        val_df['label'].tolist(),
        tokenizer,
        max_len=MAX_LEN
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS // ACCUMULATION_STEPS
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-7)

    output_dir = 'models/bert_v6_merged'
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)

    best_acc = 0
    results = []

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"{'='*60}")

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device, ACCUMULATION_STEPS
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
            print(f"  >>> New best! Saving model...")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

        torch.cuda.empty_cache()
        gc.collect()

    with open(f'{output_dir}/training_log.json', 'w') as f:
        json.dump({
            'params': {
                'batch_size': BATCH_SIZE,
                'max_len': MAX_LEN,
                'epochs': EPOCHS,
                'lr': LR,
                'train_samples': len(train_df),
                'val_samples': len(val_df)
            },
            'results': results,
            'best_val_acc': best_acc
        }, f, indent=2)

    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"Best Val Acc: {best_acc:.4f}")
    print(f"Model saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
