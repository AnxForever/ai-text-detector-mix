#!/usr/bin/env python3
"""
A/B 对比实验 - 从 bert-base-chinese 全新初始化训练
对比 bert_v3_core_v2 (继承 with_sep) 来确定问题根因
"""
import os
import time
import json
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import gc

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'


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


def train_epoch(model, loader, optimizer, device, accumulation_steps=4):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for i, batch in enumerate(tqdm(loader, desc="Training")):
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
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

        if (i + 1) % 500 == 0:
            torch.cuda.empty_cache()

    if len(loader) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    accuracy = correct / total
    human_correct = sum(1 for p, l in zip(all_preds, all_labels) if l == 0 and p == 0)
    human_total = sum(1 for l in all_labels if l == 0)
    ai_correct = sum(1 for p, l in zip(all_preds, all_labels) if l == 1 and p == 1)
    ai_total = sum(1 for l in all_labels if l == 1)

    human_acc = human_correct / human_total if human_total > 0 else 0
    ai_acc = ai_correct / ai_total if ai_total > 0 else 0

    return accuracy, human_acc, ai_acc


def main():
    start_time = time.time()
    print("=" * 60)
    print("A/B 对比实验 - bert-base-chinese 全新初始化")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 参数 - 与 bert_v3_core_v2 保持一致
    BATCH_SIZE = 4
    MAX_LEN = 256
    EPOCHS = 5
    ACCUMULATION_STEPS = 4
    LR = 2e-5

    print(f"\nParameters (same as V3):")
    print(f"  batch_size: {BATCH_SIZE}")
    print(f"  accumulation_steps: {ACCUMULATION_STEPS}")
    print(f"  effective_batch: {BATCH_SIZE * ACCUMULATION_STEPS}")
    print(f"  max_length: {MAX_LEN}")
    print(f"  epochs: {EPOCHS}")
    print(f"  learning_rate: {LR}")

    # 关键区别：从 bert-base-chinese 全新初始化
    print("\n>>> 从 bert-base-chinese 全新初始化 (不继承任何微调偏差)")
    # 使用本地缓存路径（避免网络问题）
    model_path = os.path.expanduser(
        '~/.cache/huggingface/hub/models--bert-base-chinese/'
        'snapshots/8f23c25b06e129b6c986331a13d8d025a92cf0ea'
    )
    if not os.path.exists(model_path):
        model_path = 'bert-base-chinese'
    print(f"  Base model: {model_path}")

    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(
        model_path,
        num_labels=2
    ).to(device)

    # 加载相同的 core_v2 数据
    print("\nLoading core_v2 dataset (same as V3)...")
    data_dir = 'datasets/active/core_v2'
    train_df = pd.read_csv(f'{data_dir}/train.csv')
    val_df = pd.read_csv(f'{data_dir}/val.csv')

    train_df = train_df.dropna(subset=['text', 'label'])
    val_df = val_df.dropna(subset=['text', 'label'])
    train_df['label'] = train_df['label'].astype(int)
    val_df['label'] = val_df['label'].astype(int)

    print(f"  Train: {len(train_df)} samples")
    print(f"  Val: {len(val_df)} samples")

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
    warmup_steps = total_steps // 10
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_steps
    )

    output_dir = 'models/bert_v3_fresh'
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("Starting training (fresh initialization)...")
    print("=" * 60)

    best_acc = 0
    results = []

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        print(f"\n{'='*60}")
        print(f"=== Epoch {epoch+1}/{EPOCHS} ===")
        print(f"{'='*60}")

        train_loss = train_epoch(
            model, train_loader, optimizer, device, ACCUMULATION_STEPS
        )
        scheduler.step()

        val_acc, human_acc, ai_acc = evaluate(model, val_loader, device)
        epoch_time = time.time() - epoch_start

        result = {
            'epoch': epoch + 1,
            'loss': train_loss,
            'val_acc': val_acc,
            'human_acc': human_acc,
            'ai_acc': ai_acc,
            'time_min': epoch_time / 60
        }
        results.append(result)

        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Loss: {train_loss:.4f}")
        print(f"  Overall Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
        print(f"  Human Accuracy: {human_acc:.4f} ({human_acc*100:.2f}%)")
        print(f"  AI Accuracy: {ai_acc:.4f} ({ai_acc*100:.2f}%)")
        print(f"  Time: {epoch_time/60:.1f} min")

        if val_acc > best_acc:
            best_acc = val_acc
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"  >>> Best model saved! acc={val_acc:.4f}")

        gc.collect()
        torch.cuda.empty_cache()

    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nTotal time: {total_time/60:.1f} min")
    print(f"Best validation accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
    print(f"Model saved to: {output_dir}")

    # 保存训练日志
    log_path = os.path.join(output_dir, 'training_log.json')
    log_data = {
        'model_name': 'bert_v3_fresh',
        'base_model': 'bert-base-chinese',
        'experiment': 'A/B comparison - fresh initialization',
        'dataset': 'datasets/active/core_v2',
        'dataset_size': len(train_df),
        'val_size': len(val_df),
        'params': {
            'batch_size': BATCH_SIZE,
            'accumulation_steps': ACCUMULATION_STEPS,
            'max_length': MAX_LEN,
            'epochs': EPOCHS,
            'learning_rate': LR
        },
        'best_accuracy': best_acc,
        'results': results,
        'total_time_seconds': total_time
    }
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    print(f"\nTraining log saved: {log_path}")


if __name__ == '__main__':
    main()
