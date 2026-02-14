#!/usr/bin/env python3
"""
Core V3 训练脚本 - 改进版

改进点：
1. Early Stopping (patience=3)
2. CosineAnnealingLR 替代 LinearLR（全程衰减）
3. 从 bert-base-chinese 全新初始化（避免继承偏差）
4. 更多训练轮数 (10 epochs)
5. 更小学习率 (1e-5)
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


def train_epoch(model, loader, optimizer, scheduler, device,
                accumulation_steps=4):
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
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

        if (i + 1) % 500 == 0:
            torch.cuda.empty_cache()

    if len(loader) % accumulation_steps != 0:
        optimizer.step()
        scheduler.step()
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
    print("Core V3 训练 - 改进策略")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM: {total_mem:.1f} GB")

    # === 改进参数 ===
    BATCH_SIZE = 4
    MAX_LEN = 256
    EPOCHS = 10              # 增加到10轮（Fresh需要更多轮）
    ACCUMULATION_STEPS = 4   # effective batch = 16
    LR = 1e-5                # 降低LR（避免Fresh模型早期过拟合）
    PATIENCE = 3             # Early Stopping patience
    WARMUP_RATIO = 0.1       # warmup占总步数的10%

    print(f"\n改进参数:")
    print(f"  batch_size: {BATCH_SIZE}")
    print(f"  accumulation_steps: {ACCUMULATION_STEPS}")
    print(f"  effective_batch: {BATCH_SIZE * ACCUMULATION_STEPS}")
    print(f"  max_length: {MAX_LEN}")
    print(f"  epochs: {EPOCHS} (增加)")
    print(f"  learning_rate: {LR} (降低)")
    print(f"  patience: {PATIENCE} (新增Early Stopping)")
    print(f"  scheduler: CosineAnnealingLR (替代LinearLR)")

    # === 从 bert-base-chinese 全新初始化 ===
    print("\n>>> 从 bert-base-chinese 全新初始化")
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

    # === 加载 core_v3 数据集 ===
    print("\nLoading core_v3 dataset...")
    data_dir = 'datasets/active/core_v3'
    if not os.path.exists(data_dir):
        print(f"  [WARN] {data_dir} not found, falling back to core_v2")
        data_dir = 'datasets/active/core_v2'

    train_df = pd.read_csv(f'{data_dir}/train.csv')
    val_df = pd.read_csv(f'{data_dir}/val.csv')

    train_df = train_df.dropna(subset=['text', 'label'])
    val_df = val_df.dropna(subset=['text', 'label'])
    train_df['label'] = train_df['label'].astype(int)
    val_df['label'] = val_df['label'].astype(int)

    print(f"  Train: {len(train_df)} samples")
    print(f"  Val: {len(val_df)} samples")
    print(f"  Labels: {train_df['label'].value_counts().to_dict()}")

    if 'source' in train_df.columns:
        human_df = train_df[train_df['label'] == 0]
        print(f"\n  Human来源分布:")
        for src, cnt in human_df['source'].value_counts().head(10).items():
            print(f"    {src}: {cnt}")

    # Create datasets
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

    # === 改进：CosineAnnealingLR（全程学习率衰减）===
    total_steps = len(train_loader) * EPOCHS // ACCUMULATION_STEPS
    print(f"\n  Total optimizer steps: {total_steps}")
    print(f"  Using CosineAnnealingLR (eta_min=1e-7)")

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=1e-7
    )

    output_dir = 'models/bert_v4_core_v3'
    os.makedirs(output_dir, exist_ok=True)

    # Training
    print("\n" + "=" * 60)
    print("Starting training...")
    print(f"Early Stopping: patience={PATIENCE}")
    print("=" * 60)

    best_acc = 0
    patience_counter = 0
    results = []

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        print(f"\n{'='*60}")
        print(f"=== Epoch {epoch+1}/{EPOCHS} ===")
        print(f"{'='*60}")

        current_lr = optimizer.param_groups[0]['lr']
        print(f"  LR: {current_lr:.2e}")

        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated(0) / 1024**3
            print(f"  GPU memory: {used:.2f} GB")

        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            ACCUMULATION_STEPS
        )

        val_acc, human_acc, ai_acc = evaluate(model, val_loader, device)
        epoch_time = time.time() - epoch_start

        result = {
            'epoch': epoch + 1,
            'loss': train_loss,
            'val_acc': val_acc,
            'human_acc': human_acc,
            'ai_acc': ai_acc,
            'lr': current_lr,
            'time_min': epoch_time / 60
        }
        results.append(result)

        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Loss: {train_loss:.4f}")
        print(f"  Overall Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
        print(f"  Human Accuracy: {human_acc:.4f} ({human_acc*100:.2f}%)")
        print(f"  AI Accuracy: {ai_acc:.4f} ({ai_acc*100:.2f}%)")
        print(f"  Time: {epoch_time/60:.1f} min")

        # === Early Stopping ===
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"  >>> Best model saved! acc={val_acc:.4f}")
        else:
            patience_counter += 1
            print(f"  --- Not better ({best_acc:.4f}), "
                  f"patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print(f"\n  [EARLY STOP] No improvement for "
                      f"{PATIENCE} epochs. Stopping.")
                break

        gc.collect()
        torch.cuda.empty_cache()

    total_time = time.time() - start_time

    # Summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nTotal time: {total_time/60:.1f} min ({total_time/3600:.1f} hours)")
    print(f"Best validation accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
    print(f"Stopped at epoch: {len(results)}/{EPOCHS}")
    print(f"Model saved to: {output_dir}")

    print("\nEpoch Comparison:")
    print(f"{'Epoch':>5} | {'Loss':>8} | {'Val Acc':>8} | "
          f"{'Human':>8} | {'AI':>8} | {'LR':>10} | {'Time':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['epoch']:>5} | {r['loss']:>8.4f} | {r['val_acc']:>7.4f} | "
              f"{r['human_acc']:>7.4f} | {r['ai_acc']:>7.4f} | "
              f"{r['lr']:>10.2e} | {r['time_min']:>6.1f}m")

    # Save training log
    log_path = os.path.join(output_dir, 'training_log.json')
    log_data = {
        'model_name': 'bert_v4_core_v3',
        'base_model': 'bert-base-chinese (fresh)',
        'dataset': data_dir,
        'dataset_size': len(train_df),
        'val_size': len(val_df),
        'params': {
            'batch_size': BATCH_SIZE,
            'accumulation_steps': ACCUMULATION_STEPS,
            'max_length': MAX_LEN,
            'max_epochs': EPOCHS,
            'actual_epochs': len(results),
            'learning_rate': LR,
            'scheduler': 'CosineAnnealingLR',
            'early_stopping_patience': PATIENCE,
        },
        'improvements': [
            'Early Stopping (patience=3)',
            'CosineAnnealingLR (eta_min=1e-7)',
            'Fresh initialization (no inherited bias)',
            'More epochs (10 max)',
            'Lower LR (1e-5)',
        ],
        'best_accuracy': best_acc,
        'results': results,
        'total_time_seconds': total_time
    }
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    print(f"\nTraining log saved: {log_path}")


if __name__ == '__main__':
    main()
