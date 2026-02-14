#!/usr/bin/env python3
"""
训练模型：在merged_v1基础上加入AI技术教程数据
解决技术教程被误判为Human的问题
"""
import os
import sys
import json
import torch
import pandas as pd
from pathlib import Path
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 配置
MODEL_NAME = "bert-base-chinese"
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
GRADIENT_ACCUMULATION = 2

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def load_and_merge_data():
    """加载并合并数据：merged_v1 + AI技术教程"""
    print("=" * 60)
    print("加载数据")
    print("=" * 60)

    # 加载merged_v1
    train_path = PROJECT_ROOT / "datasets" / "merged_v1" / "train.csv"
    val_path = PROJECT_ROOT / "datasets" / "merged_v1" / "val.csv"

    train_df = pd.read_csv(train_path, encoding='utf-8-sig')
    val_df = pd.read_csv(val_path, encoding='utf-8-sig')

    print(f"原始训练集: {len(train_df)} 条")
    print(f"原始验证集: {len(val_df)} 条")

    # 加载AI技术教程
    tutorials_path = PROJECT_ROOT / "datasets" / "generated_tutorials" / "ai_tech_tutorials.csv"
    tutorials_df = pd.read_csv(tutorials_path, encoding='utf-8-sig')

    # 统一列名
    tutorials_df = tutorials_df[['text', 'label', 'source']]

    print(f"AI技术教程: {len(tutorials_df)} 条")

    # 合并到训练集
    train_df = pd.concat([train_df, tutorials_df], ignore_index=True)

    # 打乱
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n合并后训练集: {len(train_df)} 条")
    print(f"  - Human: {len(train_df[train_df['label'] == 0])}")
    print(f"  - AI: {len(train_df[train_df['label'] == 1])}")

    return train_df, val_df


def train_epoch(model, dataloader, optimizer, scheduler, device, accumulation_steps):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    optimizer.zero_grad()

    progress = tqdm(dataloader, desc="Training")
    for step, batch in enumerate(progress):
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

        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += outputs.loss.item()

        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        progress.set_postfix({'loss': f'{outputs.loss.item():.4f}'})

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy


def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
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
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy, all_preds, all_labels


def test_tutorial_detection(model, tokenizer, device):
    """测试技术教程检测"""
    print("\n" + "=" * 60)
    print("技术教程检测测试")
    print("=" * 60)

    test_text = """梯度爆炸（gradient explosion）是指在训练神经网络时，反向传播得到的梯度在某些层或某些时间步变得非常大，导致参数更新幅度失控，训练不稳定甚至直接"发散"（loss 变成 NaN/Inf 或剧烈震荡）。常见表现：loss 突然飙升、出现 NaN；权重数值变得很大；训练过程极不稳定。为什么会发生：反向传播的梯度本质上是很多个雅可比矩阵/导数的连乘。若这些连乘中对应的"增益"长期大于 1，梯度会指数式增长。"""

    model.eval()
    encoding = tokenizer(
        test_text,
        truncation=True,
        max_length=MAX_LEN,
        padding='max_length',
        return_tensors='pt'
    )

    with torch.no_grad():
        outputs = model(
            input_ids=encoding['input_ids'].to(device),
            attention_mask=encoding['attention_mask'].to(device)
        )
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    label_map = {0: 'Human', 1: 'AI'}
    result = label_map[pred]

    print(f"测试文本: 梯度爆炸技术教程")
    print(f"预测结果: {result} ({confidence*100:.1f}%)")
    print(f"期望结果: AI")
    print(f"判断: {'✓ 正确' if pred == 1 else '✗ 错误'}")

    return pred == 1


def main():
    print("=" * 60)
    print("bert_v7_with_tutorials 训练")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # 加载数据
    train_df, val_df = load_and_merge_data()

    # 加载tokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # 创建数据集
    train_dataset = TextDataset(
        train_df['text'].tolist(),
        train_df['label'].tolist(),
        tokenizer,
        MAX_LEN
    )
    val_dataset = TextDataset(
        val_df['text'].tolist(),
        val_df['label'].tolist(),
        tokenizer,
        MAX_LEN
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, num_workers=2)

    # 加载模型
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model = model.to(device)

    # 优化器和调度器
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS // GRADIENT_ACCUMULATION
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    # 训练
    best_val_acc = 0
    output_dir = PROJECT_ROOT / "models" / "bert_v7_with_tutorials"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_log = {
        'config': {
            'model': MODEL_NAME,
            'max_len': MAX_LEN,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'lr': LR,
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'tutorials_added': 29,
        },
        'history': []
    }

    for epoch in range(EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print('='*60)

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device, GRADIENT_ACCUMULATION
        )
        val_loss, val_acc, _, _ = evaluate(model, val_loader, device)

        print(f"\n训练 - Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}%")
        print(f"验证 - Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%")

        training_log['history'].append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"保存最佳模型 (val_acc: {val_acc*100:.2f}%)")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    # 测试技术教程检测
    tutorial_correct = test_tutorial_detection(model, tokenizer, device)
    training_log['tutorial_test_passed'] = tutorial_correct
    training_log['best_val_acc'] = best_val_acc

    # 保存日志
    log_path = output_dir / "training_log.json"
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(training_log, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("训练完成!")
    print(f"最佳验证准确率: {best_val_acc*100:.2f}%")
    print(f"模型保存至: {output_dir}")
    print(f"技术教程测试: {'通过 ✓' if tutorial_correct else '失败 ✗'}")
    print('='*60)

    return 0 if tutorial_correct else 1


if __name__ == "__main__":
    sys.exit(main())
