#!/usr/bin/env python3
"""
修复 Formal Human 误判问题的微调训练

策略：
1. 加载现有 bert_v3_core_v2 模型
2. 使用收集的 formal human 数据 + 平衡的 AI 数据进行微调
3. 验证修复效果，同时确保整体准确率不下降

用法：
    python scripts/training/train_formal_fix.py
"""
import os
import sys
import json
import random
import torch
import pandas as pd
from pathlib import Path
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

PROJECT_ROOT = Path(__file__).parent.parent.parent
random.seed(42)
torch.manual_seed(42)


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
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def load_formal_human_data():
    """加载 formal human 数据"""
    data_file = PROJECT_ROOT / 'datasets' / 'collected_formal_human' / 'all_formal_human_merged.json'

    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    texts = [item['text'] for item in data]
    labels = [item['label'] for item in data]  # 都是 0 (Human)
    categories = [item.get('category', 'unknown') for item in data]

    return texts, labels, categories


def load_ai_samples_from_core_v2(n_samples=200):
    """从 core_v2 加载 AI 样本（用于平衡）"""
    train_file = PROJECT_ROOT / 'datasets' / 'active' / 'core_v2' / 'train.csv'

    df = pd.read_csv(train_file, encoding='utf-8-sig')
    ai_df = df[df['label'] == 1].sample(n=min(n_samples, len(df[df['label'] == 1])), random_state=42)

    texts = ai_df['text'].tolist()
    labels = ai_df['label'].tolist()

    return texts, labels


def load_human_samples_from_core_v2(n_samples=100):
    """从 core_v2 加载 Human 样本（保持平衡）"""
    train_file = PROJECT_ROOT / 'datasets' / 'active' / 'core_v2' / 'train.csv'

    df = pd.read_csv(train_file, encoding='utf-8-sig')
    human_df = df[df['label'] == 0].sample(n=min(n_samples, len(df[df['label'] == 0])), random_state=42)

    texts = human_df['text'].tolist()
    labels = human_df['label'].tolist()

    return texts, labels


def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total if total > 0 else 0


def evaluate_by_category(model, tokenizer, device):
    """按类别评估 formal human 数据"""
    data_file = PROJECT_ROOT / 'datasets' / 'collected_formal_human' / 'all_formal_human_merged.json'

    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = {}
    model.eval()

    for item in data:
        text = item['text']
        category = item.get('category', 'unknown')
        true_label = item.get('label', 0)

        encoding = tokenizer(text, max_length=256, truncation=True, padding='max_length', return_tensors='pt')
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pred = torch.argmax(outputs.logits, dim=1).item()

        if category not in results:
            results[category] = {'correct': 0, 'total': 0}

        results[category]['total'] += 1
        if pred == true_label:
            results[category]['correct'] += 1

    return results


def main():
    print("=" * 60)
    print("Formal Human 修复微调训练")
    print("=" * 60)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 配置
    BASE_MODEL = 'bert_v3_core_v2'
    OUTPUT_MODEL = 'bert_v4_formal_fix'

    EPOCHS = 5
    BATCH_SIZE = 8
    MAX_LEN = 256
    LR = 1e-5  # 较小的学习率，避免破坏原有知识

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # 加载模型
    model_path = str(PROJECT_ROOT / 'models' / BASE_MODEL)
    print(f"\n加载基础模型: {model_path}")

    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
    model = model.to(device)

    # 加载数据
    print("\n准备训练数据...")

    # Formal Human 数据
    formal_texts, formal_labels, formal_cats = load_formal_human_data()
    print(f"  Formal Human: {len(formal_texts)} 条")

    # 复制 formal human 数据多次增强权重
    REPEAT_FORMAL = 5  # 复制5次增强权重
    formal_texts_repeated = formal_texts * REPEAT_FORMAL
    formal_labels_repeated = formal_labels * REPEAT_FORMAL
    print(f"  Formal Human (复制后): {len(formal_texts_repeated)} 条")

    # AI 样本平衡
    ai_texts, ai_labels = load_ai_samples_from_core_v2(n_samples=len(formal_texts_repeated))
    print(f"  AI 样本: {len(ai_texts)} 条")

    # 再加入一些 core_v2 的 Human 样本保持分布
    core_human_texts, core_human_labels = load_human_samples_from_core_v2(n_samples=100)
    print(f"  Core Human 样本: {len(core_human_texts)} 条")

    # 合并数据
    all_texts = formal_texts_repeated + ai_texts + core_human_texts
    all_labels = formal_labels_repeated + ai_labels + core_human_labels

    # 打乱
    combined = list(zip(all_texts, all_labels))
    random.shuffle(combined)
    all_texts, all_labels = zip(*combined)
    all_texts, all_labels = list(all_texts), list(all_labels)

    print(f"\n训练集总计: {len(all_texts)} 条")
    print(f"  Human: {sum(1 for l in all_labels if l == 0)} 条")
    print(f"  AI: {sum(1 for l in all_labels if l == 1)} 条")

    # 创建数据集
    train_dataset = TextDataset(all_texts, all_labels, tokenizer, MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 训练前评估
    print("\n训练前 Formal Human 识别率:")
    before_results = evaluate_by_category(model, tokenizer, device)
    for cat in sorted(before_results.keys()):
        res = before_results[cat]
        acc = res['correct'] / res['total'] * 100
        print(f"  {cat}: {res['correct']}/{res['total']} = {acc:.1f}%")

    # 优化器和调度器
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )

    # 训练
    print(f"\n开始训练 ({EPOCHS} epochs)...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in progress:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: 平均 loss = {avg_loss:.4f}")

        # 每轮评估
        epoch_results = evaluate_by_category(model, tokenizer, device)
        total_correct = sum(r['correct'] for r in epoch_results.values())
        total_count = sum(r['total'] for r in epoch_results.values())
        print(f"  Formal Human 识别率: {total_correct}/{total_count} = {total_correct/total_count*100:.1f}%")

    # 训练后评估
    print("\n" + "=" * 60)
    print("训练后 Formal Human 识别率:")
    after_results = evaluate_by_category(model, tokenizer, device)
    for cat in sorted(after_results.keys()):
        res = after_results[cat]
        acc = res['correct'] / res['total'] * 100
        before_acc = before_results[cat]['correct'] / before_results[cat]['total'] * 100
        diff = acc - before_acc
        status = '✅' if acc > 80 else '⚠️'
        print(f"  {cat}: {res['correct']}/{res['total']} = {acc:.1f}% ({diff:+.1f}%) {status}")

    # 保存模型
    output_path = PROJECT_ROOT / 'models' / OUTPUT_MODEL
    output_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print(f"\n模型已保存: {output_path}")

    # 保存训练日志
    log = {
        'base_model': BASE_MODEL,
        'output_model': OUTPUT_MODEL,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'max_len': MAX_LEN,
            'lr': LR,
            'formal_repeat': REPEAT_FORMAL,
        },
        'training_data': {
            'formal_human': len(formal_texts),
            'formal_human_repeated': len(formal_texts_repeated),
            'ai_samples': len(ai_texts),
            'core_human': len(core_human_texts),
            'total': len(all_texts),
        },
        'before_results': {k: {'accuracy': v['correct']/v['total']} for k, v in before_results.items()},
        'after_results': {k: {'accuracy': v['correct']/v['total']} for k, v in after_results.items()},
    }

    with open(output_path / 'training_log.json', 'w', encoding='utf-8') as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"\n下一步: 验证整体准确率未下降")
    print(f"命令: python scripts/evaluation/eval_complete.py --model {OUTPUT_MODEL}")


if __name__ == '__main__':
    main()
