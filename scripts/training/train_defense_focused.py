#!/usr/bin/env python3
"""
答辩聚焦模型训练 - 在 bert_v3_core_v2 基础上用聚焦数据微调

数据：datasets/defense_focused/ (8169条，100-500字中等长度)
策略：低学习率 + 3 epochs + 双验证（聚焦测试集 + core_v2 验证集）
"""
import os
import sys
import time
import json
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import gc
from datetime import datetime

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

PROJECT_ROOT = Path(__file__).parent.parent.parent
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


def train_epoch(model, loader, optimizer, device, accumulation_steps=2):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for i, batch in enumerate(tqdm(loader, desc="Training")):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

    if len(loader) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
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

    return accuracy, human_correct / max(human_total, 1), ai_correct / max(ai_total, 1)


def evaluate_by_scenario(model, test_df, tokenizer, device, max_len=256):
    """按场景评估"""
    results = {}

    for scenario in test_df['scenario'].unique():
        subset = test_df[test_df['scenario'] == scenario]
        dataset = TextDataset(
            subset['text'].tolist(),
            subset['label'].astype(int).tolist(),
            tokenizer, max_len
        )
        loader = DataLoader(dataset, batch_size=16)
        acc, h_acc, a_acc = evaluate(model, loader, device)

        results[scenario] = {
            'accuracy': acc,
            'human_acc': h_acc,
            'ai_acc': a_acc,
            'count': len(subset),
            'human_count': int((subset['label'] == 0).sum()),
            'ai_count': int((subset['label'] == 1).sum()),
        }

    return results


def main():
    start_time = time.time()
    print("=" * 60)
    print("答辩聚焦模型训练")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 训练参数
    BATCH_SIZE = 16
    MAX_LEN = 256
    EPOCHS = 3
    ACCUMULATION_STEPS = 2
    LR = 2e-5

    print(f"\n参数:")
    print(f"  batch_size: {BATCH_SIZE} (effective: {BATCH_SIZE * ACCUMULATION_STEPS})")
    print(f"  max_len: {MAX_LEN}")
    print(f"  epochs: {EPOCHS}")
    print(f"  learning_rate: {LR}")

    # 加载模型
    from transformers import BertTokenizer, BertForSequenceClassification

    model_path = str(PROJECT_ROOT / 'models' / 'bert_v3_core_v2')
    if not os.path.exists(model_path):
        print(f"[ERROR] 模型不存在: {model_path}")
        sys.exit(1)

    print(f"\n加载基础模型: {model_path}")
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2).to(device)

    # 加载聚焦数据
    data_dir = PROJECT_ROOT / 'datasets' / 'defense_focused'
    train_df = pd.read_csv(data_dir / 'train.csv', encoding='utf-8-sig')
    val_df = pd.read_csv(data_dir / 'val.csv', encoding='utf-8-sig')
    test_df = pd.read_csv(data_dir / 'test.csv', encoding='utf-8-sig')

    # 加载 core_v2 验证集（回归测试）
    core_val_path = PROJECT_ROOT / 'datasets' / 'active' / 'core_v2' / 'val.csv'
    if not core_val_path.exists():
        core_val_path = PROJECT_ROOT / 'datasets' / 'active' / 'core_v1' / 'val.csv'
    core_val_df = pd.read_csv(core_val_path, encoding='utf-8-sig')
    core_val_df = core_val_df.dropna(subset=['text', 'label'])

    print(f"\n数据集:")
    print(f"  训练集: {len(train_df)} 条")
    print(f"  验证集: {len(val_df)} 条")
    print(f"  测试集: {len(test_df)} 条")
    print(f"  Core 回归验证: {len(core_val_df)} 条")

    # 创建 DataLoader
    train_dataset = TextDataset(
        train_df['text'].tolist(),
        train_df['label'].astype(int).tolist(),
        tokenizer, MAX_LEN
    )
    val_dataset = TextDataset(
        val_df['text'].tolist(),
        val_df['label'].astype(int).tolist(),
        tokenizer, MAX_LEN
    )
    core_val_dataset = TextDataset(
        core_val_df['text'].tolist(),
        core_val_df['label'].astype(int).tolist(),
        tokenizer, MAX_LEN
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0)
    core_val_loader = DataLoader(core_val_dataset, batch_size=BATCH_SIZE, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    # 训练前评估
    print("\n" + "=" * 60)
    print("训练前评估（基线）")
    print("=" * 60)

    print("\n[聚焦测试集 - 分场景]")
    pre_scenario = evaluate_by_scenario(model, test_df, tokenizer, device, MAX_LEN)
    for scenario, r in sorted(pre_scenario.items()):
        print(f"  {scenario:16s} | {r['accuracy']*100:5.1f}% (H:{r['human_acc']*100:.0f}% A:{r['ai_acc']*100:.0f}%) | {r['count']}条")

    pre_val_acc, _, _ = evaluate(model, val_loader, device)
    print(f"\n[聚焦验证集] 准确率: {pre_val_acc*100:.2f}%")

    pre_core_acc, pre_core_h, pre_core_a = evaluate(model, core_val_loader, device)
    print(f"[Core 回归验证] 准确率: {pre_core_acc*100:.2f}% (H:{pre_core_h*100:.2f}% A:{pre_core_a*100:.2f}%)")

    # 训练
    output_dir = PROJECT_ROOT / 'models' / 'bert_v4_defense_focused'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("开始训练...")
    print(f"{'='*60}")

    best_val_acc = 0
    training_log = []

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")

        train_loss = train_epoch(model, train_loader, optimizer, device, ACCUMULATION_STEPS)
        val_acc, val_h, val_a = evaluate(model, val_loader, device)
        core_acc, core_h, core_a = evaluate(model, core_val_loader, device)
        epoch_time = time.time() - epoch_start

        print(f"  Loss: {train_loss:.4f}")
        print(f"  聚焦验证: {val_acc*100:.2f}% (H:{val_h*100:.1f}% A:{val_a*100:.1f}%)")
        print(f"  Core回归: {core_acc*100:.2f}% (H:{core_h*100:.1f}% A:{core_a*100:.1f}%)")
        print(f"  耗时: {epoch_time:.1f}s")

        # 按场景评估
        epoch_scenario = evaluate_by_scenario(model, test_df, tokenizer, device, MAX_LEN)
        print(f"\n  [分场景测试]")
        for scenario, r in sorted(epoch_scenario.items()):
            pre_acc = pre_scenario[scenario]['accuracy'] * 100
            cur_acc = r['accuracy'] * 100
            change = cur_acc - pre_acc
            sign = "+" if change >= 0 else ""
            print(f"    {scenario:16s} | {cur_acc:5.1f}% ({sign}{change:.1f}%)")

        training_log.append({
            'epoch': epoch + 1,
            'loss': train_loss,
            'val_acc': val_acc,
            'core_acc': core_acc,
            'core_human_acc': core_h,
            'core_ai_acc': core_a,
            'scenario_detail': {k: v['accuracy'] for k, v in epoch_scenario.items()},
            'time': epoch_time,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))
            print(f"\n  ★ 保存最佳模型！Val Acc: {val_acc*100:.2f}%")

        gc.collect()
        torch.cuda.empty_cache()

    # 最终结果
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print("训练完成！")
    print(f"{'='*60}")
    print(f"总耗时: {total_time:.1f}s")
    print(f"模型保存至: {output_dir}")

    # 保存训练日志
    log = {
        'model_name': 'bert_v4_defense_focused',
        'base_model': 'bert_v3_core_v2',
        'timestamp': datetime.now().isoformat(),
        'params': {
            'lr': LR, 'epochs': EPOCHS, 'batch_size': BATCH_SIZE,
            'accumulation_steps': ACCUMULATION_STEPS, 'max_len': MAX_LEN,
        },
        'data': {
            'train': len(train_df), 'val': len(val_df), 'test': len(test_df),
            'core_val': len(core_val_df),
        },
        'baseline': {
            'val_acc': pre_val_acc,
            'core_acc': pre_core_acc,
            'scenario': {k: v['accuracy'] for k, v in pre_scenario.items()},
        },
        'final': {
            'val_acc': val_acc,
            'core_acc': core_acc,
            'scenario': {k: v['accuracy'] for k, v in epoch_scenario.items()},
        },
        'training_log': training_log,
    }

    with open(output_dir / 'training_log.json', 'w', encoding='utf-8') as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    # 最终对比表
    print(f"\n{'='*60}")
    print("效果对比")
    print(f"{'='*60}")
    print(f"\n{'场景':<18} | {'训练前':>6} | {'训练后':>6} | {'变化':>6}")
    print("-" * 50)
    for scenario in sorted(pre_scenario.keys()):
        pre_s = pre_scenario[scenario]['accuracy'] * 100
        post_s = epoch_scenario[scenario]['accuracy'] * 100
        change = post_s - pre_s
        sign = "+" if change >= 0 else ""
        print(f"{scenario:<18} | {pre_s:5.1f}% | {post_s:5.1f}% | {sign}{change:.1f}%")
    print("-" * 50)
    print(f"{'聚焦验证集':<18} | {pre_val_acc*100:5.2f}% | {val_acc*100:5.2f}% | {(val_acc-pre_val_acc)*100:+.2f}%")
    print(f"{'Core回归':<18} | {pre_core_acc*100:5.2f}% | {core_acc*100:5.2f}% | {(core_acc-pre_core_acc)*100:+.2f}%")


if __name__ == '__main__':
    main()
