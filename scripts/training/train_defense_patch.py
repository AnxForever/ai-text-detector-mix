#!/usr/bin/env python3
"""
答辩补丁微调 - 在 bert_v3_core_v2 基础上用补丁数据微调 1-2 epoch
将补丁数据混入原训练集，低学习率微调，避免灾难性遗忘

文件结构：
  datasets/defense_patch/
  ├── scenarios/           # 训练数据
  │   ├── p0_casual_human.jsonl
  │   ├── p0_short_confirm.jsonl
  │   ├── p0_formal_notice.jsonl
  │   ├── p1_positive_review.jsonl
  │   └── p1_casual_ai.jsonl
  └── test_cases/
      └── defense_test.jsonl  # 测试集（永不训练）
"""
import os
import sys
import time
import json
import random
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

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
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
    all_preds, all_labels = [], []

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

    return accuracy, human_correct / max(human_total, 1), ai_correct / max(ai_total, 1)


def load_jsonl(filepath):
    """加载 JSONL 文件"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_defense_scenarios():
    """加载所有 defense_patch 场景数据"""
    scenario_dir = PROJECT_ROOT / 'datasets' / 'defense_patch' / 'scenarios'
    all_data = []

    for jsonl_file in sorted(scenario_dir.glob('*.jsonl')):
        data = load_jsonl(jsonl_file)
        all_data.extend(data)
        print(f"    {jsonl_file.name}: {len(data)} 条")

    return all_data


def load_defense_test():
    """加载 defense_test 测试集（按场景分组）"""
    test_file = PROJECT_ROOT / 'datasets' / 'defense_patch' / 'test_cases' / 'defense_test.jsonl'
    data = load_jsonl(test_file)

    by_scenario = {}
    for d in data:
        scenario = d.get('scenario', 'unknown')
        if scenario not in by_scenario:
            by_scenario[scenario] = []
        by_scenario[scenario].append(d)

    return by_scenario


def run_defense_test(model, tokenizer, device, max_len=128):
    """运行 defense_test.jsonl 测试"""
    defense_test = load_defense_test()

    model.eval()
    print("\n" + "=" * 70)
    print("Defense Test 分场景评估")
    print("=" * 70)

    results = {}
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for scenario, samples in sorted(defense_test.items()):
            correct = 0
            for sample in samples:
                text = sample['text']
                label = sample['label']

                enc = tokenizer(text, max_length=max_len, padding='max_length', truncation=True, return_tensors='pt')
                input_ids = enc['input_ids'].to(device)
                attention_mask = enc['attention_mask'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                pred = torch.argmax(outputs.logits, dim=1).item()

                if pred == label:
                    correct += 1

            acc = correct / len(samples) if samples else 0
            results[scenario] = {'correct': correct, 'total': len(samples), 'accuracy': acc}
            total_correct += correct
            total_count += len(samples)

            status = "✅" if acc >= 0.7 else "⚠️" if acc >= 0.5 else "❌"
            print(f"  {scenario:20s} | {correct:2d}/{len(samples):2d} ({acc*100:5.1f}%) {status}")

    overall_acc = total_correct / total_count if total_count > 0 else 0
    print("-" * 70)
    print(f"  {'总计':20s} | {total_correct:2d}/{total_count:2d} ({overall_acc*100:5.1f}%)")

    return results, overall_acc


def main():
    start_time = time.time()
    print("=" * 60)
    print("答辩补丁微调 - bert_v3_core_v2 + defense_patch")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Parameters - 低学习率避免灾难性遗忘
    BATCH_SIZE = 8
    MAX_LEN = 128  # 短文本优化
    EPOCHS = 2
    ACCUMULATION_STEPS = 2
    LR = 1e-5  # 低学习率

    print(f"\nParameters:")
    print(f"  batch_size: {BATCH_SIZE} (effective: {BATCH_SIZE * ACCUMULATION_STEPS})")
    print(f"  max_len: {MAX_LEN}")
    print(f"  epochs: {EPOCHS}")
    print(f"  learning_rate: {LR}")

    # Load model from bert_v3_core_v2
    from transformers import BertTokenizer, BertForSequenceClassification

    model_path = str(PROJECT_ROOT / 'models' / 'bert_v3_core_v2')
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        sys.exit(1)

    print(f"\nLoading model: {model_path}")
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2).to(device)

    # Run pre-finetune defense test
    print("\n--- 微调前 Defense Test ---")
    pre_results, pre_acc = run_defense_test(model, tokenizer, device, MAX_LEN)

    # Load defense_patch scenarios
    print("\n加载 defense_patch 场景数据...")
    patch_data = load_defense_scenarios()
    patch_df = pd.DataFrame(patch_data)[['text', 'label']]

    patch_human = sum(1 for d in patch_data if d['label'] == 0)
    patch_ai = sum(1 for d in patch_data if d['label'] == 1)
    print(f"  总计: {len(patch_df)} 条 (Human: {patch_human}, AI: {patch_ai})")

    # Load core training data (sample to balance)
    # 目标: 500 条训练数据, Human:AI ≈ 1:1
    TARGET_TOTAL = 500
    target_human = TARGET_TOTAL // 2
    target_ai = TARGET_TOTAL // 2
    need_human = max(0, target_human - patch_human)
    need_ai = max(0, target_ai - patch_ai)

    # 尝试 core_v2，不存在则 core_v1
    core_train_path = PROJECT_ROOT / 'datasets' / 'active' / 'core_v2' / 'train.csv'
    if not core_train_path.exists():
        core_train_path = PROJECT_ROOT / 'datasets' / 'active' / 'core_v1' / 'train.csv'
        print(f"\n使用 core_v1: {core_train_path}")
    else:
        print(f"\n使用 core_v2: {core_train_path}")

    core_df = pd.read_csv(core_train_path, encoding='utf-8-sig')
    core_human = core_df[core_df['label'] == 0].sample(n=min(need_human, len(core_df[core_df['label'] == 0])), random_state=42)
    core_ai = core_df[core_df['label'] == 1].sample(n=min(need_ai, len(core_df[core_df['label'] == 1])), random_state=42)
    core_sample = pd.concat([core_human, core_ai])[['text', 'label']]
    print(f"  Core 采样: {len(core_sample)} 条 (Human: {len(core_human)}, AI: {len(core_ai)})")

    # Merge training data
    combined_train = pd.concat([patch_df, core_sample], ignore_index=True)
    combined_train = combined_train.sample(frac=1, random_state=42).reset_index(drop=True)

    final_human = (combined_train['label'] == 0).sum()
    final_ai = (combined_train['label'] == 1).sum()
    print(f"\n训练集总计: {len(combined_train)} 条 (Human: {final_human}, AI: {final_ai})")

    # Load validation set
    core_val_path = PROJECT_ROOT / 'datasets' / 'active' / 'core_v2' / 'val.csv'
    if not core_val_path.exists():
        core_val_path = PROJECT_ROOT / 'datasets' / 'active' / 'core_v1' / 'val.csv'
    val_df = pd.read_csv(core_val_path, encoding='utf-8-sig')
    val_df = val_df.dropna(subset=['text', 'label'])
    print(f"验证集: {len(val_df)} 条")

    # Create datasets
    train_dataset = TextDataset(combined_train['text'].tolist(), combined_train['label'].astype(int).tolist(), tokenizer, MAX_LEN)
    val_dataset = TextDataset(val_df['text'].tolist(), val_df['label'].astype(int).tolist(), tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    # Output directory
    output_dir = PROJECT_ROOT / 'models' / 'bert_v3_defense'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training
    print(f"\n{'='*60}")
    print("开始训练...")
    print(f"{'='*60}")

    best_defense_acc = 0
    training_log = []

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")

        train_loss = train_epoch(model, train_loader, optimizer, device, ACCUMULATION_STEPS)
        val_acc, human_acc, ai_acc = evaluate(model, val_loader, device)
        epoch_time = time.time() - epoch_start

        print(f"  Loss: {train_loss:.4f}")
        print(f"  Val Accuracy: {val_acc*100:.2f}% (Human: {human_acc*100:.2f}%, AI: {ai_acc*100:.2f}%)")
        print(f"  Time: {epoch_time:.1f}s")

        # Defense test
        print(f"\n  [Epoch {epoch+1} Defense Test]")
        epoch_results, epoch_defense_acc = run_defense_test(model, tokenizer, device, MAX_LEN)

        training_log.append({
            'epoch': epoch + 1,
            'loss': train_loss,
            'val_acc': val_acc,
            'human_acc': human_acc,
            'ai_acc': ai_acc,
            'defense_acc': epoch_defense_acc,
            'defense_detail': {k: v['accuracy'] for k, v in epoch_results.items()}
        })

        if epoch_defense_acc > best_defense_acc:
            best_defense_acc = epoch_defense_acc
            model.save_pretrained(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))
            print(f"  ★ 新最佳模型！Defense Acc: {epoch_defense_acc*100:.1f}%")

        gc.collect()
        torch.cuda.empty_cache()

    # Final results
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print("训练完成！")
    print(f"{'='*60}")
    print(f"总耗时: {total_time:.1f}s")
    print(f"模型保存至: {output_dir}")

    # Save log
    log = {
        'model_name': 'bert_v3_defense',
        'base_model': 'bert_v3_core_v2',
        'timestamp': datetime.now().isoformat(),
        'params': {
            'lr': LR, 'epochs': EPOCHS, 'batch_size': BATCH_SIZE,
            'accumulation_steps': ACCUMULATION_STEPS, 'max_len': MAX_LEN
        },
        'data': {
            'patch_samples': len(patch_df),
            'core_samples': len(core_sample),
            'total_train': len(combined_train),
            'val_samples': len(val_df)
        },
        'pre_training': {
            'defense_acc': pre_acc,
            'defense_detail': {k: v['accuracy'] for k, v in pre_results.items()}
        },
        'post_training': {
            'defense_acc': epoch_defense_acc,
            'defense_detail': {k: v['accuracy'] for k, v in epoch_results.items()},
            'val_acc': val_acc
        },
        'training_log': training_log
    }

    with open(output_dir / 'training_log.json', 'w', encoding='utf-8') as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    # Print comparison
    print(f"\n{'='*60}")
    print("效果对比")
    print(f"{'='*60}")
    print(f"\n场景                 | 训练前 | 训练后 | 变化")
    print("-" * 55)
    for scenario in sorted(pre_results.keys()):
        pre_s = pre_results[scenario]['accuracy'] * 100
        post_s = epoch_results[scenario]['accuracy'] * 100
        change = post_s - pre_s
        sign = "+" if change >= 0 else ""
        print(f"{scenario:20s} | {pre_s:5.1f}% | {post_s:5.1f}% | {sign}{change:.1f}%")
    print("-" * 55)
    print(f"{'总计':20s} | {pre_acc*100:5.1f}% | {epoch_defense_acc*100:5.1f}% | {(epoch_defense_acc-pre_acc)*100:+.1f}%")


if __name__ == '__main__':
    main()
