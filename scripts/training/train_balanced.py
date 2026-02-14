#!/usr/bin/env python3
"""
平衡数据集训练脚本
使用长度平衡的 72K 数据集训练 BERT 分类器
消除旧模型中的长度偏差问题
"""
import os
import time
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


def train_epoch(model, loader, optimizer, device, accumulation_steps=8):
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

        if (i + 1) % 200 == 0:
            torch.cuda.empty_cache()

    # 处理剩余梯度
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

    # 分类别统计
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
    print("平衡数据集训练 - 消除长度偏差")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"显存: {total_mem:.1f} GB")

    # 参数 - 保守设置防止 OOM
    BATCH_SIZE = 2
    MAX_LEN = 256
    EPOCHS = 3
    ACCUMULATION_STEPS = 8  # 等效 batch_size = 16
    LR = 2e-5

    print(f"\n参数配置:")
    print(f"  batch_size: {BATCH_SIZE}")
    print(f"  accumulation_steps: {ACCUMULATION_STEPS}")
    print(f"  effective_batch: {BATCH_SIZE * ACCUMULATION_STEPS}")
    print(f"  max_length: {MAX_LEN}")
    print(f"  epochs: {EPOCHS}")
    print(f"  learning_rate: {LR}")

    # 加载基础模型 (从原始 bert_v2_with_sep 开始微调)
    print("\n加载模型...")
    model_path = 'models/bert_v2_with_sep'
    if not os.path.exists(model_path):
        model_path = 'bert-base-chinese'
        print(f"  [WARN] 使用预训练模型: {model_path}")
    else:
        print(f"  使用本地模型: {model_path}")

    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(
        model_path,
        num_labels=2
    ).to(device)

    # 加载平衡数据集
    print("\n加载平衡数据集...")
    data_dir = 'datasets/bert_v2_overnight'
    train_df = pd.read_csv(f'{data_dir}/train_balanced.csv')
    val_df = pd.read_csv(f'{data_dir}/val_balanced.csv')

    # 过滤无效数据
    train_df = train_df.dropna(subset=['text', 'label'])
    val_df = val_df.dropna(subset=['text', 'label'])
    train_df['label'] = train_df['label'].astype(int)
    val_df['label'] = val_df['label'].astype(int)

    print(f"  训练集: {len(train_df)} 条")
    print(f"  验证集: {len(val_df)} 条")
    print(f"  标签分布 (train): {train_df['label'].value_counts().to_dict()}")

    # 数据集长度统计
    train_df['text_len'] = train_df['text'].astype(str).apply(len)
    for label_name, label_val in [('Human', 0), ('AI', 1)]:
        subset = train_df[train_df['label'] == label_val]
        print(f"  {label_name} 平均长度: {subset['text_len'].mean():.0f} chars")

    # 来源分布
    if 'source' in train_df.columns:
        print(f"\n  来源分布:")
        for src, cnt in train_df['source'].value_counts().items():
            print(f"    {src}: {cnt}")

    # 创建数据集
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

    # 学习率调度器 - 线性衰减
    total_steps = len(train_loader) * EPOCHS // ACCUMULATION_STEPS
    warmup_steps = total_steps // 10
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_steps
    )

    # 输出目录
    output_dir = 'models/bert_v2_balanced'
    os.makedirs(output_dir, exist_ok=True)

    # 训练
    print("\n" + "=" * 60)
    print("开始训练...")
    est_steps = len(train_loader) * EPOCHS
    print(f"预计总步数: {est_steps}")
    print("=" * 60)

    best_acc = 0
    results = []

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        print(f"\n{'='*60}")
        print(f"=== Epoch {epoch+1}/{EPOCHS} ===")
        print(f"{'='*60}")

        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated(0) / 1024**3
            print(f"GPU 显存使用: {used:.2f} GB")

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

        print(f"\nEpoch {epoch+1} 结果:")
        print(f"  Loss: {train_loss:.4f}")
        print(f"  总准确率: {val_acc:.4f} ({val_acc*100:.2f}%)")
        print(f"  Human 准确率: {human_acc:.4f} ({human_acc*100:.2f}%)")
        print(f"  AI 准确率: {ai_acc:.4f} ({ai_acc*100:.2f}%)")
        print(f"  耗时: {epoch_time/60:.1f} 分钟")
        print(f"  当前学习率: {optimizer.param_groups[0]['lr']:.2e}")

        if val_acc > best_acc:
            best_acc = val_acc
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"  >>> 最佳模型已保存! acc={val_acc:.4f}")
        else:
            print(f"  --- 未超过最佳 ({best_acc:.4f})")

        gc.collect()
        torch.cuda.empty_cache()

    total_time = time.time() - start_time

    # 训练总结
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"\n总耗时: {total_time/60:.1f} 分钟 ({total_time/3600:.1f} 小时)")
    print(f"最佳验证准确率: {best_acc:.4f} ({best_acc*100:.2f}%)")
    print(f"模型保存位置: {output_dir}")

    print("\n各 Epoch 对比:")
    print(f"{'Epoch':>5} | {'Loss':>8} | {'Val Acc':>8} | {'Human':>8} | {'AI':>8} | {'Time':>8}")
    print("-" * 56)
    for r in results:
        print(f"{r['epoch']:>5} | {r['loss']:>8.4f} | {r['val_acc']:>7.4f} | "
              f"{r['human_acc']:>7.4f} | {r['ai_acc']:>7.4f} | {r['time_min']:>6.1f}m")

    # 保存训练日志
    import json
    log_path = os.path.join(output_dir, 'training_log.json')
    log_data = {
        'model_name': 'bert_v2_balanced',
        'base_model': model_path,
        'dataset': 'bert_v2_overnight/train_balanced.csv',
        'dataset_size': len(train_df),
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
    print(f"\n训练日志已保存: {log_path}")


if __name__ == '__main__':
    main()
