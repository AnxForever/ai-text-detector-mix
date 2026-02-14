#!/usr/bin/env python3
"""
稳妥过夜训练脚本 - 极简版
使用最保守的参数防止 OOM
"""
import os
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import gc

# 环境设置
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

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

        # 定期清理显存
        if (i + 1) % 100 == 0:
            torch.cuda.empty_cache()

    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

def main():
    print("=" * 50)
    print("稳妥过夜训练 - 开始")
    print("=" * 50)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 参数 - 最保守设置
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

    # 加载模型
    print("\n加载模型...")
    model_path = 'models/bert_v2_with_sep'
    if not os.path.exists(model_path):
        model_path = 'bert-base-chinese'
        print(f"  使用预训练模型: {model_path}")
    else:
        print(f"  使用本地模型: {model_path}")

    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(
        model_path,
        num_labels=2
    ).to(device)

    # 加载数据
    print("\n加载数据...")
    data_dir = 'datasets/bert_v2_overnight'
    train_df = pd.read_csv(f'{data_dir}/train.csv')
    val_df = pd.read_csv(f'{data_dir}/val.csv')

    print(f"  训练集: {len(train_df)} 条")
    print(f"  验证集: {len(val_df)} 条")
    print(f"  标签分布 (train): {train_df['label'].value_counts().to_dict()}")

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

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # 输出目录
    output_dir = 'models/bert_v2_overnight'
    os.makedirs(output_dir, exist_ok=True)

    # 训练
    print("\n" + "=" * 50)
    print("开始训练...")
    print("=" * 50)

    best_acc = 0
    for epoch in range(EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")

        # 显存状态
        if torch.cuda.is_available():
            print(f"GPU 显存使用: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

        train_loss = train_epoch(model, train_loader, optimizer, device, ACCUMULATION_STEPS)
        val_acc = evaluate(model, val_loader, device)

        print(f"Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"  ✓ 最佳模型已保存! acc={val_acc:.4f}")

        # 清理显存
        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "=" * 50)
    print(f"训练完成! 最佳准确率: {best_acc:.4f}")
    print(f"模型保存位置: {output_dir}")
    print("=" * 50)

if __name__ == '__main__':
    main()
