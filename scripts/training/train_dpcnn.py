#!/usr/bin/env python3
"""DPCNN模型训练 - 深度金字塔卷积神经网络"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
import pandas as pd
from tqdm import tqdm
import json
from pathlib import Path

class DPCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, num_filters=250, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.region_embed = nn.Sequential(
            nn.Conv1d(embed_dim, num_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(num_filters, num_classes)
        
    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)  # (B, E, L)
        x = self.region_embed(x)
        
        # 重复卷积+池化直到长度为1
        while x.size(2) > 2:
            x = x + self.conv(x)  # 残差连接
            x = self.pool(x)
        
        x = x.max(dim=2)[0]  # 全局最大池化
        return self.fc(x)

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.encodings = tokenizer(texts, truncation=True, max_length=max_len, padding='max_length')
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'labels': torch.tensor(self.labels[idx])
        }

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    train_df = pd.read_csv('datasets/final_clean/train.csv')
    val_df = pd.read_csv('datasets/final_clean/val.csv')
    test_df = pd.read_csv('datasets/final_clean/test.csv')
    
    print(f"训练集: {len(train_df)}, 验证集: {len(val_df)}, 测试集: {len(test_df)}")
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    train_dataset = TextDataset(train_df['text'].tolist(), train_df['label'].tolist(), tokenizer)
    val_dataset = TextDataset(val_df['text'].tolist(), val_df['label'].tolist(), tokenizer)
    test_dataset = TextDataset(test_df['text'].tolist(), test_df['label'].tolist(), tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    model = DPCNN(vocab_size=tokenizer.vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    output_dir = Path('models/dpcnn')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_acc = 0
    
    for epoch in range(10):
        model.train()
        train_correct, train_total = 0, 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/10')
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{train_correct/train_total:.4f}'})
        
        # 验证
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                logits = model(input_ids)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        print(f'Epoch {epoch+1}: 训练={train_correct/train_total:.4f}, 验证={val_acc:.4f}')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
    
    # 测试
    model.load_state_dict(torch.load(output_dir / 'best_model.pt'))
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            logits = model(input_ids)
            preds = torch.argmax(logits, dim=1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)
    
    test_acc = test_correct / test_total
    print(f'\n测试准确率: {test_acc:.4f}')
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump({'test_accuracy': test_acc, 'best_val_accuracy': best_val_acc}, f, indent=2)

if __name__ == '__main__':
    train()
