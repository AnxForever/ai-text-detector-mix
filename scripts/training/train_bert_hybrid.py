#!/usr/bin/env python3
"""在混合数据集上训练BERT"""
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import json

class TextDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len=512):
        df = pd.read_csv(csv_path)
        self.texts = df['text'].tolist()
        self.labels = df['label'].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self): return len(self.texts)
    
    def __getitem__(self, idx):
        enc = self.tokenizer(str(self.texts[idx])[:self.max_len], truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        return {k: v.squeeze(0) for k, v in enc.items()}, self.labels[idx]

def train():
    device = torch.device('cuda')
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2).to(device)
    
    train_ds = TextDataset('datasets/hybrid/train.csv', tokenizer)
    val_ds = TextDataset('datasets/hybrid/val.csv', tokenizer)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=len(train_loader)*3)
    
    best_acc = 0
    for epoch in range(3):
        model.train()
        total_loss = 0
        for batch, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/3'):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = torch.tensor(labels).to(device)
            
            outputs = model(**batch, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        # 验证
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch, labels in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                preds = model(**batch).logits.argmax(dim=1).cpu()
                correct += (preds == torch.tensor(labels)).sum().item()
                total += len(labels)
        
        acc = correct / total
        print(f'Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, val_acc={acc:.4f}')
        
        if acc > best_acc:
            best_acc = acc
            model.save_pretrained('models/bert_hybrid')
            tokenizer.save_pretrained('models/bert_hybrid')
    
    print(f'Best val acc: {best_acc:.4f}')
    
    # 测试
    test_ds = TextDataset('datasets/hybrid/test.csv', tokenizer)
    test_loader = DataLoader(test_ds, batch_size=32)
    model = BertForSequenceClassification.from_pretrained('models/bert_hybrid').to(device).eval()
    
    correct, total = 0, 0
    with torch.no_grad():
        for batch, labels in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            preds = model(**batch).logits.argmax(dim=1).cpu()
            correct += (preds == torch.tensor(labels)).sum().item()
            total += len(labels)
    
    print(f'Test acc: {correct/total:.4f}')
    json.dump({'test_acc': correct/total, 'best_val_acc': best_acc}, open('models/bert_hybrid/results.json', 'w'))

if __name__ == '__main__':
    train()
