#!/usr/bin/env python3
"""图增强模型 - BERT + 文本图结构"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer
import pandas as pd
from tqdm import tqdm
import json
from pathlib import Path

class BertGraphModel(nn.Module):
    """BERT + 句子级图注意力"""
    def __init__(self, bert_name='bert-base-chinese', hidden_size=256):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        # 句子级特征提取
        self.sent_proj = nn.Linear(768, hidden_size)
        # 图注意力层
        self.attn_q = nn.Linear(hidden_size, hidden_size)
        self.attn_k = nn.Linear(hidden_size, hidden_size)
        self.attn_v = nn.Linear(hidden_size, hidden_size)
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + 768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
        
    def forward(self, input_ids, attention_mask):
        # BERT编码
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]  # [CLS]
        seq_output = outputs.last_hidden_state  # 全序列
        
        # 句子级特征 (简化：用滑动窗口模拟句子)
        B, L, D = seq_output.shape
        sent_feats = self.sent_proj(seq_output)  # (B, L, H)
        
        # 自注意力 (模拟图注意力)
        Q = self.attn_q(sent_feats)
        K = self.attn_k(sent_feats)
        V = self.attn_v(sent_feats)
        
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / (sent_feats.size(-1) ** 0.5)
        attn_scores = attn_scores.masked_fill(attention_mask.unsqueeze(1) == 0, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        graph_output = torch.matmul(attn_weights, V)
        graph_pooled = graph_output.mean(dim=1)  # 池化
        
        # 融合CLS和图特征
        combined = torch.cat([cls_output, graph_pooled], dim=-1)
        logits = self.classifier(combined)
        return logits

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
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
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    test_loader = DataLoader(test_dataset, batch_size=8)
    
    model = BertGraphModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    
    output_dir = Path('models/bert_graph')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_acc = 0
    
    for epoch in range(3):
        model.train()
        train_correct, train_total = 0, 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/3')
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
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
            for batch in tqdm(val_loader, desc='验证'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                logits = model(input_ids, attention_mask)
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
        for batch in tqdm(test_loader, desc='测试'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)
    
    test_acc = test_correct / test_total
    print(f'\n测试准确率: {test_acc:.4f}')
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump({'test_accuracy': test_acc, 'best_val_accuracy': best_val_acc}, f, indent=2)

if __name__ == '__main__':
    train()
