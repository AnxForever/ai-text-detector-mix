#!/usr/bin/env python3
"""
混合特征模型：BERT + 统计特征 + BiGRU
结合深度语义特征和人工统计特征
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import numpy as np

class HybridDetectionModel(nn.Module):
    """混合特征AI文本检测模型"""
    
    def __init__(self, 
                 bert_model_name='bert-base-chinese',
                 stat_feature_dim=10,
                 hidden_dim=256,
                 bigru_hidden=128,
                 dropout=0.3):
        super().__init__()
        
        # 1. BERT编码器
        self.bert = BertModel.from_pretrained(bert_model_name)
        bert_dim = self.bert.config.hidden_size  # 768
        
        # 2. BiGRU层（可选）
        self.use_bigru = True
        if self.use_bigru:
            self.bigru = nn.GRU(
                bert_dim, 
                bigru_hidden, 
                num_layers=1,
                batch_first=True,
                bidirectional=True,
                dropout=0
            )
            semantic_dim = bigru_hidden * 2  # 双向
        else:
            semantic_dim = bert_dim
        
        # 3. 统计特征处理
        self.stat_fc = nn.Sequential(
            nn.Linear(stat_feature_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 4. 特征融合分类器
        fusion_dim = semantic_dim + 32
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)  # 二分类
        )
    
    def forward(self, input_ids, attention_mask, stat_features):
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            stat_features: [batch, stat_dim]
        """
        # 1. BERT编码
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 2. BiGRU增强（可选）
        if self.use_bigru:
            sequence_output = bert_output.last_hidden_state  # [batch, seq, 768]
            gru_output, _ = self.bigru(sequence_output)  # [batch, seq, 256]
            semantic_features = gru_output[:, 0, :]  # 取第一个token
        else:
            semantic_features = bert_output.pooler_output  # [batch, 768]
        
        # 3. 统计特征处理
        stat_embed = self.stat_fc(stat_features)  # [batch, 32]
        
        # 4. 特征融合
        fused = torch.cat([semantic_features, stat_embed], dim=1)  # [batch, 288]
        
        # 5. 分类
        logits = self.classifier(fused)  # [batch, 2]
        
        return logits


class HybridModelTrainer:
    """混合模型训练器"""
    
    def __init__(self, model_name='bert-base-chinese', device='cuda'):
        self.device = device
        self.model = HybridDetectionModel(bert_model_name=model_name).to(device)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
    def train(self, train_loader, val_loader, epochs=3, lr=2e-5):
        """训练模型"""
        optimizer = torch.optim.AdamW([
            {'params': self.model.bert.parameters(), 'lr': lr},
            {'params': self.model.bigru.parameters(), 'lr': lr * 2},
            {'params': self.model.stat_fc.parameters(), 'lr': lr * 5},
            {'params': self.model.classifier.parameters(), 'lr': lr * 5}
        ], weight_decay=0.01)
        
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                stat_features = batch['stat_features'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                logits = self.model(input_ids, attention_mask, stat_features)
                loss = criterion(logits, labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            val_acc = self.evaluate(val_loader)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {avg_loss:.4f}")
            print(f"  Val Acc: {val_acc:.4f}")
    
    def evaluate(self, data_loader):
        """评估模型"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                stat_features = batch['stat_features'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits = self.model(input_ids, attention_mask, stat_features)
                preds = torch.argmax(logits, dim=1)
                
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        return correct / total


def main():
    """测试混合模型"""
    print("混合特征模型架构测试")
    
    model = HybridDetectionModel()
    
    # 模拟输入
    batch_size = 4
    seq_len = 128
    stat_dim = 10
    
    input_ids = torch.randint(0, 21128, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    stat_features = torch.randn(batch_size, stat_dim)
    
    # 前向传播
    logits = model(input_ids, attention_mask, stat_features)
    
    print(f"✓ 模型测试成功")
    print(f"  输入: input_ids {input_ids.shape}, stat_features {stat_features.shape}")
    print(f"  输出: logits {logits.shape}")
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    main()
