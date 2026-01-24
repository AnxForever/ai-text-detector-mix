#!/usr/bin/env python3
"""
多任务学习模型
Task 1: AI vs Human 二分类
Task 2: AI模型归属分类（GPT-4, Claude, Gemini等）
"""

import torch
import torch.nn as nn
from transformers import BertModel

class MultiTaskDetectionModel(nn.Module):
    """多任务AI文本检测模型"""
    
    def __init__(self, 
                 bert_model_name='bert-base-chinese',
                 num_sources=5,  # AI模型数量
                 stat_feature_dim=10,
                 hidden_dim=256,
                 dropout=0.3):
        super().__init__()
        
        # 共享BERT编码器
        self.bert = BertModel.from_pretrained(bert_model_name)
        bert_dim = self.bert.config.hidden_size
        
        # 统计特征处理
        self.stat_fc = nn.Linear(stat_feature_dim, 32)
        
        fusion_dim = bert_dim + 32
        
        # Task 1: AI vs Human 分类头
        self.detection_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )
        
        # Task 2: AI模型归属分类头
        self.attribution_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_sources)
        )
    
    def forward(self, input_ids, attention_mask, stat_features):
        # BERT编码
        bert_output = self.bert(input_ids, attention_mask)
        semantic = bert_output.pooler_output
        
        # 统计特征
        stat_embed = self.stat_fc(stat_features)
        
        # 融合
        fused = torch.cat([semantic, stat_embed], dim=1)
        
        # 两个任务的输出
        detection_logits = self.detection_head(fused)
        attribution_logits = self.attribution_head(fused)
        
        return detection_logits, attribution_logits


class MultiTaskTrainer:
    """多任务训练器"""
    
    def __init__(self, model, device='cuda', task_weights=(1.0, 0.5)):
        self.model = model.to(device)
        self.device = device
        self.task_weights = task_weights  # (检测权重, 归属权重)
        
    def train_epoch(self, train_loader, optimizer):
        self.model.train()
        total_loss = 0
        
        criterion = nn.CrossEntropyLoss()
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            stat_features = batch['stat_features'].to(self.device)
            detection_labels = batch['detection_labels'].to(self.device)
            attribution_labels = batch['attribution_labels'].to(self.device)
            
            optimizer.zero_grad()
            
            detection_logits, attribution_logits = self.model(
                input_ids, attention_mask, stat_features
            )
            
            # 计算两个任务的损失
            loss_detection = criterion(detection_logits, detection_labels)
            loss_attribution = criterion(attribution_logits, attribution_labels)
            
            # 加权组合
            loss = (self.task_weights[0] * loss_detection + 
                   self.task_weights[1] * loss_attribution)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)


def main():
    print("多任务学习模型测试")
    
    model = MultiTaskDetectionModel(num_sources=5)
    
    # 模拟输入
    batch_size = 4
    input_ids = torch.randint(0, 21128, (batch_size, 128))
    attention_mask = torch.ones(batch_size, 128)
    stat_features = torch.randn(batch_size, 10)
    
    detection_logits, attribution_logits = model(input_ids, attention_mask, stat_features)
    
    print(f"✓ 多任务模型测试成功")
    print(f"  检测输出: {detection_logits.shape} (AI vs Human)")
    print(f"  归属输出: {attribution_logits.shape} (模型来源)")


if __name__ == "__main__":
    main()
