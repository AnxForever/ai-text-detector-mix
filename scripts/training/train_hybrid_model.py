#!/usr/bin/env python3
"""
混合特征模型训练脚本
架构：BERT(768维) + 统计特征(10维) -> 融合层(128维) -> Dropout -> 分类层(2类)
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pickle
from tqdm import tqdm

class HybridModel(nn.Module):
    def __init__(self, bert_model_name='bert-base-chinese', dropout_rate=0.1):
        super(HybridModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.fusion_layer = nn.Linear(768 + 10, 128)  # BERT(768) + 统计特征(10)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(128, 2)
        
    def forward(self, input_ids, attention_mask, statistical_features):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_pooled = bert_output.pooler_output  # [batch_size, 768]
        
        # 融合BERT特征和统计特征
        combined_features = torch.cat([bert_pooled, statistical_features], dim=1)  # [batch_size, 778]
        
        # 通过融合层
        fused = torch.relu(self.fusion_layer(combined_features))  # [batch_size, 128]
        fused = self.dropout(fused)
        
        # 分类
        logits = self.classifier(fused)  # [batch_size, 2]
        return logits

class HybridDataset(Dataset):
    def __init__(self, texts, labels, statistical_features, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.statistical_features = statistical_features
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        stat_features = self.statistical_features[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'statistical_features': torch.FloatTensor(stat_features),
            'labels': torch.LongTensor([label])
        }

def extract_statistical_features(text):
    """提取10维统计特征"""
    if not isinstance(text, str):
        text = str(text)
    
    # 基础统计
    char_count = len(text)
    word_count = len(text.split())
    sentence_count = text.count('。') + text.count('！') + text.count('？') + 1
    
    # 标点符号统计
    punct_count = sum(1 for c in text if c in '，。！？；：""''（）【】')
    punct_ratio = punct_count / max(char_count, 1)
    
    # 平均长度
    avg_word_len = char_count / max(word_count, 1)
    avg_sent_len = char_count / max(sentence_count, 1)
    
    # 复杂度指标
    unique_chars = len(set(text))
    char_diversity = unique_chars / max(char_count, 1)
    
    # 特殊字符
    digit_ratio = sum(1 for c in text if c.isdigit()) / max(char_count, 1)
    english_ratio = sum(1 for c in text if c.isalpha() and ord(c) < 128) / max(char_count, 1)
    
    return np.array([
        char_count, word_count, sentence_count, punct_ratio, avg_word_len,
        avg_sent_len, char_diversity, digit_ratio, english_ratio, punct_count
    ], dtype=np.float32)

def load_data(data_path):
    """加载数据并提取统计特征"""
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    # 提取统计特征
    print("Extracting statistical features...")
    statistical_features = []
    for text in tqdm(texts, desc="Processing texts"):
        features = extract_statistical_features(text)
        statistical_features.append(features)
    
    statistical_features = np.array(statistical_features)
    
    return texts, labels, statistical_features

def train_model(model, train_loader, val_loader, device, epochs=3, lr=1e-5):
    """训练模型"""
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # 训练
        model.train()
        total_loss = 0
        train_preds, train_labels = [], []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            statistical_features = batch['statistical_features'].to(device)
            labels = batch['labels'].to(device).flatten()
            
            optimizer.zero_grad()
            
            logits = model(input_ids, attention_mask, statistical_features)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        train_acc = accuracy_score(train_labels, train_preds)
        avg_loss = total_loss / len(train_loader)
        
        # 验证
        model.eval()
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                statistical_features = batch['statistical_features'].to(device)
                labels = batch['labels'].to(device).flatten()
                
                logits = model(input_ids, attention_mask, statistical_features)
                preds = torch.argmax(logits, dim=1)
                
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(val_labels, val_preds)
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Acc: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"  New best validation accuracy: {best_val_acc:.4f}")
    
    return best_model_state, best_val_acc

def main():
    parser = argparse.ArgumentParser(description='Train Hybrid Model (BERT + Statistical Features)')
    parser.add_argument('--train_data', default='datasets/bert_debiased/train.csv', help='Training data path')
    parser.add_argument('--val_data', default='datasets/bert_debiased/val.csv', help='Validation data path')
    parser.add_argument('--test_data', default='datasets/bert_debiased/test.csv', help='Test data path')
    parser.add_argument('--output_dir', default='models/hybrid_model', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=512, help='Max sequence length')
    parser.add_argument('--device', default='auto', help='Device (auto/cpu/cuda)')
    
    args = parser.parse_args()
    
    # 设备设置
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据
    train_texts, train_labels, train_stat_features = load_data(args.train_data)
    val_texts, val_labels, val_stat_features = load_data(args.val_data)
    
    # 标准化统计特征
    print("Standardizing statistical features...")
    scaler = StandardScaler()
    train_stat_features = scaler.fit_transform(train_stat_features)
    val_stat_features = scaler.transform(val_stat_features)
    
    # 保存scaler
    with open(os.path.join(args.output_dir, 'feature_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    # 初始化tokenizer和模型
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = HybridModel().to(device)
    
    # 创建数据集
    train_dataset = HybridDataset(train_texts, train_labels, train_stat_features, tokenizer, args.max_length)
    val_dataset = HybridDataset(val_texts, val_labels, val_stat_features, tokenizer, args.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 训练模型
    print("Starting training...")
    best_model_state, best_val_acc = train_model(
        model, train_loader, val_loader, device, args.epochs, args.lr
    )
    
    # 保存最佳模型
    model.load_state_dict(best_model_state)
    
    best_model_dir = os.path.join(args.output_dir, 'best_model')
    os.makedirs(best_model_dir, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(best_model_dir, 'pytorch_model.bin'))
    tokenizer.save_pretrained(best_model_dir)
    
    # 保存配置
    config = {
        'model_type': 'hybrid',
        'bert_model': 'bert-base-chinese',
        'max_length': args.max_length,
        'best_val_acc': best_val_acc,
        'statistical_features_dim': 10,
        'fusion_dim': 128
    }
    
    with open(os.path.join(best_model_dir, 'config.json'), 'w', encoding='utf-8') as f:
        import json
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    # 测试集评估
    if os.path.exists(args.test_data):
        print("\nEvaluating on test set...")
        test_texts, test_labels, test_stat_features = load_data(args.test_data)
        test_stat_features = scaler.transform(test_stat_features)
        
        test_dataset = HybridDataset(test_texts, test_labels, test_stat_features, tokenizer, args.max_length)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        model.eval()
        test_preds, test_labels_list = [], []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                statistical_features = batch['statistical_features'].to(device)
                labels = batch['labels'].to(device).flatten()
                
                logits = model(input_ids, attention_mask, statistical_features)
                preds = torch.argmax(logits, dim=1)
                
                test_preds.extend(preds.cpu().numpy())
                test_labels_list.extend(labels.cpu().numpy())
        
        test_acc = accuracy_score(test_labels_list, test_preds)
        print(f"\nTest Results:")
        print(f"Accuracy: {test_acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(test_labels_list, test_preds, target_names=['Human', 'AI']))
        
        # 保存测试结果
        results = {
            'test_accuracy': test_acc,
            'validation_accuracy': best_val_acc,
            'model_architecture': 'BERT(768) + Statistical(10) -> Fusion(128) -> Classifier(2)'
        }
        
        with open(os.path.join(best_model_dir, 'test_results.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {best_model_dir}")

if __name__ == "__main__":
    main()