#!/usr/bin/env python3
"""综合评估脚本：ROC曲线、混淆矩阵、模型对比"""
import os, json, torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class TextDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512):
        self.texts = df['text'].tolist()
        self.labels = df['label'].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self): return len(self.texts)
    
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', 
                            max_length=self.max_len, return_tensors='pt')
        return {k: v.squeeze(0) for k, v in enc.items()}, self.labels[idx]

def evaluate_model(model_path, test_df, tokenizer, device, model_name):
    """评估单个模型"""
    model = BertForSequenceClassification.from_pretrained(model_path).to(device).eval()
    dataset = TextDataset(test_df, tokenizer)
    loader = DataLoader(dataset, batch_size=16)
    
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch, labels in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels)
    
    return np.array(all_probs), np.array(all_labels)

def main():
    os.makedirs('evaluation_results', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    test_df = pd.read_csv('datasets/final_clean/test.csv')
    
    # BERT V2使用标准格式
    models = {
        'BERT V2': 'models/bert_v2/best_model',
    }
    
    results = {}
    plt.figure(figsize=(10, 8))
    
    for name, path in models.items():
        if not os.path.exists(path): continue
        print(f"评估 {name}...")
        probs, labels = evaluate_model(path, test_df, tokenizer, device, name)
        
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        preds = (probs > 0.5).astype(int)
        
        results[name] = {
            'auc': roc_auc,
            'accuracy': (preds == labels).mean(),
            'confusion_matrix': confusion_matrix(labels, preds).tolist()
        }
        
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC={roc_auc:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - AI Text Detection Models')
    plt.legend(loc='lower right')
    plt.savefig('evaluation_results/roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    with open('evaluation_results/model_comparison.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n=== 评估完成 ===")
    for name, r in results.items():
        print(f"{name}: Acc={r['accuracy']:.4f}, AUC={r['auc']:.4f}")
    print(f"\n结果保存至 evaluation_results/")

if __name__ == '__main__':
    main()
