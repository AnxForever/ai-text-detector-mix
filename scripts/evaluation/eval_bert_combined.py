#!/usr/bin/env python3
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys

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
            self.texts[idx],
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

def evaluate(model, dataloader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pred = torch.argmax(outputs.logits, dim=1)
            
            preds.extend(pred.cpu().numpy())
            labels.extend(label.cpu().numpy())
    
    return preds, labels

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model_path = 'models/bert_combined'
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path).to(device)
    
    # Test on overall test set
    print("\n=== Overall Test Set ===")
    test_df = pd.read_csv('datasets/combined/test.csv')
    test_dataset = TextDataset(test_df['text'].tolist(), test_df['label'].tolist(), tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    preds, labels = evaluate(model, test_loader, device)
    acc = accuracy_score(labels, preds)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=['Human', 'AI'], digits=4))
    print("\nConfusion Matrix:")
    print(confusion_matrix(labels, preds))
    
    # Test on hybrid-only test set
    print("\n=== Hybrid-Only Test Set ===")
    hybrid_df = pd.read_csv('datasets/combined/test_hybrid_only.csv')
    
    # Per-category accuracy
    for category in ['C2', 'C3', 'C4', 'Human']:
        cat_df = hybrid_df[hybrid_df['category'] == category]
        if len(cat_df) == 0:
            continue
        
        cat_dataset = TextDataset(cat_df['text'].tolist(), cat_df['label'].tolist(), tokenizer)
        cat_loader = DataLoader(cat_dataset, batch_size=16, shuffle=False)
        
        preds, labels = evaluate(model, cat_loader, device)
        acc = accuracy_score(labels, preds)
        print(f"\n{category}: {len(cat_df)} samples, Accuracy: {acc:.4f}")
        
        if category != 'Human':
            # For AI categories, show how many were correctly identified as AI
            correct = sum([p == l for p, l in zip(preds, labels)])
            print(f"  Correctly identified as AI: {correct}/{len(labels)}")

if __name__ == '__main__':
    main()
