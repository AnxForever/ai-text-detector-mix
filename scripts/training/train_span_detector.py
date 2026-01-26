#!/usr/bin/env python3
"""
训练C2 span检测模型 (Token-level分类)
"""
import torch
import json
from transformers import BertTokenizer, BertForTokenClassification
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split

class SpanDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            item['text'],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Prepare labels
        labels = item['token_labels'][:self.max_len]
        # Pad labels
        labels = labels + [-100] * (self.max_len - len(labels))
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(loader, desc="Training"):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            
            # 只计算非-100的标签
            mask = labels != -100
            correct += ((preds == labels) & mask).sum().item()
            total += mask.sum().item()
    
    return correct / total if total > 0 else 0

def calculate_boundary_accuracy(model, loader, tokenizer, device, tolerance=5):
    """
    计算边界定位准确率
    tolerance: 允许的字符偏差
    """
    model.eval()
    correct_boundaries = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Boundary eval"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1).cpu()
            
            for i in range(len(preds)):
                # 找到真实边界
                true_labels = labels[i]
                pred_labels = preds[i]
                
                # 找到标签从0变到1的位置
                true_boundary = None
                pred_boundary = None
                
                for j in range(1, len(true_labels)):
                    if true_labels[j-1] == 0 and true_labels[j] == 1:
                        true_boundary = j
                        break
                
                for j in range(1, len(pred_labels)):
                    if pred_labels[j-1] == 0 and pred_labels[j] == 1:
                        pred_boundary = j
                        break
                
                if true_boundary and pred_boundary:
                    # 计算token位置差异（简化为直接比较）
                    if abs(true_boundary - pred_boundary) <= tolerance:
                        correct_boundaries += 1
                
                total_samples += 1
    
    return correct_boundaries / total_samples if total_samples > 0 else 0

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load data
    with open('datasets/hybrid/c2_span_labels.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total samples: {len(data)}")
    
    # Split
    train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}\n")
    
    # Load model (从bert_combined初始化)
    tokenizer = BertTokenizer.from_pretrained('models/bert_combined')
    model = BertForTokenClassification.from_pretrained(
        'models/bert_combined',
        num_labels=2,  # 0=Human, 1=AI
        ignore_mismatched_sizes=True
    ).to(device)
    
    # Datasets
    train_dataset = SpanDataset(train_data, tokenizer)
    val_dataset = SpanDataset(val_data, tokenizer)
    test_dataset = SpanDataset(test_data, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    # Training
    best_acc = 0
    for epoch in range(3):
        print(f"\n=== Epoch {epoch+1}/3 ===")
        
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        
        print(f"Loss: {train_loss:.4f}, Val Token Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            model.save_pretrained('models/bert_span_detector')
            tokenizer.save_pretrained('models/bert_span_detector')
            print(f"  ✓ Best model saved! acc={val_acc:.4f}")
    
    # Final evaluation
    print(f"\n=== Final Evaluation ===")
    test_acc = evaluate(model, test_loader, device)
    print(f"Test Token Accuracy: {test_acc:.4f}")
    
    boundary_acc = calculate_boundary_accuracy(model, test_loader, tokenizer, device, tolerance=5)
    print(f"Boundary Accuracy (±5 tokens): {boundary_acc:.4f}")
    
    print(f"\nTraining complete! Best val acc: {best_acc:.4f}")

if __name__ == '__main__':
    main()
