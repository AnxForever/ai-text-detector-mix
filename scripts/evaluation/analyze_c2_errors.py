#!/usr/bin/env python3
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import json

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
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
            'text': self.texts[idx]
        }

def analyze_c2_errors():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model_path = 'models/bert_v2_with_sep'
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()
    
    # Load C2 test data (from combined_v2)
    hybrid_df = pd.read_csv('datasets/combined_v2/test_hybrid_only.csv')
    c2_df = hybrid_df[hybrid_df['category'] == 'C2'].copy()
    
    print(f"Analyzing {len(c2_df)} C2 samples...")
    
    # Predict
    dataset = TextDataset(c2_df['text'].tolist(), c2_df['label'].tolist(), tokenizer)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    errors = []
    correct = []
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch['label'].item()
            text = batch['text'][0]
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=0)
            pred = torch.argmax(logits).item()
            
            result = {
                'text': text,
                'true_label': label,
                'pred_label': pred,
                'prob_human': probs[0].item(),
                'prob_ai': probs[1].item(),
                'confidence': probs[pred].item()
            }
            
            if pred != label:
                errors.append(result)
            else:
                correct.append(result)
    
    print(f"\nErrors: {len(errors)}/{len(c2_df)} ({len(errors)/len(c2_df)*100:.1f}%)")
    print(f"Correct: {len(correct)}/{len(c2_df)} ({len(correct)/len(c2_df)*100:.1f}%)")
    
    # Analyze errors
    print("\n=== Error Analysis ===")
    print(f"Average confidence on errors: {sum(e['confidence'] for e in errors)/len(errors):.4f}")
    print(f"Average confidence on correct: {sum(c['confidence'] for c in correct)/len(correct):.4f}")
    
    # Check if [SEP] marker exists
    errors_with_sep = [e for e in errors if '[SEP]' in e['text']]
    correct_with_sep = [c for c in correct if '[SEP]' in c['text']]
    
    print(f"\nWith [SEP] marker:")
    print(f"  Errors: {len(errors_with_sep)}/{len(errors)}")
    print(f"  Correct: {len(correct_with_sep)}/{len(correct)}")
    
    # Analyze text length
    error_lengths = [len(e['text']) for e in errors]
    correct_lengths = [len(c['text']) for c in correct]
    
    print(f"\nText length:")
    print(f"  Errors avg: {sum(error_lengths)/len(error_lengths):.0f} chars")
    print(f"  Correct avg: {sum(correct_lengths)/len(correct_lengths):.0f} chars")
    
    # Show examples
    print("\n=== Misclassified Examples (predicted as Human) ===")
    for i, err in enumerate(errors[:3]):
        print(f"\n[Example {i+1}] Confidence: {err['prob_human']:.4f}")
        print(f"Text preview: {err['text'][:200]}...")
        if '[SEP]' in err['text']:
            parts = err['text'].split('[SEP]')
            print(f"  Human part: {len(parts[0])} chars")
            print(f"  AI part: {len(parts[1]) if len(parts) > 1 else 0} chars")
    
    # Save detailed results
    with open('logs/c2_error_analysis.json', 'w', encoding='utf-8') as f:
        json.dump({
            'errors': errors,
            'correct': correct,
            'summary': {
                'total': len(c2_df),
                'errors': len(errors),
                'correct': len(correct),
                'error_rate': len(errors)/len(c2_df),
                'avg_confidence_errors': sum(e['confidence'] for e in errors)/len(errors) if errors else 0,
                'avg_confidence_correct': sum(c['confidence'] for c in correct)/len(correct) if correct else 0
            }
        }, f, ensure_ascii=False, indent=2)
    
    print("\n\nDetailed results saved to: logs/c2_error_analysis.json")

if __name__ == '__main__':
    analyze_c2_errors()
