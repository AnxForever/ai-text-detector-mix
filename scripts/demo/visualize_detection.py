#!/usr/bin/env python3
"""
混合文本检测可视化演示
"""
import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertForTokenClassification
import json

class HybridTextDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载分类模型
        self.classifier_tokenizer = BertTokenizer.from_pretrained('models/bert_v2_with_sep')
        self.classifier = BertForSequenceClassification.from_pretrained('models/bert_v2_with_sep').to(self.device)
        
        # 加载span检测模型
        self.span_tokenizer = BertTokenizer.from_pretrained('models/bert_span_detector')
        self.span_detector = BertForTokenClassification.from_pretrained('models/bert_span_detector').to(self.device)
        
        self.classifier.eval()
        self.span_detector.eval()
    
    def classify(self, text):
        """判断文本是否为AI生成"""
        encoding = self.classifier_tokenizer(
            text, max_length=512, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        
        with torch.no_grad():
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            outputs = self.classifier(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits[0], dim=0)
            pred = torch.argmax(outputs.logits[0]).item()
        
        return {
            'label': 'AI' if pred == 1 else 'Human',
            'confidence': probs[pred].item(),
            'prob_human': probs[0].item(),
            'prob_ai': probs[1].item()
        }
    
    def detect_boundary(self, text):
        """检测混合文本的边界"""
        # 移除[SEP]标记
        text_clean = text.replace('[SEP]', '')
        
        encoding = self.span_tokenizer(
            text_clean, max_length=512, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        
        with torch.no_grad():
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            outputs = self.span_detector(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits[0], dim=-1).cpu()
        
        # 找到边界
        tokens = self.span_tokenizer.convert_ids_to_tokens(input_ids[0])
        labels = preds.numpy()
        
        # 找到从Human(0)到AI(1)的转换点
        boundary_idx = None
        for i in range(1, len(labels)):
            if labels[i-1] == 0 and labels[i] == 1:
                boundary_idx = i
                break
        
        # 重建文本并标记
        result = []
        char_pos = 0
        boundary_char = None
        
        for i, (token, label) in enumerate(zip(tokens, labels)):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
            
            token_text = token.replace('##', '')
            
            if i == boundary_idx:
                boundary_char = char_pos
            
            result.append({
                'token': token_text,
                'label': 'Human' if label == 0 else 'AI',
                'position': char_pos
            })
            
            char_pos += len(token_text)
        
        return {
            'tokens': result,
            'boundary_token': boundary_idx,
            'boundary_char': boundary_char,
            'text': text_clean
        }
    
    def visualize(self, text):
        """完整检测并可视化"""
        print("=" * 80)
        print("混合文本检测系统")
        print("=" * 80)
        
        # 分类
        cls_result = self.classify(text)
        print(f"\n【分类结果】")
        print(f"  预测: {cls_result['label']}")
        print(f"  置信度: {cls_result['confidence']:.2%}")
        print(f"  Human概率: {cls_result['prob_human']:.2%}")
        print(f"  AI概率: {cls_result['prob_ai']:.2%}")
        
        # 如果是AI文本，进行边界检测
        if cls_result['label'] == 'AI':
            print(f"\n【边界检测】")
            boundary_result = self.detect_boundary(text)
            
            if boundary_result['boundary_char']:
                print(f"  检测到边界位置: 第 {boundary_result['boundary_char']} 字符")
                
                text_clean = boundary_result['text']
                boundary = boundary_result['boundary_char']
                
                human_part = text_clean[:boundary]
                ai_part = text_clean[boundary:]
                
                print(f"\n【文本分段】")
                print(f"  人类部分 ({len(human_part)} 字符):")
                print(f"    {human_part[:100]}{'...' if len(human_part) > 100 else ''}")
                print(f"\n  AI部分 ({len(ai_part)} 字符):")
                print(f"    {ai_part[:100]}{'...' if len(ai_part) > 100 else ''}")
            else:
                print("  未检测到明显边界（可能为纯AI生成）")
        
        print("\n" + "=" * 80)

def main():
    detector = HybridTextDetector()
    
    # 加载测试样本
    with open('datasets/hybrid/c2_span_labels.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 随机选择3个样本演示
    import random
    samples = random.sample(data, 3)
    
    for i, sample in enumerate(samples, 1):
        print(f"\n\n{'#' * 80}")
        print(f"示例 {i}")
        print(f"{'#' * 80}")
        
        text = sample['text']
        true_boundary = sample['boundary']
        
        print(f"\n原文 (前200字):")
        print(f"{text[:200]}...")
        print(f"\n真实边界: 第 {true_boundary} 字符")
        
        detector.visualize(text)

if __name__ == '__main__':
    main()
