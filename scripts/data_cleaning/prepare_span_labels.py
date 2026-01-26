#!/usr/bin/env python3
"""
准备C2 span检测的token-level标注数据
"""
import json
import pandas as pd
from transformers import BertTokenizer
from tqdm import tqdm

def align_labels(text, boundary, tokenizer):
    """
    将字符级别的boundary转换为token级别的标签
    简化版：基于字符位置估算
    """
    # Tokenize
    encoding = tokenizer(text, add_special_tokens=True)
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
    
    # 简单对齐：计算每个token对应的大致字符位置
    token_labels = []
    char_pos = 0
    
    for i, token in enumerate(tokens):
        if token in ['[CLS]', '[SEP]', '[PAD]']:
            token_labels.append(-100)  # 忽略
        else:
            # 移除##前缀
            token_text = token.replace('##', '')
            token_len = len(token_text)
            
            # token中点位置
            token_mid = char_pos + token_len / 2
            
            if token_mid < boundary:
                token_labels.append(0)  # Human
            else:
                token_labels.append(1)  # AI
            
            char_pos += token_len
    
    return token_labels, tokens

def prepare_span_dataset(input_file, output_file, tokenizer):
    """
    处理C2数据，生成token-level标注
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed = []
    skipped = 0
    
    for item in tqdm(data, desc="Processing"):
        if 'boundary' not in item:
            skipped += 1
            continue
        
        text = item['text']
        boundary = item['boundary']
        
        # 检查[SEP]标记
        if '[SEP]' in text:
            # 移除[SEP]，调整boundary
            sep_pos = text.index('[SEP]')
            if sep_pos < boundary:
                boundary -= 5  # len('[SEP]')
            text_clean = text.replace('[SEP]', '')
        else:
            text_clean = text
        
        # 生成token标签
        try:
            token_labels, tokens = align_labels(text_clean, boundary, tokenizer)
            
            processed.append({
                'text_id': item.get('text_id', f"c2_{len(processed)}"),
                'text': text_clean,
                'boundary': boundary,
                'token_labels': token_labels,
                'tokens': tokens,
                'num_tokens': len(tokens),
                'human_tokens': sum(1 for l in token_labels if l == 0),
                'ai_tokens': sum(1 for l in token_labels if l == 1)
            })
        except Exception as e:
            print(f"Error processing item: {e}")
            skipped += 1
    
    # 保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)
    
    print(f"\nProcessed: {len(processed)}")
    print(f"Skipped: {skipped}")
    print(f"Saved to: {output_file}")
    
    # 统计
    avg_tokens = sum(item['num_tokens'] for item in processed) / len(processed)
    avg_human = sum(item['human_tokens'] for item in processed) / len(processed)
    avg_ai = sum(item['ai_tokens'] for item in processed) / len(processed)
    
    print(f"\nStatistics:")
    print(f"  Avg tokens: {avg_tokens:.1f}")
    print(f"  Avg human tokens: {avg_human:.1f}")
    print(f"  Avg AI tokens: {avg_ai:.1f}")
    
    return processed

def verify_alignment(processed_data, num_samples=3):
    """
    验证token对齐是否正确
    """
    print("\n=== Verification ===")
    for i, item in enumerate(processed_data[:num_samples]):
        print(f"\n[Sample {i+1}]")
        print(f"Text: {item['text'][:100]}...")
        print(f"Boundary: {item['boundary']}")
        print(f"Tokens: {len(item['tokens'])}")
        
        # 显示边界附近的tokens
        labels = item['token_labels']
        tokens = item['tokens']
        
        # 找到标签变化的位置
        boundary_idx = None
        for j in range(1, len(labels)):
            if labels[j-1] == 0 and labels[j] == 1:
                boundary_idx = j
                break
        
        if boundary_idx:
            print(f"Boundary at token {boundary_idx}")
            print(f"  Human tokens: {tokens[max(0, boundary_idx-3):boundary_idx]}")
            print(f"  AI tokens: {tokens[boundary_idx:min(len(tokens), boundary_idx+3)]}")

def main():
    # 加载tokenizer
    tokenizer = BertTokenizer.from_pretrained('models/bert_combined')
    
    # 处理所有C2数据文件
    import glob
    c2_files = glob.glob('datasets/hybrid/c2*.json')
    
    print(f"Found {len(c2_files)} C2 files")
    
    all_processed = []
    for file in c2_files:
        print(f"\nProcessing: {file}")
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in tqdm(data, desc=f"  {file.split('/')[-1]}"):
            if 'boundary' not in item:
                continue
            
            text = item['text']
            boundary = item['boundary']
            
            # 移除[SEP]
            if '[SEP]' in text:
                sep_pos = text.index('[SEP]')
                if sep_pos < boundary:
                    boundary -= 5
                text = text.replace('[SEP]', '')
            
            try:
                token_labels, tokens = align_labels(text, boundary, tokenizer)
                all_processed.append({
                    'text_id': item.get('text_id', f"c2_{len(all_processed)}"),
                    'text': text,
                    'boundary': boundary,
                    'token_labels': token_labels,
                    'tokens': tokens,
                    'num_tokens': len(tokens),
                    'human_tokens': sum(1 for l in token_labels if l == 0),
                    'ai_tokens': sum(1 for l in token_labels if l == 1)
                })
            except Exception as e:
                print(f"Error: {e}")
                pass
    
    # 保存
    output_file = 'datasets/hybrid/c2_span_labels.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_processed, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Total processed: {len(all_processed)}")
    print(f"✓ Saved to: {output_file}")
    
    # 验证
    verify_alignment(all_processed, num_samples=3)

if __name__ == '__main__':
    main()
