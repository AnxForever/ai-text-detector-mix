#!/usr/bin/env python3
"""
数据质量评估脚本
评估所有可用数据集的质量指标并生成质量报告
"""

import os
import json
import pandas as pd
import numpy as np
import re
from pathlib import Path
from collections import Counter
import argparse

def detect_language(text):
    """简单的中文检测"""
    if not isinstance(text, str):
        return 'unknown'
    
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    total_chars = len(text.strip())
    
    if total_chars == 0:
        return 'empty'
    
    chinese_ratio = chinese_chars / total_chars
    if chinese_ratio > 0.5:
        return 'chinese'
    elif chinese_ratio > 0.1:
        return 'mixed'
    else:
        return 'non_chinese'

def calculate_special_char_ratio(text):
    """计算特殊字符比例"""
    if not isinstance(text, str):
        return 1.0
    
    # 定义特殊字符（非中文、英文、数字、常见标点）
    special_chars = re.findall(r'[^\u4e00-\u9fff\w\s.,!?;:()""''—\-]', text)
    return len(special_chars) / max(len(text), 1)

def evaluate_dataset_quality(file_path):
    """评估单个数据集质量"""
    try:
        # 读取数据
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix == '.json':
            df = pd.read_json(file_path, lines=True)
        else:
            return None
        
        # 确定文本和标签列
        text_col = None
        label_col = None
        
        for col in df.columns:
            if col.lower() in ['text', 'content', '文本', '内容']:
                text_col = col
            elif col.lower() in ['label', 'is_ai', 'human_ai', '标签']:
                label_col = col
        
        if text_col is None:
            # 尝试第一列作为文本
            text_col = df.columns[0]
        
        results = {
            'file_path': str(file_path),
            'total_samples': len(df),
            'text_column': text_col,
            'label_column': label_col
        }
        
        # 1. 检查缺失值
        missing_text = df[text_col].isna().sum()
        results['missing_values'] = {
            'text_missing': int(missing_text),
            'text_missing_rate': float(missing_text / len(df))
        }
        
        if label_col:
            missing_label = df[label_col].isna().sum()
            results['missing_values']['label_missing'] = int(missing_label)
            results['missing_values']['label_missing_rate'] = float(missing_label / len(df))
        
        # 2. 文本长度分布
        valid_texts = df[text_col].dropna().astype(str)
        lengths = valid_texts.str.len()
        
        results['length_distribution'] = {
            'mean': float(lengths.mean()),
            'median': float(lengths.median()),
            'std': float(lengths.std()),
            'min': int(lengths.min()),
            'max': int(lengths.max()),
            'q25': float(lengths.quantile(0.25)),
            'q75': float(lengths.quantile(0.75))
        }
        
        # 3. 重复率
        duplicates = valid_texts.duplicated().sum()
        results['duplicates'] = {
            'duplicate_count': int(duplicates),
            'duplicate_rate': float(duplicates / len(valid_texts))
        }
        
        # 4. 标签平衡
        if label_col:
            label_counts = df[label_col].value_counts()
            results['label_balance'] = {
                'label_distribution': label_counts.to_dict(),
                'balance_ratio': float(label_counts.min() / label_counts.max()) if len(label_counts) > 1 else 1.0
            }
        
        # 5. 特殊字符比例
        special_ratios = valid_texts.apply(calculate_special_char_ratio)
        results['special_characters'] = {
            'mean_special_ratio': float(special_ratios.mean()),
            'high_special_count': int((special_ratios > 0.1).sum()),
            'high_special_rate': float((special_ratios > 0.1).sum() / len(special_ratios))
        }
        
        # 6. 语言检测
        languages = valid_texts.apply(detect_language)
        lang_counts = languages.value_counts()
        results['language_detection'] = {
            'language_distribution': lang_counts.to_dict(),
            'chinese_rate': float(lang_counts.get('chinese', 0) / len(languages))
        }
        
        # 质量评分
        quality_score = calculate_quality_score(results)
        results['quality_score'] = quality_score
        results['quality_level'] = get_quality_level(quality_score)
        
        return results
        
    except Exception as e:
        return {
            'file_path': str(file_path),
            'error': str(e),
            'quality_level': 'error'
        }

def calculate_quality_score(results):
    """计算质量分数 (0-100)"""
    score = 100
    
    # 缺失值惩罚
    text_missing_rate = results['missing_values']['text_missing_rate']
    score -= text_missing_rate * 30
    
    # 重复率惩罚
    duplicate_rate = results['duplicates']['duplicate_rate']
    score -= duplicate_rate * 25
    
    # 标签平衡奖励
    if 'label_balance' in results:
        balance_ratio = results['label_balance']['balance_ratio']
        score += (balance_ratio - 0.5) * 20  # 0.5-1.0 范围给予奖励
    
    # 特殊字符惩罚
    high_special_rate = results['special_characters']['high_special_rate']
    score -= high_special_rate * 15
    
    # 语言一致性奖励
    chinese_rate = results['language_detection']['chinese_rate']
    score += (chinese_rate - 0.8) * 10 if chinese_rate > 0.8 else 0
    
    # 长度分布惩罚（过短或过长）
    mean_length = results['length_distribution']['mean']
    if mean_length < 50 or mean_length > 2000:
        score -= 10
    
    return max(0, min(100, score))

def get_quality_level(score):
    """根据分数确定质量等级"""
    if score >= 80:
        return 'high_quality'
    elif score >= 60:
        return 'medium_quality'
    else:
        return 'low_quality'

def find_datasets(base_dir):
    """查找所有数据集文件"""
    datasets = []
    base_path = Path(base_dir)
    
    # 查找CSV和JSON文件
    for pattern in ['**/*.csv', '**/*.json']:
        for file_path in base_path.glob(pattern):
            # 跳过隐藏文件和临时文件
            if not file_path.name.startswith('.') and not file_path.name.startswith('~'):
                datasets.append(file_path)
    
    return datasets

def main():
    parser = argparse.ArgumentParser(description='评估数据集质量')
    parser.add_argument('--input_dir', default='datasets', help='数据集目录')
    parser.add_argument('--output', default='data_quality_report.json', help='输出报告文件')
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    # 查找所有数据集
    datasets = find_datasets(args.input_dir)
    
    if not datasets:
        print(f"在 {args.input_dir} 中未找到数据集文件")
        return
    
    print(f"找到 {len(datasets)} 个数据集文件")
    
    # 评估每个数据集
    quality_report = {
        'evaluation_summary': {
            'total_datasets': len(datasets),
            'high_quality': 0,
            'medium_quality': 0,
            'low_quality': 0,
            'errors': 0
        },
        'datasets': []
    }
    
    for dataset_path in datasets:
        if args.verbose:
            print(f"评估: {dataset_path}")
        
        result = evaluate_dataset_quality(dataset_path)
        quality_report['datasets'].append(result)
        
        # 更新统计
        quality_level = result.get('quality_level', 'error')
        if quality_level in quality_report['evaluation_summary']:
            quality_report['evaluation_summary'][quality_level] += 1
        
        if args.verbose:
            if 'error' in result:
                print(f"  错误: {result['error']}")
            else:
                print(f"  质量: {quality_level} (分数: {result.get('quality_score', 0):.1f})")
                print(f"  样本数: {result['total_samples']}")
    
    # 保存报告
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(quality_report, f, ensure_ascii=False, indent=2)
    
    # 输出摘要
    summary = quality_report['evaluation_summary']
    print(f"\n质量评估完成！")
    print(f"总数据集: {summary['total_datasets']}")
    print(f"高质量: {summary['high_quality']}")
    print(f"中质量: {summary['medium_quality']}")
    print(f"低质量: {summary['low_quality']}")
    print(f"错误: {summary['errors']}")
    print(f"报告已保存到: {args.output}")

if __name__ == '__main__':
    main()