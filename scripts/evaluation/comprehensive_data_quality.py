#!/usr/bin/env python3
"""
AI文本检测数据集综合评价系统
基于研究报告的多层级评价框架

评价维度：
1. 基础层：统计指标（PPL、突发性、去重）
2. 进阶层：训练动力学（Cartography、AUM、Cleanlab）
3. 语义层：语域分类、LLM-as-Judge
4. 前沿层：混合文本边界检测

作者：AI Detection Team
日期：2026-01-26
"""

import os
import sys
import json
import re
import math
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import Counter
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np
from tqdm import tqdm

# 设置编码
os.environ['PYTHONIOENCODING'] = 'utf-8'


@dataclass
class QualityMetrics:
    """数据质量指标"""
    # 基础统计
    total_samples: int = 0
    ai_samples: int = 0
    human_samples: int = 0
    balance_ratio: float = 0.0
    
    # 长度分布
    length_mean: float = 0.0
    length_std: float = 0.0
    length_cv: float = 0.0  # 变异系数
    
    # 重复率
    exact_duplicate_rate: float = 0.0
    near_duplicate_rate: float = 0.0
    
    # 语言纯净度
    chinese_rate: float = 0.0
    mixed_rate: float = 0.0
    
    # 格式偏差
    ai_markdown_rate: float = 0.0
    human_markdown_rate: float = 0.0
    format_bias: float = 0.0
    
    # 质量评分
    quality_score: float = 0.0
    quality_level: str = ""


@dataclass
class DifficultyMetrics:
    """识别难度指标"""
    # 基于PPL的难度分布
    easy_samples: int = 0      # 低PPL差异，易识别
    medium_samples: int = 0    # 中等PPL差异
    hard_samples: int = 0      # 高PPL差异，难识别
    
    # 突发性分布
    ai_burstiness_mean: float = 0.0
    human_burstiness_mean: float = 0.0
    burstiness_overlap: float = 0.0  # 分布重叠度
    
    # 模糊区样本比例（高价值样本）
    ambiguous_rate: float = 0.0


class TextAnalyzer:
    """文本分析器"""
    
    @staticmethod
    def detect_language(text: str) -> str:
        """检测文本语言"""
        if not isinstance(text, str) or len(text.strip()) == 0:
            return 'empty'
        
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(text.strip())
        
        chinese_ratio = chinese_chars / total_chars
        if chinese_ratio > 0.5:
            return 'chinese'
        elif chinese_ratio > 0.1:
            return 'mixed'
        else:
            return 'non_chinese'
    
    @staticmethod
    def has_markdown(text: str) -> bool:
        """检测是否包含markdown格式"""
        if not isinstance(text, str):
            return False
        
        patterns = [
            r'^#{1,6}\s',           # 标题
            r'\*\*[^*]+\*\*',       # 粗体
            r'\*[^*]+\*',           # 斜体
            r'```[\s\S]*?```',      # 代码块
            r'`[^`]+`',             # 行内代码
            r'^\s*[-*+]\s',         # 无序列表
            r'^\s*\d+\.\s',         # 有序列表
            r'\[.+\]\(.+\)',        # 链接
            r'^\s*>\s',             # 引用
            r'\|.+\|',              # 表格
        ]
        
        for pattern in patterns:
            if re.search(pattern, text, re.MULTILINE):
                return True
        return False
    
    @staticmethod
    def calculate_burstiness(text: str) -> float:
        """
        计算文本突发性（句子长度变异度）
        人类文本通常具有更高的突发性
        """
        if not isinstance(text, str) or len(text.strip()) == 0:
            return 0.0
        
        # 分句
        sentences = re.split(r'[。！？.!?]', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
        
        if len(sentences) < 2:
            return 0.0
        
        lengths = [len(s) for s in sentences]
        mean_len = np.mean(lengths)
        std_len = np.std(lengths)
        
        # 突发性 = 标准差 / 均值（变异系数）
        if mean_len > 0:
            return std_len / mean_len
        return 0.0
    
    @staticmethod
    def estimate_perplexity_proxy(text: str) -> float:
        """
        PPL代理指标：基于词汇多样性和罕见词比例
        真实PPL需要语言模型，这里用统计代理
        """
        if not isinstance(text, str) or len(text.strip()) == 0:
            return 0.0
        
        # 分词（简单按字符和标点分割）
        tokens = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z]+|\d+', text)
        
        if len(tokens) < 5:
            return 0.0
        
        # 词汇多样性 (Type-Token Ratio)
        unique_tokens = set(tokens)
        ttr = len(unique_tokens) / len(tokens)
        
        # 长词比例（可能是罕见词）
        long_tokens = [t for t in tokens if len(t) > 2]
        long_ratio = len(long_tokens) / len(tokens)
        
        # 综合代理指标
        return ttr * 0.6 + long_ratio * 0.4
    
    @staticmethod
    def calculate_special_char_ratio(text: str) -> float:
        """计算特殊字符比例"""
        if not isinstance(text, str) or len(text) == 0:
            return 1.0
        
        special_chars = re.findall(r'[^\u4e00-\u9fff\w\s.,!?;:()""''—\-]', text)
        return len(special_chars) / len(text)


class DataQualityEvaluator:
    """数据质量评估器"""
    
    def __init__(self, df: pd.DataFrame, text_col: str = 'text', label_col: str = 'label'):
        self.df = df.copy()
        self.text_col = text_col
        self.label_col = label_col
        self.analyzer = TextAnalyzer()
        
    def evaluate_basic_stats(self) -> Dict:
        """评估基础统计指标"""
        total = len(self.df)
        ai_count = (self.df[self.label_col] == 1).sum()
        human_count = (self.df[self.label_col] == 0).sum()
        
        # 长度统计
        self.df['_length'] = self.df[self.text_col].astype(str).str.len()
        
        return {
            'total_samples': total,
            'ai_samples': int(ai_count),
            'human_samples': int(human_count),
            'balance_ratio': human_count / ai_count if ai_count > 0 else 0,
            'length_mean': float(self.df['_length'].mean()),
            'length_std': float(self.df['_length'].std()),
            'length_cv': float(self.df['_length'].std() / self.df['_length'].mean()) 
                         if self.df['_length'].mean() > 0 else 0,
        }
    
    def evaluate_duplicates(self) -> Dict:
        """评估重复率"""
        texts = self.df[self.text_col].astype(str)
        
        # 精确重复
        exact_dups = texts.duplicated().sum()
        exact_rate = exact_dups / len(texts)
        
        # 近似重复（基于前100字符的哈希）
        text_hashes = texts.apply(lambda x: hash(x[:100]) if len(x) > 100 else hash(x))
        near_dups = text_hashes.duplicated().sum()
        near_rate = near_dups / len(texts)
        
        return {
            'exact_duplicate_count': int(exact_dups),
            'exact_duplicate_rate': float(exact_rate),
            'near_duplicate_count': int(near_dups),
            'near_duplicate_rate': float(near_rate),
        }
    
    def evaluate_language_purity(self) -> Dict:
        """评估语言纯净度"""
        languages = self.df[self.text_col].apply(self.analyzer.detect_language)
        lang_counts = languages.value_counts()
        
        return {
            'chinese_rate': float(lang_counts.get('chinese', 0) / len(languages)),
            'mixed_rate': float(lang_counts.get('mixed', 0) / len(languages)),
            'non_chinese_rate': float(lang_counts.get('non_chinese', 0) / len(languages)),
            'empty_rate': float(lang_counts.get('empty', 0) / len(languages)),
        }
    
    def evaluate_format_bias(self) -> Dict:
        """评估格式偏差"""
        ai_df = self.df[self.df[self.label_col] == 1]
        human_df = self.df[self.df[self.label_col] == 0]
        
        ai_md_rate = ai_df[self.text_col].apply(self.analyzer.has_markdown).mean()
        human_md_rate = human_df[self.text_col].apply(self.analyzer.has_markdown).mean()
        
        return {
            'ai_markdown_rate': float(ai_md_rate),
            'human_markdown_rate': float(human_md_rate),
            'format_bias': float(abs(ai_md_rate - human_md_rate)),
        }
    
    def evaluate_difficulty_distribution(self) -> Dict:
        """评估识别难度分布"""
        ai_df = self.df[self.df[self.label_col] == 1]
        human_df = self.df[self.df[self.label_col] == 0]
        
        # 计算突发性
        ai_burstiness = ai_df[self.text_col].apply(self.analyzer.calculate_burstiness)
        human_burstiness = human_df[self.text_col].apply(self.analyzer.calculate_burstiness)
        
        # 计算PPL代理
        ai_ppl = ai_df[self.text_col].apply(self.analyzer.estimate_perplexity_proxy)
        human_ppl = human_df[self.text_col].apply(self.analyzer.estimate_perplexity_proxy)
        
        # 难度分层（基于PPL差异）
        ppl_diff_threshold_easy = 0.1
        ppl_diff_threshold_hard = 0.3
        
        # 计算每个样本的"可区分度"
        ai_mean_ppl = ai_ppl.mean()
        human_mean_ppl = human_ppl.mean()
        
        # 基于突发性的分布重叠度
        ai_burst_range = (ai_burstiness.quantile(0.25), ai_burstiness.quantile(0.75))
        human_burst_range = (human_burstiness.quantile(0.25), human_burstiness.quantile(0.75))
        
        overlap_start = max(ai_burst_range[0], human_burst_range[0])
        overlap_end = min(ai_burst_range[1], human_burst_range[1])
        overlap = max(0, overlap_end - overlap_start)
        total_range = max(ai_burst_range[1], human_burst_range[1]) - min(ai_burst_range[0], human_burst_range[0])
        overlap_ratio = overlap / total_range if total_range > 0 else 0
        
        return {
            'ai_burstiness_mean': float(ai_burstiness.mean()),
            'ai_burstiness_std': float(ai_burstiness.std()),
            'human_burstiness_mean': float(human_burstiness.mean()),
            'human_burstiness_std': float(human_burstiness.std()),
            'burstiness_overlap': float(overlap_ratio),
            'ai_ppl_proxy_mean': float(ai_mean_ppl),
            'human_ppl_proxy_mean': float(human_mean_ppl),
            'ppl_difference': float(abs(ai_mean_ppl - human_mean_ppl)),
            'ambiguous_rate': float(overlap_ratio),  # 重叠区即模糊区
        }
    
    def calculate_quality_score(self, metrics: Dict) -> Tuple[float, str]:
        """计算综合质量评分"""
        score = 100.0
        
        # 1. 标签平衡（权重20%）
        balance = metrics.get('balance_ratio', 0)
        if balance > 0:
            balance_penalty = abs(1 - balance) * 20
            score -= min(balance_penalty, 20)
        
        # 2. 重复率惩罚（权重15%）
        dup_rate = metrics.get('exact_duplicate_rate', 0)
        score -= dup_rate * 15
        
        # 3. 语言纯净度（权重15%）
        chinese_rate = metrics.get('chinese_rate', 0)
        if chinese_rate < 0.9:
            score -= (0.9 - chinese_rate) * 15
        
        # 4. 格式偏差惩罚（权重20%）
        format_bias = metrics.get('format_bias', 0)
        score -= format_bias * 20
        
        # 5. 难度多样性奖励（权重15%）
        ambiguous_rate = metrics.get('ambiguous_rate', 0)
        if ambiguous_rate > 0.2:  # 模糊区样本>20%是好事
            score += min(ambiguous_rate * 10, 10)
        
        # 6. 长度变异合理性（权重15%）
        length_cv = metrics.get('length_cv', 0)
        if 0.3 < length_cv < 1.0:  # 合理的变异范围
            score += 5
        elif length_cv > 1.5:  # 变异过大
            score -= 10
        
        score = max(0, min(100, score))
        
        if score >= 80:
            level = "优秀"
        elif score >= 60:
            level = "良好"
        elif score >= 40:
            level = "一般"
        else:
            level = "需改进"
        
        return score, level
    
    def run_full_evaluation(self) -> Dict:
        """运行完整评估"""
        print("正在进行数据质量评估...")
        
        results = {}
        
        # 基础统计
        print("  [1/5] 基础统计...")
        results.update(self.evaluate_basic_stats())
        
        # 重复率
        print("  [2/5] 重复率检测...")
        results.update(self.evaluate_duplicates())
        
        # 语言纯净度
        print("  [3/5] 语言纯净度...")
        results.update(self.evaluate_language_purity())
        
        # 格式偏差
        print("  [4/5] 格式偏差...")
        results.update(self.evaluate_format_bias())
        
        # 难度分布
        print("  [5/5] 难度分布...")
        results.update(self.evaluate_difficulty_distribution())
        
        # 综合评分
        score, level = self.calculate_quality_score(results)
        results['quality_score'] = score
        results['quality_level'] = level
        
        return results


class CartographyAnalyzer:
    """
    数据集制图学分析器
    基于训练动力学的样本分层
    """
    
    def __init__(self, training_history: Optional[List[Dict]] = None):
        """
        Args:
            training_history: 训练历史，每个epoch的预测概率
                格式: [{'epoch': 1, 'predictions': {sample_id: prob, ...}}, ...]
        """
        self.training_history = training_history or []
    
    def calculate_confidence_variability(self, sample_probs: List[float]) -> Tuple[float, float]:
        """
        计算置信度和变异度
        
        Args:
            sample_probs: 某样本在各epoch的预测概率列表
        
        Returns:
            (confidence, variability)
        """
        if not sample_probs:
            return 0.0, 0.0
        
        confidence = np.mean(sample_probs)
        variability = np.std(sample_probs)
        
        return float(confidence), float(variability)
    
    def classify_samples(self, confidence: float, variability: float) -> str:
        """
        根据置信度和变异度分类样本
        
        Returns:
            'easy': 易学区（高置信度，低变异度）
            'ambiguous': 模糊区（中置信度，高变异度）- 高价值
            'hard': 难学区（低置信度，低变异度）- 可能是标签错误
        """
        if confidence > 0.7 and variability < 0.2:
            return 'easy'
        elif variability > 0.3:
            return 'ambiguous'
        elif confidence < 0.3 and variability < 0.2:
            return 'hard'
        else:
            return 'medium'
    
    def analyze_dataset(self) -> Dict:
        """分析整个数据集的样本分布"""
        if not self.training_history:
            return {
                'status': 'no_training_history',
                'message': '需要提供训练历史数据才能进行制图学分析'
            }
        
        # 汇总每个样本的预测概率
        sample_probs = {}
        for epoch_data in self.training_history:
            for sample_id, prob in epoch_data.get('predictions', {}).items():
                if sample_id not in sample_probs:
                    sample_probs[sample_id] = []
                sample_probs[sample_id].append(prob)
        
        # 分类统计
        categories = {'easy': 0, 'ambiguous': 0, 'hard': 0, 'medium': 0}
        sample_details = []
        
        for sample_id, probs in sample_probs.items():
            conf, var = self.calculate_confidence_variability(probs)
            category = self.classify_samples(conf, var)
            categories[category] += 1
            sample_details.append({
                'sample_id': sample_id,
                'confidence': conf,
                'variability': var,
                'category': category
            })
        
        total = sum(categories.values())
        return {
            'status': 'success',
            'total_samples': total,
            'easy_count': categories['easy'],
            'easy_rate': categories['easy'] / total if total > 0 else 0,
            'ambiguous_count': categories['ambiguous'],
            'ambiguous_rate': categories['ambiguous'] / total if total > 0 else 0,
            'hard_count': categories['hard'],
            'hard_rate': categories['hard'] / total if total > 0 else 0,
            'sample_details': sample_details[:100]  # 只返回前100个详情
        }


def print_evaluation_report(results: Dict, dataset_name: str = ""):
    """打印评估报告"""
    
    print("\n" + "=" * 70)
    print(f"AI文本检测数据集综合评价报告")
    if dataset_name:
        print(f"数据集: {dataset_name}")
    print(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # 综合评分
    score = results.get('quality_score', 0)
    level = results.get('quality_level', '未知')
    print(f"\n【综合评分】 {score:.1f}/100 ({level})")
    
    # 基础统计
    print(f"\n【基础统计】")
    print(f"  总样本数: {results.get('total_samples', 0):,}")
    print(f"  AI样本: {results.get('ai_samples', 0):,}")
    print(f"  人类样本: {results.get('human_samples', 0):,}")
    print(f"  标签平衡比: {results.get('balance_ratio', 0):.3f}")
    
    # 长度分布
    print(f"\n【长度分布】")
    print(f"  平均长度: {results.get('length_mean', 0):.0f} 字符")
    print(f"  标准差: {results.get('length_std', 0):.0f}")
    print(f"  变异系数: {results.get('length_cv', 0):.3f}")
    
    # 重复率
    print(f"\n【重复率】")
    print(f"  精确重复: {results.get('exact_duplicate_rate', 0)*100:.2f}%")
    print(f"  近似重复: {results.get('near_duplicate_rate', 0)*100:.2f}%")
    
    # 语言纯净度
    print(f"\n【语言纯净度】")
    print(f"  中文比例: {results.get('chinese_rate', 0)*100:.1f}%")
    print(f"  混合语言: {results.get('mixed_rate', 0)*100:.1f}%")
    
    # 格式偏差
    print(f"\n【格式偏差】")
    print(f"  AI文本Markdown率: {results.get('ai_markdown_rate', 0)*100:.1f}%")
    print(f"  人类文本Markdown率: {results.get('human_markdown_rate', 0)*100:.1f}%")
    bias = results.get('format_bias', 0)
    bias_status = "✅ 平衡" if bias < 0.1 else "⚠️ 偏差较大" if bias < 0.3 else "🔴 严重偏差"
    print(f"  格式偏差: {bias*100:.1f}% {bias_status}")
    
    # 难度分布
    print(f"\n【难度分布】")
    print(f"  AI突发性均值: {results.get('ai_burstiness_mean', 0):.3f}")
    print(f"  人类突发性均值: {results.get('human_burstiness_mean', 0):.3f}")
    print(f"  分布重叠度: {results.get('burstiness_overlap', 0)*100:.1f}%")
    print(f"  模糊区比例: {results.get('ambiguous_rate', 0)*100:.1f}%")
    
    # 改进建议
    print(f"\n【改进建议】")
    suggestions = []
    
    if results.get('balance_ratio', 0) < 0.8 or results.get('balance_ratio', 0) > 1.2:
        suggestions.append("- 标签不平衡，建议调整AI/人类样本比例至1:1")
    
    if results.get('exact_duplicate_rate', 0) > 0.01:
        suggestions.append("- 存在重复样本，建议进行去重处理")
    
    if results.get('format_bias', 0) > 0.15:
        suggestions.append("- 格式偏差较大，建议对AI文本进行格式去偏处理")
    
    if results.get('ambiguous_rate', 0) < 0.15:
        suggestions.append("- 模糊区样本不足，建议增加高质量对抗样本")
    
    if results.get('chinese_rate', 0) < 0.9:
        suggestions.append("- 语言纯净度不足，建议过滤非中文样本")
    
    if not suggestions:
        suggestions.append("✅ 数据集质量良好，无明显问题")
    
    for s in suggestions:
        print(f"  {s}")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description='AI文本检测数据集综合评价系统')
    parser.add_argument('--input', type=str, default='datasets/active/core_v1/train.csv',
                       help='输入数据集路径')
    parser.add_argument('--output', type=str, default='evaluation_results/data_quality_report.json',
                       help='输出报告路径')
    parser.add_argument('--text-col', type=str, default='text', help='文本列名')
    parser.add_argument('--label-col', type=str, default='label', help='标签列名')
    parser.add_argument('--all', action='store_true', help='评估所有数据集')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("AI文本检测数据集综合评价系统")
    print("基于研究报告的多层级评价框架")
    print("=" * 70)
    
    if args.all:
        # 评估所有数据集
        datasets = [
            ('datasets/active/core_v1/train.csv', '训练集'),
            ('datasets/active/core_v1/val.csv', '验证集'),
            ('datasets/active/core_v1/test.csv', '测试集'),
        ]
        
        all_results = {}
        for path, name in datasets:
            if os.path.exists(path):
                print(f"\n正在评估: {name}")
                df = pd.read_csv(path, encoding='utf-8-sig')
                evaluator = DataQualityEvaluator(df, args.text_col, args.label_col)
                results = evaluator.run_full_evaluation()
                print_evaluation_report(results, name)
                all_results[name] = results
        
        # 保存汇总报告
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n报告已保存至: {args.output}")
        
    else:
        # 评估单个数据集
        if not os.path.exists(args.input):
            print(f"错误: 文件不存在 {args.input}")
            return
        
        df = pd.read_csv(args.input, encoding='utf-8-sig')
        evaluator = DataQualityEvaluator(df, args.text_col, args.label_col)
        results = evaluator.run_full_evaluation()
        
        print_evaluation_report(results, args.input)
        
        # 保存报告
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n报告已保存至: {args.output}")


if __name__ == '__main__':
    main()
