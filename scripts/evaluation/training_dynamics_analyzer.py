#!/usr/bin/env python3
"""
训练动力学分析器
实现数据集制图学(Cartography)和AUM方法

功能：
1. 数据集制图学：基于置信度和变异度的样本分层
2. AUM (Area Under Margin)：识别噪声标签
3. 置信学习：标签错误检测

作者：AI Detection Team
日期：2026-01-26
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

# 设置编码
os.environ['PYTHONIOENCODING'] = 'utf-8'


@dataclass
class SampleMetrics:
    """单个样本的训练动力学指标"""
    sample_id: int
    text: str
    true_label: int
    confidence: float      # 平均置信度
    variability: float     # 置信度变异度
    correctness: float     # 正确率
    aum: float            # Area Under Margin
    category: str         # easy/ambiguous/hard
    is_potential_error: bool  # 是否可能是标签错误


class TrainingDynamicsAnalyzer:
    """
    训练动力学分析器
    
    使用方法：
    1. 在训练过程中记录每个epoch的预测概率
    2. 训练结束后调用analyze()进行分析
    """
    
    def __init__(self):
        self.epoch_predictions = []  # List[Dict[sample_id, prob]]
        self.epoch_logits = []       # List[Dict[sample_id, (correct_logit, max_other_logit)]]
        self.sample_labels = {}      # Dict[sample_id, true_label]
        self.sample_texts = {}       # Dict[sample_id, text]
    
    def record_epoch(self, predictions: Dict[int, float], 
                     logits: Optional[Dict[int, Tuple[float, float]]] = None):
        """
        记录一个epoch的预测结果
        
        Args:
            predictions: {sample_id: probability_of_correct_class}
            logits: {sample_id: (correct_class_logit, max_other_class_logit)}
        """
        self.epoch_predictions.append(predictions)
        if logits:
            self.epoch_logits.append(logits)
    
    def set_ground_truth(self, labels: Dict[int, int], texts: Dict[int, str]):
        """设置真实标签和文本"""
        self.sample_labels = labels
        self.sample_texts = texts
    
    def calculate_confidence_variability(self, sample_id: int) -> Tuple[float, float, float]:
        """
        计算样本的置信度、变异度和正确率
        
        Returns:
            (confidence, variability, correctness)
        """
        probs = [ep.get(sample_id, 0.5) for ep in self.epoch_predictions]
        
        if not probs:
            return 0.5, 0.0, 0.0
        
        confidence = np.mean(probs)
        variability = np.std(probs)
        correctness = np.mean([1 if p > 0.5 else 0 for p in probs])
        
        return float(confidence), float(variability), float(correctness)
    
    def calculate_aum(self, sample_id: int) -> float:
        """
        计算Area Under Margin
        
        AUM = 平均(正确类logit - 最大错误类logit)
        正AUM表示模型倾向于正确分类
        负AUM表示可能是标签错误
        """
        if not self.epoch_logits:
            return 0.0
        
        margins = []
        for ep_logits in self.epoch_logits:
            if sample_id in ep_logits:
                correct_logit, max_other_logit = ep_logits[sample_id]
                margin = correct_logit - max_other_logit
                margins.append(margin)
        
        if not margins:
            return 0.0
        
        return float(np.mean(margins))
    
    def classify_sample(self, confidence: float, variability: float) -> str:
        """
        根据置信度和变异度分类样本
        
        基于数据集制图学的三区域划分：
        - easy: 高置信度(>0.7)，低变异度(<0.2)
        - ambiguous: 高变异度(>0.25) - 这是最有价值的样本
        - hard: 低置信度(<0.3)，低变异度(<0.2) - 可能是标签错误
        """
        if confidence > 0.7 and variability < 0.2:
            return 'easy'
        elif variability > 0.25:
            return 'ambiguous'
        elif confidence < 0.3 and variability < 0.2:
            return 'hard'
        else:
            return 'medium'
    
    def analyze(self) -> Dict:
        """
        执行完整的训练动力学分析
        
        Returns:
            分析结果字典
        """
        if not self.epoch_predictions:
            return {
                'status': 'error',
                'message': '没有记录的训练数据，请先调用record_epoch()'
            }
        
        results = {
            'status': 'success',
            'num_epochs': len(self.epoch_predictions),
            'num_samples': len(self.sample_labels),
            'samples': [],
            'statistics': {
                'easy_count': 0,
                'ambiguous_count': 0,
                'hard_count': 0,
                'medium_count': 0,
                'potential_errors': 0
            }
        }
        
        # 分析每个样本
        for sample_id in tqdm(self.sample_labels.keys(), desc="分析样本"):
            conf, var, corr = self.calculate_confidence_variability(sample_id)
            aum = self.calculate_aum(sample_id)
            category = self.classify_sample(conf, var)
            
            # 判断是否可能是标签错误
            # 条件：低置信度 + 低变异度 + 负AUM
            is_potential_error = (conf < 0.3 and var < 0.2) or (aum < -0.5)
            
            sample_metrics = SampleMetrics(
                sample_id=sample_id,
                text=self.sample_texts.get(sample_id, "")[:100],  # 只保留前100字符
                true_label=self.sample_labels[sample_id],
                confidence=conf,
                variability=var,
                correctness=corr,
                aum=aum,
                category=category,
                is_potential_error=is_potential_error
            )
            
            results['samples'].append({
                'sample_id': sample_metrics.sample_id,
                'text_preview': sample_metrics.text,
                'true_label': sample_metrics.true_label,
                'confidence': sample_metrics.confidence,
                'variability': sample_metrics.variability,
                'correctness': sample_metrics.correctness,
                'aum': sample_metrics.aum,
                'category': sample_metrics.category,
                'is_potential_error': sample_metrics.is_potential_error
            })
            
            # 更新统计
            results['statistics'][f'{category}_count'] += 1
            if is_potential_error:
                results['statistics']['potential_errors'] += 1
        
        # 计算比例
        total = len(self.sample_labels)
        for key in ['easy', 'ambiguous', 'hard', 'medium']:
            results['statistics'][f'{key}_rate'] = results['statistics'][f'{key}_count'] / total
        results['statistics']['potential_error_rate'] = results['statistics']['potential_errors'] / total
        
        return results
    
    def get_samples_by_category(self, category: str) -> List[Dict]:
        """获取特定类别的样本"""
        results = self.analyze()
        if results['status'] != 'success':
            return []
        
        return [s for s in results['samples'] if s['category'] == category]
    
    def get_potential_label_errors(self) -> List[Dict]:
        """获取可能的标签错误样本"""
        results = self.analyze()
        if results['status'] != 'success':
            return []
        
        return [s for s in results['samples'] if s['is_potential_error']]


class ConfidentLearning:
    """
    置信学习：基于模型预测概率识别标签错误
    
    实现Cleanlab的核心算法
    """
    
    def __init__(self, pred_probs: np.ndarray, labels: np.ndarray):
        """
        Args:
            pred_probs: 预测概率矩阵 (n_samples, n_classes)
            labels: 真实标签数组 (n_samples,)
        """
        self.pred_probs = pred_probs
        self.labels = labels
        self.n_samples = len(labels)
        self.n_classes = pred_probs.shape[1] if len(pred_probs.shape) > 1 else 2
    
    def estimate_confident_joint(self) -> np.ndarray:
        """
        估计置信联合分布矩阵
        
        Returns:
            (n_classes, n_classes) 矩阵，C[i,j]表示真实标签为i但被预测为j的样本数
        """
        # 计算每个类别的阈值（使用自适应阈值）
        thresholds = []
        for c in range(self.n_classes):
            class_probs = self.pred_probs[self.labels == c]
            if len(class_probs) > 0:
                # 使用该类别预测概率的平均值作为阈值
                if len(class_probs.shape) > 1:
                    threshold = np.mean(class_probs[:, c])
                else:
                    threshold = np.mean(class_probs)
            else:
                threshold = 0.5
            thresholds.append(threshold)
        
        # 构建置信联合矩阵
        C = np.zeros((self.n_classes, self.n_classes))
        
        for i in range(self.n_samples):
            given_label = self.labels[i]
            if len(self.pred_probs.shape) > 1:
                pred_prob = self.pred_probs[i]
                pred_label = np.argmax(pred_prob)
            else:
                pred_prob = self.pred_probs[i]
                pred_label = 1 if pred_prob > 0.5 else 0
            
            C[given_label, pred_label] += 1
        
        return C
    
    def find_label_issues(self, threshold: float = 0.5) -> List[int]:
        """
        找出可能的标签错误
        
        Args:
            threshold: 置信度阈值，低于此值的样本被标记为可疑
        
        Returns:
            可疑样本的索引列表
        """
        issues = []
        
        for i in range(self.n_samples):
            given_label = self.labels[i]
            
            if len(self.pred_probs.shape) > 1:
                prob_given_label = self.pred_probs[i, given_label]
                pred_label = np.argmax(self.pred_probs[i])
            else:
                prob_given_label = self.pred_probs[i] if given_label == 1 else 1 - self.pred_probs[i]
                pred_label = 1 if self.pred_probs[i] > 0.5 else 0
            
            # 如果模型对给定标签的置信度很低，且预测了不同的标签
            if prob_given_label < threshold and pred_label != given_label:
                issues.append(i)
        
        return issues
    
    def get_label_quality_scores(self) -> np.ndarray:
        """
        计算每个样本的标签质量分数
        
        Returns:
            (n_samples,) 数组，分数越低表示标签越可能错误
        """
        scores = np.zeros(self.n_samples)
        
        for i in range(self.n_samples):
            given_label = self.labels[i]
            
            if len(self.pred_probs.shape) > 1:
                scores[i] = self.pred_probs[i, given_label]
            else:
                scores[i] = self.pred_probs[i] if given_label == 1 else 1 - self.pred_probs[i]
        
        return scores


def simulate_training_dynamics(df: pd.DataFrame, 
                               text_col: str = 'text',
                               label_col: str = 'label',
                               n_epochs: int = 5) -> TrainingDynamicsAnalyzer:
    """
    模拟训练动力学（用于演示，实际应在训练过程中记录）
    
    基于文本特征模拟预测概率的变化
    """
    analyzer = TrainingDynamicsAnalyzer()
    
    # 设置真实标签和文本
    labels = {i: int(row[label_col]) for i, row in df.iterrows()}
    texts = {i: str(row[text_col]) for i, row in df.iterrows()}
    analyzer.set_ground_truth(labels, texts)
    
    # 模拟每个epoch的预测
    for epoch in range(n_epochs):
        predictions = {}
        logits = {}
        
        for i, row in df.iterrows():
            text = str(row[text_col])
            label = int(row[label_col])
            
            # 基于文本特征模拟预测概率
            # 这里使用简单的启发式规则
            has_markdown = bool(re.search(r'[#*`\[\]]', text))
            length = len(text)
            
            # 基础概率
            if label == 1:  # AI文本
                base_prob = 0.7 if has_markdown else 0.5
            else:  # 人类文本
                base_prob = 0.3 if has_markdown else 0.6
            
            # 添加随机波动模拟训练过程
            noise = np.random.normal(0, 0.1 * (1 - epoch / n_epochs))
            prob = np.clip(base_prob + noise, 0.1, 0.9)
            
            # 随着epoch增加，概率趋于稳定
            if epoch > 2:
                prob = prob * 0.8 + base_prob * 0.2
            
            predictions[i] = prob
            
            # 模拟logits
            correct_logit = np.log(prob / (1 - prob)) if prob < 1 else 5
            other_logit = -correct_logit
            logits[i] = (correct_logit, other_logit)
        
        analyzer.record_epoch(predictions, logits)
    
    return analyzer


def print_dynamics_report(results: Dict):
    """打印训练动力学分析报告"""
    
    print("\n" + "=" * 70)
    print("训练动力学分析报告")
    print("=" * 70)
    
    if results['status'] != 'success':
        print(f"错误: {results.get('message', '未知错误')}")
        return
    
    stats = results['statistics']
    
    print(f"\n【数据集制图学分析】")
    print(f"  分析轮次: {results['num_epochs']}")
    print(f"  样本总数: {results['num_samples']}")
    
    print(f"\n【样本分布】")
    print(f"  易学区 (Easy):     {stats['easy_count']:5d} ({stats['easy_rate']*100:5.1f}%)")
    print(f"  模糊区 (Ambiguous): {stats['ambiguous_count']:5d} ({stats['ambiguous_rate']*100:5.1f}%) ⭐ 高价值")
    print(f"  难学区 (Hard):     {stats['hard_count']:5d} ({stats['hard_rate']*100:5.1f}%) ⚠️ 可能标签错误")
    print(f"  中间区 (Medium):   {stats['medium_count']:5d} ({stats['medium_rate']*100:5.1f}%)")
    
    print(f"\n【标签质量】")
    print(f"  潜在标签错误: {stats['potential_errors']} ({stats['potential_error_rate']*100:.2f}%)")
    
    # 显示一些示例
    if results['samples']:
        print(f"\n【模糊区样本示例】（最有价值的训练样本）")
        ambiguous = [s for s in results['samples'] if s['category'] == 'ambiguous'][:3]
        for s in ambiguous:
            print(f"  ID {s['sample_id']}: 置信度={s['confidence']:.2f}, 变异度={s['variability']:.2f}")
            print(f"    文本: {s['text_preview'][:50]}...")
        
        print(f"\n【潜在标签错误示例】")
        errors = [s for s in results['samples'] if s['is_potential_error']][:3]
        for s in errors:
            label_name = "AI" if s['true_label'] == 1 else "人类"
            print(f"  ID {s['sample_id']}: 标记为{label_name}, 置信度={s['confidence']:.2f}, AUM={s['aum']:.2f}")
            print(f"    文本: {s['text_preview'][:50]}...")
    
    print("\n" + "=" * 70)


# 需要导入re模块
import re


def main():
    parser = argparse.ArgumentParser(description='训练动力学分析器')
    parser.add_argument('--input', type=str, default='datasets/active/core_v1/train.csv',
                       help='输入数据集路径')
    parser.add_argument('--output', type=str, default='evaluation_results/dynamics_report.json',
                       help='输出报告路径')
    parser.add_argument('--epochs', type=int, default=5, help='模拟训练轮次')
    parser.add_argument('--text-col', type=str, default='text', help='文本列名')
    parser.add_argument('--label-col', type=str, default='label', help='标签列名')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("训练动力学分析器")
    print("基于数据集制图学和AUM方法")
    print("=" * 70)
    
    if not os.path.exists(args.input):
        print(f"错误: 文件不存在 {args.input}")
        return
    
    # 读取数据
    print(f"\n读取数据: {args.input}")
    df = pd.read_csv(args.input, encoding='utf-8-sig')
    print(f"样本数: {len(df)}")
    
    # 模拟训练动力学
    print(f"\n模拟 {args.epochs} 轮训练...")
    analyzer = simulate_training_dynamics(df, args.text_col, args.label_col, args.epochs)
    
    # 分析
    print("\n执行训练动力学分析...")
    results = analyzer.analyze()
    
    # 打印报告
    print_dynamics_report(results)
    
    # 保存报告
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 只保存统计信息和前100个样本详情
    save_results = {
        'status': results['status'],
        'num_epochs': results['num_epochs'],
        'num_samples': results['num_samples'],
        'statistics': results['statistics'],
        'sample_examples': results['samples'][:100]
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存至: {args.output}")


if __name__ == '__main__':
    main()
