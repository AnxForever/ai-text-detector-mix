"""
格式偏差验证工具

功能：
1. 评估数据集的格式偏差
2. 测试简单规则的准确率
3. 对比简单规则 vs BERT模型的性能
4. 验证格式去偏效果

作者：Format Debiasing Team
日期：2026-01-11
"""

import sys
import io
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except:
        pass

import pandas as pd
import argparse
from datetime import datetime
from typing import Dict, Tuple

# 导入格式处理函数
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data_cleaning'))
from format_handler import (
    has_markdown,
    has_markdown_detailed,
    get_format_statistics
)


def evaluate_format_bias(df: pd.DataFrame) -> Dict:
    """
    评估数据集的格式偏差

    Args:
        df: 数据框（包含 text 和 label 列）

    Returns:
        格式偏差评估结果
    """
    ai_df = df[df['label'] == 1]
    human_df = df[df['label'] == 0]

    # 统计markdown比例
    ai_md_rate = ai_df['text'].apply(has_markdown).mean()
    human_md_rate = human_df['text'].apply(has_markdown).mean()
    bias = abs(ai_md_rate - human_md_rate)

    # 获取详细统计
    ai_stats = get_format_statistics(ai_df['text'].tolist())
    human_stats = get_format_statistics(human_df['text'].tolist())

    # 模拟简单规则（只判断markdown）
    df_temp = df.copy()
    df_temp['simple_pred'] = df_temp['text'].apply(lambda x: 1 if has_markdown(x) else 0)
    simple_accuracy = (df_temp['simple_pred'] == df_temp['label']).mean()

    # 计算混淆矩阵
    tp = ((df_temp['simple_pred'] == 1) & (df_temp['label'] == 1)).sum()  # 正确识别AI
    fp = ((df_temp['simple_pred'] == 1) & (df_temp['label'] == 0)).sum()  # 人类误判为AI
    tn = ((df_temp['simple_pred'] == 0) & (df_temp['label'] == 0)).sum()  # 正确识别人类
    fn = ((df_temp['simple_pred'] == 0) & (df_temp['label'] == 1)).sum()  # AI误判为人类

    simple_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    simple_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    simple_f1 = 2 * (simple_precision * simple_recall) / (simple_precision + simple_recall) \
                if (simple_precision + simple_recall) > 0 else 0

    # 判断状态
    if bias < 0.05:
        status = 'pass'
        recommendation = "✅ 格式偏差<5%，分布非常平衡！"
    elif bias < 0.15:
        status = 'warning'
        recommendation = "⚠️ 格式偏差5-15%，基本平衡，但可以进一步改善"
    else:
        status = 'fail'
        recommendation = "🔴 格式偏差>15%，模型可能过拟合格式特征"

    return {
        'ai_markdown_rate': ai_md_rate,
        'human_markdown_rate': human_md_rate,
        'bias': bias,
        'ai_stats': ai_stats,
        'human_stats': human_stats,
        'simple_rule': {
            'accuracy': simple_accuracy,
            'precision': simple_precision,
            'recall': simple_recall,
            'f1': simple_f1,
            'confusion_matrix': {
                'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
            }
        },
        'status': status,
        'recommendation': recommendation
    }


def compare_simple_vs_bert(test_df: pd.DataFrame, bert_model=None, tokenizer=None) -> Dict:
    """
    对比简单规则 vs BERT模型

    Args:
        test_df: 测试数据集
        bert_model: BERT模型（可选）
        tokenizer: Tokenizer（可选）

    Returns:
        对比结果
    """
    # 简单规则性能
    test_df_temp = test_df.copy()
    test_df_temp['simple_pred'] = test_df_temp['text'].apply(lambda x: 1 if has_markdown(x) else 0)
    simple_accuracy = (test_df_temp['simple_pred'] == test_df_temp['label']).mean()

    result = {
        'simple_accuracy': simple_accuracy,
        'bert_accuracy': None,
        'improvement': None
    }

    # 如果提供了BERT模型，评估BERT性能
    if bert_model is not None and tokenizer is not None:
        import torch

        bert_model.eval()
        predictions = []

        with torch.no_grad():
            for text in test_df['text']:
                encoding = tokenizer(
                    text,
                    max_length=512,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )

                outputs = bert_model(**encoding)
                pred = torch.argmax(outputs.logits, dim=1).item()
                predictions.append(pred)

        bert_accuracy = (pd.Series(predictions) == test_df['label']).mean()

        result['bert_accuracy'] = bert_accuracy
        result['improvement'] = bert_accuracy - simple_accuracy

    return result


def print_format_bias_report(results: Dict, dataset_name: str = ""):
    """打印格式偏差报告"""

    print("\n" + "="*70)
    if dataset_name:
        print(f"格式偏差评估报告 - {dataset_name}")
    else:
        print("格式偏差评估报告")
    print("="*70)

    # 基本统计
    print(f"\n【格式分布统计】")
    print(f"AI文本 markdown率: {results['ai_markdown_rate']*100:.2f}%")
    print(f"人类文本 markdown率: {results['human_markdown_rate']*100:.2f}%")
    print(f"格式偏差: {results['bias']*100:.2f}%")

    print(f"\n状态: {results['recommendation']}")

    # 详细格式类型统计
    print(f"\n【格式类型统计】")
    print("\nAI文本格式类型分布:")
    for fmt_type, pct in results['ai_stats']['format_type_percentages'].items():
        if pct > 0:
            print(f"  {fmt_type}: {pct:.1f}%")

    print("\n人类文本格式类型分布:")
    for fmt_type, pct in results['human_stats']['format_type_percentages'].items():
        if pct > 0:
            print(f"  {fmt_type}: {pct:.1f}%")

    # 简单规则性能
    print(f"\n【简单规则性能】（仅判断markdown）")
    print(f"准确率: {results['simple_rule']['accuracy']*100:.2f}%")
    print(f"精确率: {results['simple_rule']['precision']*100:.2f}%")
    print(f"召回率: {results['simple_rule']['recall']*100:.2f}%")
    print(f"F1分数: {results['simple_rule']['f1']*100:.2f}%")

    cm = results['simple_rule']['confusion_matrix']
    print(f"\n混淆矩阵:")
    print(f"  真正例(TP): {cm['tp']} | 假正例(FP): {cm['fp']}")
    print(f"  假负例(FN): {cm['fn']} | 真负例(TN): {cm['tn']}")

    # 警告信息
    if results['simple_rule']['accuracy'] > 0.70:
        print(f"\n⚠️ 警告：简单规则准确率 > 70%，格式是强信号！")
        print(f"   模型可能主要学习格式而非语义")
        if cm['fn'] > 0:
            print(f"   发现 {cm['fn']} 条无markdown的AI文本被误判为人类")
    elif results['simple_rule']['accuracy'] < 0.50:
        print(f"\n✅ 优秀：简单规则准确率 < 50%，格式已不是有效特征")


def validate_debiasing_effect(original_results: Dict, debiased_results: Dict):
    """
    验证去偏效果

    Args:
        original_results: 原始数据集的评估结果
        debiased_results: 去偏后数据集的评估结果
    """
    print("\n" + "="*70)
    print("去偏效果验证")
    print("="*70)

    # 格式偏差变化
    bias_reduction = original_results['bias'] - debiased_results['bias']
    print(f"\n【格式偏差变化】")
    print(f"原始: {original_results['bias']*100:.2f}%")
    print(f"去偏后: {debiased_results['bias']*100:.2f}%")
    print(f"降低: {bias_reduction*100:.2f}% ({'✅' if bias_reduction > 0.50 else '⚠️'})")

    # 简单规则准确率变化
    acc_change = original_results['simple_rule']['accuracy'] - debiased_results['simple_rule']['accuracy']
    print(f"\n【简单规则准确率变化】")
    print(f"原始: {original_results['simple_rule']['accuracy']*100:.2f}%")
    print(f"去偏后: {debiased_results['simple_rule']['accuracy']*100:.2f}%")
    print(f"降低: {acc_change*100:.2f}% ({'✅' if acc_change > 0.20 else '⚠️'})")

    # AI markdown率变化
    ai_md_reduction = original_results['ai_markdown_rate'] - debiased_results['ai_markdown_rate']
    print(f"\n【AI文本 markdown率变化】")
    print(f"原始: {original_results['ai_markdown_rate']*100:.2f}%")
    print(f"去偏后: {debiased_results['ai_markdown_rate']*100:.2f}%")
    print(f"降低: {ai_md_reduction*100:.2f}%")

    # 总体评估
    print(f"\n【总体评估】")
    if debiased_results['bias'] < 0.05 and debiased_results['simple_rule']['accuracy'] < 0.50:
        print("✅ 去偏效果优秀！格式偏差<5%，简单规则失效")
        print("   模型将被迫学习语义特征而非格式特征")
    elif debiased_results['bias'] < 0.10 and debiased_results['simple_rule']['accuracy'] < 0.60:
        print("⚠️ 去偏效果良好，但仍有改进空间")
    else:
        print("🔴 去偏效果不足，需要进一步调整")


def main():
    parser = argparse.ArgumentParser(description='格式偏差验证工具')
    parser.add_argument('--original-dir', type=str, default='datasets/bert',
                       help='原始数据集目录')
    parser.add_argument('--debiased-dir', type=str, default='datasets/bert_debiased',
                       help='去偏后数据集目录')
    parser.add_argument('--compare', action='store_true',
                       help='对比原始和去偏后的数据集')
    parser.add_argument('--dataset', type=str, choices=['train', 'val', 'test', 'all'],
                       default='all', help='要评估的数据集')

    args = parser.parse_args()

    print("="*70)
    print("格式偏差验证工具")
    print("="*70)
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    datasets_to_check = ['train', 'val', 'test'] if args.dataset == 'all' else [args.dataset]

    if args.compare:
        # 对比模式
        print("模式：对比原始 vs 去偏后\n")

        for ds_name in datasets_to_check:
            print(f"\n{'='*70}")
            print(f"评估 {ds_name.upper()} 数据集")
            print(f"{'='*70}")

            # 读取数据
            original_df = pd.read_csv(
                f"{args.original_dir}/{ds_name}.csv",
                encoding='utf-8-sig'
            )
            debiased_df = pd.read_csv(
                f"{args.debiased_dir}/{ds_name}.csv",
                encoding='utf-8-sig'
            )

            print(f"\n原始数据集: {len(original_df)}条")
            print(f"去偏数据集: {len(debiased_df)}条")

            # 评估原始数据集
            print(f"\n[1/2] 评估原始数据集...")
            original_results = evaluate_format_bias(original_df)
            print_format_bias_report(original_results, f"{ds_name.upper()} - 原始")

            # 评估去偏数据集
            print(f"\n[2/2] 评估去偏数据集...")
            debiased_results = evaluate_format_bias(debiased_df)
            print_format_bias_report(debiased_results, f"{ds_name.upper()} - 去偏后")

            # 验证去偏效果
            validate_debiasing_effect(original_results, debiased_results)

    else:
        # 单独评估模式
        data_dir = args.debiased_dir if os.path.exists(args.debiased_dir) else args.original_dir
        print(f"模式：评估 {data_dir}\n")

        for ds_name in datasets_to_check:
            print(f"\n{'='*70}")
            print(f"评估 {ds_name.upper()} 数据集")
            print(f"{'='*70}")

            # 读取数据
            df = pd.read_csv(
                f"{data_dir}/{ds_name}.csv",
                encoding='utf-8-sig'
            )

            print(f"样本数: {len(df)}条\n")

            # 评估
            results = evaluate_format_bias(df)
            print_format_bias_report(results, ds_name.upper())

    print("\n" + "="*70)
    print("评估完成！")
    print("="*70)


if __name__ == "__main__":
    main()
