"""
分长度区间评估脚本
评估模型在不同文本长度区间的性能，验证模型是否真正学习语义而非长度
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
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import json
from datetime import datetime
from pathlib import Path


def load_model_and_tokenizer(model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """加载训练好的模型和tokenizer"""

    print(f"加载模型: {model_path}", flush=True)
    print(f"设备: {device}", flush=True)

    try:
        # 加载tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

        # 加载模型
        model = BertForSequenceClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()

        print(f"  ✓ 模型加载成功", flush=True)

        return model, tokenizer, device

    except Exception as e:
        print(f"  ✗ 模型加载失败: {e}", flush=True)
        raise


def evaluate_length_bin(model, tokenizer, df_bin, device, batch_size=16, max_length=512):
    """
    评估单个长度区间的性能

    Args:
        model: BERT模型
        tokenizer: BERT tokenizer
        df_bin: 该长度区间的数据
        device: 设备
        batch_size: 批次大小
        max_length: 最大长度

    Returns:
        评估指标字典
    """
    if len(df_bin) == 0:
        return None

    # 准备数据
    texts = df_bin['text'].tolist()
    labels = df_bin['label'].tolist()

    # Tokenization
    encodings = tokenizer(
        texts,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # 创建DataLoader
    dataset = torch.utils.data.TensorDataset(
        encodings['input_ids'],
        encodings['attention_mask'],
        encodings['token_type_ids'],
        torch.tensor(labels, dtype=torch.long)
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 评估
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            token_type_ids = batch[2].to(device)
            batch_labels = batch[3].to(device)

            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy()[:, 1])  # AI类别的概率

    # 计算指标
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    accuracy = accuracy_score(all_labels, all_preds)

    # 分别计算AI和人类的准确率
    ai_mask = all_labels == 1
    human_mask = all_labels == 0

    ai_accuracy = accuracy_score(all_labels[ai_mask], all_preds[ai_mask]) if ai_mask.sum() > 0 else 0
    human_accuracy = accuracy_score(all_labels[human_mask], all_preds[human_mask]) if human_mask.sum() > 0 else 0

    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    return {
        'accuracy': float(accuracy),
        'ai_accuracy': float(ai_accuracy),
        'human_accuracy': float(human_accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': cm.tolist(),
        'sample_count': len(df_bin),
        'ai_count': int(ai_mask.sum()),
        'human_count': int(human_mask.sum())
    }


def evaluate_by_length_bins(
    model_path,
    test_csv='datasets/bert/test.csv',
    bins=[0, 300, 600, 1000, 1500, 2000, 3000, 10000],
    batch_size=16,
    max_length=512,
    output_dir='evaluation_results'
):
    """
    按长度区间评估模型性能

    Args:
        model_path: 训练好的模型路径
        test_csv: 测试集CSV文件路径
        bins: 长度区间边界
        batch_size: 批次大小
        max_length: 最大序列长度
        output_dir: 输出目录

    Returns:
        评估结果字典
    """
    print("=" * 70, flush=True)
    print("分长度区间评估", flush=True)
    print("=" * 70, flush=True)
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    # 1. 加载模型
    print("\n[1/4] 加载模型...", flush=True)
    model, tokenizer, device = load_model_and_tokenizer(model_path)

    # 2. 加载测试集
    print("\n[2/4] 加载测试集...", flush=True)
    test_df = pd.read_csv(test_csv, encoding='utf-8-sig')
    print(f"  ✓ 测试集样本数: {len(test_df)}", flush=True)
    print(f"    AI样本: {(test_df['label']==1).sum()}", flush=True)
    print(f"    人类样本: {(test_df['label']==0).sum()}", flush=True)

    # 3. 按长度分组
    print("\n[3/4] 按长度区间评估...", flush=True)
    print(f"  长度区间: {bins}", flush=True)

    test_df['length_bin'] = pd.cut(test_df['length'], bins=bins, include_lowest=True)

    results = {}
    detailed_results = []

    for i, bin_range in enumerate(test_df['length_bin'].cat.categories):
        bin_df = test_df[test_df['length_bin'] == bin_range]

        if len(bin_df) == 0:
            continue

        print(f"\n  区间 {bin_range}:", flush=True)
        print(f"    样本数: {len(bin_df)}", flush=True)

        # 评估该区间
        metrics = evaluate_length_bin(model, tokenizer, bin_df, device, batch_size, max_length)

        if metrics:
            print(f"    总体准确率: {metrics['accuracy']:.4f}", flush=True)
            print(f"    AI检测准确率: {metrics['ai_accuracy']:.4f}", flush=True)
            print(f"    人类检测准确率: {metrics['human_accuracy']:.4f}", flush=True)
            print(f"    F1分数: {metrics['f1']:.4f}", flush=True)

            # 保存结果
            bin_key = f"{bins[i]}-{bins[i+1]}"
            results[bin_key] = metrics

            detailed_results.append({
                'bin': bin_key,
                'bin_range_str': str(bin_range),
                **metrics
            })

    # 4. 统计分析
    print("\n[4/4] 统计分析...", flush=True)

    # 计算方差（越低越好）
    accuracies = [r['accuracy'] for r in results.values()]
    ai_accuracies = [r['ai_accuracy'] for r in results.values()]
    human_accuracies = [r['human_accuracy'] for r in results.values()]

    variance_overall = np.var(accuracies) if len(accuracies) > 0 else 0
    variance_ai = np.var(ai_accuracies) if len(ai_accuracies) > 0 else 0
    variance_human = np.var(human_accuracies) if len(human_accuracies) > 0 else 0

    print(f"\n性能方差（越低越好）:", flush=True)
    print(f"  总体准确率方差: {variance_overall:.6f}", flush=True)
    print(f"  AI检测准确率方差: {variance_ai:.6f}", flush=True)
    print(f"  人类检测准确率方差: {variance_human:.6f}", flush=True)

    # 判断长度独立性
    if variance_overall < 0.01:
        print("\n  ✓ 优秀！模型对长度不敏感", flush=True)
    elif variance_overall < 0.03:
        print("\n  ✓ 良好，模型对长度依赖较小", flush=True)
    elif variance_overall < 0.05:
        print("\n  ⚠ 中等，模型对长度有一定依赖", flush=True)
    else:
        print("\n  ✗ 较差，模型过度依赖长度特征", flush=True)

    # 计算平均性能
    avg_accuracy = np.mean(accuracies) if len(accuracies) > 0 else 0
    avg_ai_accuracy = np.mean(ai_accuracies) if len(ai_accuracies) > 0 else 0
    avg_human_accuracy = np.mean(human_accuracies) if len(human_accuracies) > 0 else 0

    print(f"\n平均性能:", flush=True)
    print(f"  总体准确率: {avg_accuracy:.4f}", flush=True)
    print(f"  AI检测准确率: {avg_ai_accuracy:.4f}", flush=True)
    print(f"  人类检测准确率: {avg_human_accuracy:.4f}", flush=True)

    # 5. 保存结果
    print("\n" + "=" * 70, flush=True)
    print("保存结果", flush=True)
    print("=" * 70, flush=True)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 保存详细结果
    results_file = os.path.join(output_dir, 'length_aware_evaluation.json')
    full_results = {
        'evaluation_date': datetime.now().isoformat(),
        'model_path': model_path,
        'test_csv': test_csv,
        'length_bins': bins,
        'bin_results': detailed_results,
        'statistics': {
            'variance_overall': float(variance_overall),
            'variance_ai': float(variance_ai),
            'variance_human': float(variance_human),
            'avg_accuracy': float(avg_accuracy),
            'avg_ai_accuracy': float(avg_ai_accuracy),
            'avg_human_accuracy': float(avg_human_accuracy)
        }
    }

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, ensure_ascii=False, indent=2)

    print(f"  ✓ 详细结果已保存: {results_file}", flush=True)

    # 保存CSV格式（便于分析）
    df_results = pd.DataFrame(detailed_results)
    csv_file = os.path.join(output_dir, 'length_aware_evaluation.csv')
    df_results.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"  ✓ CSV结果已保存: {csv_file}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("评估完成！", flush=True)
    print("=" * 70, flush=True)

    return full_results


def create_evaluation_report(results_file):
    """根据评估结果生成可读报告"""

    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    report = []
    report.append("=" * 70)
    report.append("分长度区间评估报告")
    report.append("=" * 70)
    report.append(f"\n评估时间: {results['evaluation_date']}")
    report.append(f"模型路径: {results['model_path']}")
    report.append(f"测试集: {results['test_csv']}")

    report.append("\n" + "=" * 70)
    report.append("各长度区间性能")
    report.append("=" * 70)

    for bin_result in results['bin_results']:
        report.append(f"\n长度区间: {bin_result['bin']}")
        report.append(f"  样本数: {bin_result['sample_count']} (AI:{bin_result['ai_count']}, 人类:{bin_result['human_count']})")
        report.append(f"  总体准确率: {bin_result['accuracy']:.4f}")
        report.append(f"  AI检测准确率: {bin_result['ai_accuracy']:.4f}")
        report.append(f"  人类检测准确率: {bin_result['human_accuracy']:.4f}")
        report.append(f"  Precision: {bin_result['precision']:.4f}")
        report.append(f"  Recall: {bin_result['recall']:.4f}")
        report.append(f"  F1: {bin_result['f1']:.4f}")

    stats = results['statistics']
    report.append("\n" + "=" * 70)
    report.append("统计分析")
    report.append("=" * 70)
    report.append(f"\n性能方差（越低越好）:")
    report.append(f"  总体准确率方差: {stats['variance_overall']:.6f}")
    report.append(f"  AI检测准确率方差: {stats['variance_ai']:.6f}")
    report.append(f"  人类检测准确率方差: {stats['variance_human']:.6f}")

    report.append(f"\n平均性能:")
    report.append(f"  总体准确率: {stats['avg_accuracy']:.4f}")
    report.append(f"  AI检测准确率: {stats['avg_ai_accuracy']:.4f}")
    report.append(f"  人类检测准确率: {stats['avg_human_accuracy']:.4f}")

    report.append("\n" + "=" * 70)

    report_text = '\n'.join(report)
    return report_text


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='分长度区间评估BERT模型')
    parser.add_argument('--model_path', type=str, required=True, help='训练好的模型路径')
    parser.add_argument('--test_csv', type=str, default='datasets/bert/test.csv', help='测试集CSV路径')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='输出目录')

    args = parser.parse_args()

    # 运行评估
    results = evaluate_by_length_bins(
        model_path=args.model_path,
        test_csv=args.test_csv,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )

    # 生成报告
    results_file = os.path.join(args.output_dir, 'length_aware_evaluation.json')
    report = create_evaluation_report(results_file)

    # 保存报告
    report_file = os.path.join(args.output_dir, 'evaluation_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n详细报告已保存: {report_file}", flush=True)
    print("\n" + report)
