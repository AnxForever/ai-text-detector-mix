"""
格式对抗测试工具

功能：
1. 测试模型对格式变化的鲁棒性
2. 验证模型是否学习了语义而非格式特征
3. 4种对抗场景全面测试

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
import torch
import argparse
from datetime import datetime
from typing import Dict, Tuple
from transformers import BertForSequenceClassification, BertTokenizer

# 导入格式处理函数
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data_cleaning'))
from format_handler import (
    remove_markdown_comprehensive,
    has_markdown,
    has_markdown_detailed,
    get_format_statistics
)


def evaluate_model(model, df: pd.DataFrame, tokenizer, device='cuda') -> Dict:
    """
    评估模型在数据集上的性能

    Args:
        model: BERT模型
        df: 测试数据框
        tokenizer: Tokenizer
        device: 设备

    Returns:
        评估结果字典
    """
    model.eval()
    predictions = []
    labels = df['label'].tolist()

    with torch.no_grad():
        for text in df['text']:
            # 确保text是字符串类型
            if pd.isna(text):
                text = ""
            text = str(text)

            encoding = tokenizer(
                text,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            # 移动到设备
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pred = torch.argmax(outputs.logits, dim=1).item()
            predictions.append(pred)

    # 计算准确率
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    accuracy = correct / len(labels)

    # 计算混淆矩阵
    tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
    tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)
    fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': {
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
        },
        'predictions': predictions
    }


def add_markdown_natural(text: str) -> str:
    """
    为文本自然地添加markdown格式

    策略：
    - 20%概率添加标题
    - 30%概率添加加粗
    - 25%概率转换为列表
    - 保持简单，避免过度格式化
    """
    import random

    # 确保text是字符串类型
    if pd.isna(text):
        return ""
    text = str(text)

    lines = text.split('\n')
    formatted_lines = []

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            formatted_lines.append('')
            continue

        # 第一行有20%概率变成标题
        if i == 0 and random.random() < 0.20:
            line = f"## {line}"

        # 短句有30%概率加粗关键词
        elif len(line) < 50 and random.random() < 0.30:
            words = line.split()
            if len(words) >= 3:
                # 加粗前几个词
                words[0] = f"**{words[0]}**"
                line = ' '.join(words)

        # 如果是逗号分隔的内容，25%概率转为列表
        elif ',' in line and random.random() < 0.25:
            items = [item.strip() for item in line.split(',') if item.strip()]
            if len(items) >= 2:
                line = '\n'.join(f"- {item}" for item in items)

        formatted_lines.append(line)

    return '\n'.join(formatted_lines)


def format_adversarial_test(model, test_df: pd.DataFrame, tokenizer, device='cuda') -> Dict:
    """
    格式对抗测试

    测试场景：
    1. 去除所有markdown（测试纯文本检测）
    2. 添加所有markdown（测试格式干扰）
    3. 格式交换（AI去格式，人类加格式）
    4. 随机格式（随机添加/删除）

    Args:
        model: BERT模型
        test_df: 测试数据集
        tokenizer: Tokenizer
        device: 设备

    Returns:
        对抗测试结果
    """

    print("="*70)
    print("格式对抗测试")
    print("="*70)
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 原始准确率
    print("[0/4] 评估原始测试集（基线）...")
    original_results = evaluate_model(model, test_df, tokenizer, device)
    original_acc = original_results['accuracy']

    print(f"  基线准确率: {original_acc*100:.2f}%")
    print(f"  基线F1分数: {original_results['f1']*100:.2f}%")

    # 测试1：去除所有markdown
    print("\n" + "="*70)
    print("[1/4] 测试1：去除所有markdown（纯文本检测）")
    print("="*70)
    print("目的：验证模型是否能检测无格式的AI文本\n")

    plain_df = test_df.copy()
    print("  正在去除所有文本的markdown...")
    plain_df['text'] = plain_df['text'].apply(remove_markdown_comprehensive)

    # 统计格式去除效果
    original_md_rate = test_df['text'].apply(has_markdown).mean()
    plain_md_rate = plain_df['text'].apply(has_markdown).mean()
    print(f"  原始markdown率: {original_md_rate*100:.1f}%")
    print(f"  去除后markdown率: {plain_md_rate*100:.1f}%")
    print(f"  格式去除效果: {(original_md_rate - plain_md_rate)*100:.1f}%\n")

    print("  评估模型性能...")
    plain_results = evaluate_model(model, plain_df, tokenizer, device)
    plain_acc = plain_results['accuracy']
    plain_drop = plain_acc - original_acc

    print(f"  准确率: {plain_acc*100:.2f}%")
    print(f"  变化: {plain_drop*100:+.2f}% ({'✅' if abs(plain_drop) < 0.05 else '⚠️'})")

    # 测试2：添加所有markdown
    print("\n" + "="*70)
    print("[2/4] 测试2：为所有文本添加markdown（格式干扰）")
    print("="*70)
    print("目的：验证格式不会误导模型判断\n")

    formatted_df = test_df.copy()
    print("  正在为所有文本添加markdown...")

    # 只为纯文本添加markdown
    formatted_df['text'] = formatted_df['text'].apply(
        lambda x: add_markdown_natural(x) if not has_markdown(x) else x
    )

    # 统计格式添加效果
    formatted_md_rate = formatted_df['text'].apply(has_markdown).mean()
    print(f"  原始markdown率: {original_md_rate*100:.1f}%")
    print(f"  添加后markdown率: {formatted_md_rate*100:.1f}%")
    print(f"  格式增加: {(formatted_md_rate - original_md_rate)*100:+.1f}%\n")

    print("  评估模型性能...")
    formatted_results = evaluate_model(model, formatted_df, tokenizer, device)
    formatted_acc = formatted_results['accuracy']
    formatted_drop = formatted_acc - original_acc

    print(f"  准确率: {formatted_acc*100:.2f}%")
    print(f"  变化: {formatted_drop*100:+.2f}% ({'✅' if abs(formatted_drop) < 0.05 else '⚠️'})")

    # 测试3：格式交换（最关键）
    print("\n" + "="*70)
    print("[3/4] 测试3：格式交换（AI→纯文本，人类→markdown）")
    print("="*70)
    print("目的：验证模型不依赖格式判断类别\n")

    swapped_df = test_df.copy()
    print("  AI文本：去除markdown...")
    ai_mask = swapped_df['label'] == 1
    swapped_df.loc[ai_mask, 'text'] = swapped_df.loc[ai_mask, 'text'].apply(
        remove_markdown_comprehensive
    )

    print("  人类文本：添加markdown...")
    human_mask = swapped_df['label'] == 0
    swapped_df.loc[human_mask, 'text'] = swapped_df.loc[human_mask, 'text'].apply(
        add_markdown_natural
    )

    # 统计交换效果
    ai_md_before = test_df[test_df['label']==1]['text'].apply(has_markdown).mean()
    ai_md_after = swapped_df[swapped_df['label']==1]['text'].apply(has_markdown).mean()
    human_md_before = test_df[test_df['label']==0]['text'].apply(has_markdown).mean()
    human_md_after = swapped_df[swapped_df['label']==0]['text'].apply(has_markdown).mean()

    print(f"  AI文本markdown率: {ai_md_before*100:.1f}% → {ai_md_after*100:.1f}%")
    print(f"  人类文本markdown率: {human_md_before*100:.1f}% → {human_md_after*100:.1f}%\n")

    print("  评估模型性能...")
    swapped_results = evaluate_model(model, swapped_df, tokenizer, device)
    swapped_acc = swapped_results['accuracy']
    swapped_drop = swapped_acc - original_acc

    print(f"  准确率: {swapped_acc*100:.2f}%")
    print(f"  变化: {swapped_drop*100:+.2f}% ({'✅' if abs(swapped_drop) < 0.10 else '⚠️'})")

    # 测试4：随机格式
    print("\n" + "="*70)
    print("[4/4] 测试4：随机格式变化")
    print("="*70)
    print("目的：验证模型在格式不确定时的稳定性\n")

    import random
    random.seed(42)

    random_df = test_df.copy()
    print("  随机对50%文本应用格式变化...")

    for idx in random_df.index:
        if random.random() < 0.5:
            # 50%概率去除格式
            random_df.at[idx, 'text'] = remove_markdown_comprehensive(
                random_df.at[idx, 'text']
            )
        else:
            # 50%概率添加格式（如果原本无格式）
            if not has_markdown(random_df.at[idx, 'text']):
                random_df.at[idx, 'text'] = add_markdown_natural(
                    random_df.at[idx, 'text']
                )

    random_md_rate = random_df['text'].apply(has_markdown).mean()
    print(f"  原始markdown率: {original_md_rate*100:.1f}%")
    print(f"  随机后markdown率: {random_md_rate*100:.1f}%\n")

    print("  评估模型性能...")
    random_results = evaluate_model(model, random_df, tokenizer, device)
    random_acc = random_results['accuracy']
    random_drop = random_acc - original_acc

    print(f"  准确率: {random_acc*100:.2f}%")
    print(f"  变化: {random_drop*100:+.2f}% ({'✅' if abs(random_drop) < 0.05 else '⚠️'})")

    # 汇总结果
    print("\n" + "="*70)
    print("格式免疫性评估")
    print("="*70)

    max_drop = max(
        abs(plain_drop),
        abs(formatted_drop),
        abs(swapped_drop),
        abs(random_drop)
    )

    print(f"\n【性能汇总】")
    print(f"基线准确率:     {original_acc*100:.2f}%")
    print(f"纯文本测试:     {plain_acc*100:.2f}% ({plain_drop*100:+.2f}%)")
    print(f"格式化测试:     {formatted_acc*100:.2f}% ({formatted_drop*100:+.2f}%)")
    print(f"格式交换测试:   {swapped_acc*100:.2f}% ({swapped_drop*100:+.2f}%)")
    print(f"随机格式测试:   {random_acc*100:.2f}% ({random_drop*100:+.2f}%)")
    print(f"\n最大下降幅度: {max_drop*100:.2f}%")

    # 评级
    print("\n【格式免疫性评级】")
    if max_drop < 0.05:
        grade = "优秀"
        emoji = "✅"
        comment = "模型对格式变化完全免疫（下降<5%）"
        recommendation = "模型已经成功学习语义特征，可以部署使用"
    elif max_drop < 0.10:
        grade = "良好"
        emoji = "✅"
        comment = "模型对格式变化较为鲁棒（下降<10%）"
        recommendation = "模型主要基于语义判断，可以使用"
    elif max_drop < 0.20:
        grade = "中等"
        emoji = "⚠️"
        comment = "模型仍然部分依赖格式特征（下降10-20%）"
        recommendation = "建议进一步增强数据去偏或增加训练轮次"
    else:
        grade = "不合格"
        emoji = "🔴"
        comment = "模型严重依赖格式特征（下降>20%）"
        recommendation = "需要重新检查数据集格式分布或训练策略"

    print(f"{emoji} 评级: {grade}")
    print(f"   {comment}")
    print(f"\n建议: {recommendation}")

    return {
        'original': original_results,
        'plain': plain_results,
        'formatted': formatted_results,
        'swapped': swapped_results,
        'random': random_results,
        'max_drop': max_drop,
        'grade': grade
    }


def main():
    parser = argparse.ArgumentParser(description='格式对抗测试工具')
    parser.add_argument('--model-dir', type=str, default='models/bert_debiased/best_model',
                       help='模型目录')
    parser.add_argument('--test-file', type=str, default='datasets/bert_debiased/test.csv',
                       help='测试数据文件')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (cuda/cpu)')

    args = parser.parse_args()

    print("="*70)
    print("格式对抗测试工具")
    print("="*70)
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️ CUDA不可用，切换到CPU")
        args.device = 'cpu'

    print(f"使用设备: {args.device}\n")

    # 加载模型
    print("正在加载模型...")
    try:
        model = BertForSequenceClassification.from_pretrained(args.model_dir)
        tokenizer = BertTokenizer.from_pretrained(args.model_dir)
        model.to(args.device)
        model.eval()
        print(f"✓ 模型加载成功: {args.model_dir}\n")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print(f"   请确认模型路径: {args.model_dir}")
        return

    # 加载测试数据
    print("正在加载测试数据...")
    try:
        test_df = pd.read_csv(args.test_file, encoding='utf-8-sig')
        print(f"✓ 测试集加载成功: {len(test_df)}条\n")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    # 运行对抗测试
    results = format_adversarial_test(model, test_df, tokenizer, args.device)

    print("\n" + "="*70)
    print("测试完成！")
    print("="*70)


if __name__ == "__main__":
    main()
