"""
单文本检测工具 - 用于测试模型

功能：
- 输入任意文本，模型判断是AI生成还是人类撰写
- 显示预测结果和置信度
- 支持批量测试
- 跨平台支持：Windows PowerShell / Linux / macOS

作者：AI文本检测项目组
日期：2026-01-11
"""

import sys
import os
import platform
import torch
import argparse
from transformers import BertForSequenceClassification, BertTokenizer
import warnings
warnings.filterwarnings('ignore')

# 设置UTF-8编码
os.environ['PYTHONIOENCODING'] = 'utf-8'


def print_platform_instructions():
    """打印特定平台的使用说明"""
    system = platform.system()

    print("=" * 70)
    print("AI文本检测工具 - 跨平台版本")
    print("=" * 70)
    print(f"\n检测到系统: {system}")

    if system == "Windows":
        print("\n【Windows PowerShell 使用说明】")
        print("\n1. 激活虚拟环境:")
        print("   .venv\\Scripts\\Activate.ps1")
        print("\n2. 设置离线模式 (如果需要):")
        print("   $env:HF_HUB_OFFLINE=1")
        print("   $env:TRANSFORMERS_OFFLINE=1")
        print("\n3. 运行测试:")
        print("   python scripts\\evaluation\\test_single_text.py --interactive")

    elif system == "Linux" or system == "Darwin":  # Darwin = macOS
        print("\n【Linux/macOS 使用说明】")
        print("\n1. 激活虚拟环境:")
        print("   source .venv/bin/activate")
        print("\n2. 设置离线模式 (如果需要):")
        print("   export HF_HUB_OFFLINE=1")
        print("   export TRANSFORMERS_OFFLINE=1")
        print("\n3. 运行测试:")
        print("   python3 scripts/evaluation/test_single_text.py --interactive")

    print("\n" + "=" * 70)
    print("")


def load_model(model_path, device='cuda'):
    """加载训练好的模型"""
    print(f"正在加载模型: {model_path}")

    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)

    model.to(device)
    model.eval()

    print(f"✓ 模型加载成功")
    print(f"  设备: {device}")
    return model, tokenizer


def predict_text(text, model, tokenizer, device='cuda'):
    """
    预测单个文本

    Returns:
        prediction: 0=人类, 1=AI
        confidence: 置信度 (0-1)
        probabilities: [人类概率, AI概率]
    """
    # 确保text是字符串
    if not isinstance(text, str):
        text = str(text)

    # Tokenize
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

    # 预测
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)[0]
        prediction = torch.argmax(logits, dim=1).item()
        confidence = probabilities[prediction].item()

    return prediction, confidence, probabilities.cpu().numpy()


def format_result(text, prediction, confidence, probabilities):
    """格式化输出结果"""
    label_map = {0: "人类撰写", 1: "AI生成"}

    print("\n" + "="*70)
    print("检测结果")
    print("="*70)

    print(f"\n【待检测文本】")
    print(f"{text[:200]}{'...' if len(text) > 200 else ''}")
    print(f"\n文本长度: {len(text)} 字符")

    print(f"\n【预测结果】")
    print(f"判断: {label_map[prediction]}")
    print(f"置信度: {confidence*100:.2f}%")

    print(f"\n【详细概率】")
    print(f"  人类撰写: {probabilities[0]*100:.2f}%")
    print(f"  AI生成:   {probabilities[1]*100:.2f}%")

    # 置信度解释
    print(f"\n【置信度解释】")
    if confidence >= 0.95:
        print(f"  ✅ 非常确定 (>95%)")
    elif confidence >= 0.80:
        print(f"  ✅ 比较确定 (80-95%)")
    elif confidence >= 0.60:
        print(f"  ⚠️ 有一定把握 (60-80%)")
    else:
        print(f"  ❓ 不太确定 (<60%)")

    print("="*70)


def interactive_mode(model, tokenizer, device):
    """交互模式：持续输入文本进行检测"""
    print("\n" + "="*70)
    print("交互检测模式")
    print("="*70)
    print("\n输入文本进行检测，输入 'quit' 或 'exit' 退出")
    print("支持多行输入，输入 'END' 结束当前文本\n")

    while True:
        print("-"*70)
        print("请输入待检测文本（多行输入请以 END 结束）:")

        lines = []
        while True:
            try:
                line = input()
                if line.strip().upper() == 'END':
                    break
                if line.strip().lower() in ['quit', 'exit']:
                    print("\n退出检测程序")
                    return
                lines.append(line)
            except EOFError:
                break

        text = '\n'.join(lines).strip()

        if not text:
            print("⚠️ 文本为空，请重新输入")
            continue

        if text.lower() in ['quit', 'exit']:
            print("\n退出检测程序")
            return

        # 预测
        prediction, confidence, probabilities = predict_text(
            text, model, tokenizer, device
        )

        # 显示结果
        format_result(text, prediction, confidence, probabilities)


def batch_test_mode(texts, model, tokenizer, device):
    """批量测试模式"""
    print("\n" + "="*70)
    print(f"批量检测模式 - 共 {len(texts)} 个文本")
    print("="*70)

    results = []

    for i, text in enumerate(texts, 1):
        print(f"\n[{i}/{len(texts)}] 检测中...")

        prediction, confidence, probabilities = predict_text(
            text, model, tokenizer, device
        )

        results.append({
            'text': text,
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': probabilities
        })

        format_result(text, prediction, confidence, probabilities)

    # 汇总统计
    print("\n" + "="*70)
    print("批量检测汇总")
    print("="*70)

    ai_count = sum(1 for r in results if r['prediction'] == 1)
    human_count = len(results) - ai_count
    avg_confidence = sum(r['confidence'] for r in results) / len(results)

    print(f"\n总数: {len(results)}")
    print(f"判断为AI生成: {ai_count} ({ai_count/len(results)*100:.1f}%)")
    print(f"判断为人类撰写: {human_count} ({human_count/len(results)*100:.1f}%)")
    print(f"平均置信度: {avg_confidence*100:.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description='AI文本检测工具 - 跨平台版本',
        epilog='支持 Windows PowerShell / Linux / macOS'
    )
    parser.add_argument('--model-dir', type=str,
                       default='models/bert_improved/best_model',
                       help='模型目录路径')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='运行设备')
    parser.add_argument('--text', type=str, default=None,
                       help='待检测的文本（单个）')
    parser.add_argument('--file', type=str, default=None,
                       help='包含待检测文本的文件（每行一个文本）')
    parser.add_argument('--interactive', action='store_true',
                       help='交互模式（持续输入）')
    parser.add_argument('--help-platform', action='store_true',
                       help='显示平台特定的使用说明')

    args = parser.parse_args()

    # 如果请求平台帮助
    if args.help_platform:
        print_platform_instructions()
        return

    # 检查CUDA可用性
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️ CUDA不可用，切换到CPU")
        args.device = 'cpu'

    # 加载模型
    try:
        model, tokenizer = load_model(args.model_dir, args.device)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 根据不同模式运行
    if args.interactive:
        # 交互模式
        interactive_mode(model, tokenizer, args.device)

    elif args.text:
        # 单文本模式
        prediction, confidence, probabilities = predict_text(
            args.text, model, tokenizer, args.device
        )
        format_result(args.text, prediction, confidence, probabilities)

    elif args.file:
        # 文件批量模式
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]

            if not texts:
                print("❌ 文件为空或格式错误")
                return

            batch_test_mode(texts, model, tokenizer, args.device)

        except Exception as e:
            print(f"❌ 文件读取失败: {e}")
            return

    else:
        # 默认：交互模式
        print("\n未指定检测内容，启动交互模式")
        interactive_mode(model, tokenizer, args.device)


if __name__ == "__main__":
    main()
