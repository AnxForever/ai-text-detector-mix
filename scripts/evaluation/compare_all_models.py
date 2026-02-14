#!/usr/bin/env python3
"""
对比所有模型的性能
- 测试集准确率
- 问题样本准确率（正式人类文本）
"""

import os
import sys
import torch
import pandas as pd
from pathlib import Path
from transformers import BertTokenizer, BertForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 所有模型路径
MODELS = [
    "bert_improved/best_model",
    "bert_v2_with_sep",
    "bert_v2_balanced",
    "bert_v2_overnight",
    "bert_v3_core_v2",
    "bert_v3_fresh",
    "bert_v3_defense",
    "bert_v4_core_v3",
    "bert_v4_defense_focused",
]

# 问题样本：正式人类文本（应该被识别为Human）
PROBLEM_SAMPLES_HUMAN = [
    ("温馨提示", "温馨提示：明天上午9点将进行消防演练，请各位同事提前做好准备，保持通道畅通。"),
    ("日常分享", "今天天气真好，中午和同事一起去公园散步了，心情舒畅。"),
    ("推荐分享", "这家店的牛肉面真的很不错，汤底浓郁，面条筋道，强烈推荐大家去尝尝。"),
    ("工作日常", "今天终于把报告写完了，感觉整个人都轻松了。"),
    ("学术摘要", "本文研究了深度学习在自然语言处理中的应用，通过实验验证了模型的有效性。"),
    ("实验结论", "实验结果表明，改进后的算法在准确率上提升了5%，验证了本文方法的可行性。"),
    ("论文结尾", "综上所述，本研究为该领域提供了新的思路和方法，具有一定的理论和实践意义。"),
    ("地方新闻", "昨日，我市召开经济工作会议，市长强调要加快产业升级，推动高质量发展。"),
    ("考试通知", "关于2024年期末考试安排的通知：考试时间为1月15日至1月20日，请同学们认真复习。"),
    ("学生邮件", "尊敬的老师：您好！我是计算机系大三学生张明，想就毕业论文选题向您请教。"),
]

# 问题样本：网络风格AI文本（应该被识别为AI）
PROBLEM_SAMPLES_AI = [
    ("犹豫式AI", "额...这个问题嘛...让我想想啊，好像是这样的...不对不对，应该是那样..."),
    ("网络语AI", "老铁们这个东西绝绝子！yyds！真的太绝了，建议大家冲冲冲！"),
]

def load_model(model_path):
    """加载模型"""
    try:
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)
        model.eval()
        return tokenizer, model
    except Exception as e:
        return None, None

def predict(tokenizer, model, text, device='cpu'):
    """预测单个文本"""
    model = model.to(device)
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        ai_prob = probs[0][1].item()

    return ai_prob

def evaluate_on_test_set(tokenizer, model, test_path, device='cpu'):
    """在测试集上评估"""
    if not test_path.exists():
        return None, None, None

    df = pd.read_csv(test_path, encoding='utf-8-sig')
    if 'text' not in df.columns or 'label' not in df.columns:
        return None, None, None

    # 只取前500条加速测试
    df = df.head(500)

    correct = 0
    total = len(df)
    human_correct = 0
    human_total = 0
    ai_correct = 0
    ai_total = 0

    for _, row in df.iterrows():
        text = str(row['text'])
        label = int(row['label'])

        ai_prob = predict(tokenizer, model, text, device)
        pred = 1 if ai_prob >= 0.5 else 0

        if pred == label:
            correct += 1

        if label == 0:
            human_total += 1
            if pred == 0:
                human_correct += 1
        else:
            ai_total += 1
            if pred == 1:
                ai_correct += 1

    accuracy = correct / total if total > 0 else 0
    human_acc = human_correct / human_total if human_total > 0 else 0
    ai_acc = ai_correct / ai_total if ai_total > 0 else 0

    return accuracy, human_acc, ai_acc

def evaluate_problem_samples(tokenizer, model, device='cpu'):
    """评估问题样本"""
    # Human样本（应该预测为Human，即AI概率<0.5）
    human_correct = 0
    human_results = []
    for name, text in PROBLEM_SAMPLES_HUMAN:
        ai_prob = predict(tokenizer, model, text, device)
        is_correct = ai_prob < 0.5
        if is_correct:
            human_correct += 1
        human_results.append((name, ai_prob, is_correct))

    # AI样本（应该预测为AI，即AI概率>=0.5）
    ai_correct = 0
    ai_results = []
    for name, text in PROBLEM_SAMPLES_AI:
        ai_prob = predict(tokenizer, model, text, device)
        is_correct = ai_prob >= 0.5
        if is_correct:
            ai_correct += 1
        ai_results.append((name, ai_prob, is_correct))

    human_acc = human_correct / len(PROBLEM_SAMPLES_HUMAN) if PROBLEM_SAMPLES_HUMAN else 0
    ai_acc = ai_correct / len(PROBLEM_SAMPLES_AI) if PROBLEM_SAMPLES_AI else 0
    total_acc = (human_correct + ai_correct) / (len(PROBLEM_SAMPLES_HUMAN) + len(PROBLEM_SAMPLES_AI))

    return {
        'human_acc': human_acc,
        'ai_acc': ai_acc,
        'total_acc': total_acc,
        'human_results': human_results,
        'ai_results': ai_results
    }

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    print("=" * 100)

    # 测试集路径
    test_paths = [
        PROJECT_ROOT / "datasets/active/core_v1/test.csv",
        PROJECT_ROOT / "datasets/active/core_v2/test.csv",
        PROJECT_ROOT / "datasets/active/core_v3/test.csv",
    ]

    # 找到存在的测试集
    test_path = None
    for p in test_paths:
        if p.exists():
            test_path = p
            break

    if test_path:
        print(f"测试集: {test_path}")
    else:
        print("警告: 未找到测试集")

    print("=" * 100)

    results = []

    for model_name in MODELS:
        model_path = PROJECT_ROOT / "models" / model_name

        if not model_path.exists():
            print(f"⏭️  {model_name}: 模型不存在")
            continue

        print(f"\n📊 测试模型: {model_name}")
        print("-" * 80)

        tokenizer, model = load_model(model_path)
        if tokenizer is None:
            print(f"   ❌ 加载失败")
            continue

        # 测试集评估
        if test_path:
            test_acc, test_human_acc, test_ai_acc = evaluate_on_test_set(tokenizer, model, test_path, device)
            if test_acc is not None:
                print(f"   测试集准确率: {test_acc*100:.2f}% (Human: {test_human_acc*100:.1f}%, AI: {test_ai_acc*100:.1f}%)")
            else:
                test_acc = test_human_acc = test_ai_acc = None
        else:
            test_acc = test_human_acc = test_ai_acc = None

        # 问题样本评估
        prob_results = evaluate_problem_samples(tokenizer, model, device)
        print(f"   问题样本准确率: {prob_results['total_acc']*100:.1f}% (Human: {prob_results['human_acc']*100:.1f}%, AI: {prob_results['ai_acc']*100:.1f}%)")

        # 详细结果
        print(f"\n   [Human样本详情] (应该<50%)")
        for name, ai_prob, is_correct in prob_results['human_results']:
            mark = "✓" if is_correct else "✗"
            print(f"      {mark} {name}: {ai_prob*100:.1f}% AI")

        print(f"\n   [AI样本详情] (应该>=50%)")
        for name, ai_prob, is_correct in prob_results['ai_results']:
            mark = "✓" if is_correct else "✗"
            print(f"      {mark} {name}: {ai_prob*100:.1f}% AI")

        results.append({
            'model': model_name,
            'test_acc': test_acc,
            'test_human_acc': test_human_acc,
            'test_ai_acc': test_ai_acc,
            'problem_total_acc': prob_results['total_acc'],
            'problem_human_acc': prob_results['human_acc'],
            'problem_ai_acc': prob_results['ai_acc'],
        })

    # 汇总表格
    print("\n" + "=" * 100)
    print("📋 模型对比汇总")
    print("=" * 100)
    print(f"{'模型':<30} {'测试集':<12} {'Human':<10} {'AI':<10} {'问题样本':<12} {'正式Human':<10} {'网络AI':<10}")
    print("-" * 100)

    for r in results:
        test_acc_str = f"{r['test_acc']*100:.1f}%" if r['test_acc'] else "N/A"
        test_human_str = f"{r['test_human_acc']*100:.1f}%" if r['test_human_acc'] else "N/A"
        test_ai_str = f"{r['test_ai_acc']*100:.1f}%" if r['test_ai_acc'] else "N/A"

        print(f"{r['model']:<30} {test_acc_str:<12} {test_human_str:<10} {test_ai_str:<10} "
              f"{r['problem_total_acc']*100:.1f}%{'':<7} {r['problem_human_acc']*100:.1f}%{'':<5} {r['problem_ai_acc']*100:.1f}%")

    # 找最佳模型
    if results:
        # 按问题样本准确率排序
        sorted_by_problem = sorted(results, key=lambda x: x['problem_total_acc'], reverse=True)
        print(f"\n🏆 问题样本最佳模型: {sorted_by_problem[0]['model']} ({sorted_by_problem[0]['problem_total_acc']*100:.1f}%)")

        # 按测试集准确率排序
        sorted_by_test = sorted([r for r in results if r['test_acc'] is not None],
                                key=lambda x: x['test_acc'], reverse=True)
        if sorted_by_test:
            print(f"🏆 测试集最佳模型: {sorted_by_test[0]['model']} ({sorted_by_test[0]['test_acc']*100:.1f}%)")

if __name__ == "__main__":
    main()
