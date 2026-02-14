#!/usr/bin/env python3
"""
跨模型泛化测试 - 用 M4 数据集测试模型对不同 AI 来源的检测能力

测试内容:
1. ChatGPT 生成文本 (训练集中有)
2. Davinci 生成文本 (训练集中无/少)
3. Human 文本 (对照组)

目标: 验证模型是否过拟合到 ChatGPT 特征
"""
import os
import sys
import json
import torch
import random
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

PROJECT_ROOT = Path(__file__).parent.parent.parent
random.seed(42)


def load_m4_data(sample_size=500):
    """加载 M4 数据集"""
    data_dir = PROJECT_ROOT / 'datasets' / 'external' / 'M4' / 'data'

    chatgpt_file = data_dir / 'qazh_chatgpt.jsonl'
    davinci_file = data_dir / 'qazh_davinci.jsonl'

    results = {
        'chatgpt_ai': [],
        'davinci_ai': [],
        'human': [],
    }

    # 加载 ChatGPT AI 文本
    with open(chatgpt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        random.shuffle(lines)
        for line in lines[:sample_size]:
            item = json.loads(line)
            if item.get('machine_text') and len(item['machine_text'].strip()) >= 50:
                results['chatgpt_ai'].append({
                    'text': item['machine_text'].strip(),
                    'source': 'chatgpt',
                    'label': 1,
                })
            # 同时收集 human 文本
            if item.get('human_text') and len(item['human_text'].strip()) >= 50:
                results['human'].append({
                    'text': item['human_text'].strip(),
                    'source': 'human_m4',
                    'label': 0,
                })

    # 加载 Davinci AI 文本
    with open(davinci_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        random.shuffle(lines)
        for line in lines[:sample_size]:
            item = json.loads(line)
            if item.get('machine_text') and len(item['machine_text'].strip()) >= 50:
                results['davinci_ai'].append({
                    'text': item['machine_text'].strip(),
                    'source': 'davinci',
                    'label': 1,
                })

    # 限制样本数量
    for key in results:
        results[key] = results[key][:sample_size]

    return results


def load_model(model_name):
    """加载模型"""
    from transformers import BertTokenizer, BertForSequenceClassification

    model_path = str(PROJECT_ROOT / 'models' / model_name)
    if not os.path.exists(model_path):
        print(f"[ERROR] 模型不存在: {model_path}")
        return None, None, None

    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    return tokenizer, model, device


def predict_batch(texts, tokenizer, model, device, max_len=256, batch_size=16):
    """批量预测"""
    all_preds = []
    all_confs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]

        encoding = tokenizer(
            batch_texts,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            confs = probs.gather(1, preds.unsqueeze(1)).squeeze()

        all_preds.extend(preds.cpu().tolist())
        # 处理单样本情况
        confs_list = confs.cpu().tolist()
        if isinstance(confs_list, float):
            confs_list = [confs_list]
        all_confs.extend(confs_list)

    return all_preds, all_confs


def evaluate_category(data_list, tokenizer, model, device):
    """评估单个类别"""
    if not data_list:
        return {'accuracy': 0, 'count': 0}

    texts = [item['text'] for item in data_list]
    labels = [item['label'] for item in data_list]

    preds, confs = predict_batch(texts, tokenizer, model, device)

    correct = sum(1 for p, l in zip(preds, labels) if p == l)
    accuracy = correct / len(labels)

    # 置信度统计
    correct_confs = [c for p, l, c in zip(preds, labels, confs) if p == l]
    wrong_confs = [c for p, l, c in zip(preds, labels, confs) if p != l]

    # 长度分布
    length_buckets = defaultdict(lambda: {'correct': 0, 'total': 0})
    for item, pred in zip(data_list, preds):
        text_len = len(item['text'])
        if text_len < 100:
            bucket = '<100'
        elif text_len < 200:
            bucket = '100-200'
        elif text_len < 300:
            bucket = '200-300'
        elif text_len < 500:
            bucket = '300-500'
        else:
            bucket = '500+'

        length_buckets[bucket]['total'] += 1
        if pred == item['label']:
            length_buckets[bucket]['correct'] += 1

    return {
        'accuracy': accuracy,
        'count': len(labels),
        'correct': correct,
        'avg_conf_correct': sum(correct_confs) / len(correct_confs) if correct_confs else 0,
        'avg_conf_wrong': sum(wrong_confs) / len(wrong_confs) if wrong_confs else 0,
        'length_buckets': dict(length_buckets),
    }


def main():
    print("=" * 70)
    print("跨模型泛化测试 - M4 数据集")
    print("=" * 70)

    # 加载数据
    print("\n加载 M4 数据...")
    data = load_m4_data(sample_size=500)

    print(f"  ChatGPT AI: {len(data['chatgpt_ai'])} 条")
    print(f"  Davinci AI: {len(data['davinci_ai'])} 条")
    print(f"  Human: {len(data['human'])} 条")

    # 测试的模型列表
    models_to_test = [
        'bert_v3_core_v2',
        'bert_v4_defense_focused',
    ]

    results = {}

    for model_name in models_to_test:
        print(f"\n{'=' * 70}")
        print(f"测试模型: {model_name}")
        print("=" * 70)

        tokenizer, model, device = load_model(model_name)
        if model is None:
            continue

        print(f"设备: {device}")

        model_results = {}

        # 测试各类别
        for category in ['chatgpt_ai', 'davinci_ai', 'human']:
            print(f"\n评估 {category}...")
            result = evaluate_category(data[category], tokenizer, model, device)
            model_results[category] = result

            acc = result['accuracy'] * 100
            print(f"  准确率: {result['correct']}/{result['count']} = {acc:.2f}%")
            print(f"  正确预测置信度: {result['avg_conf_correct']*100:.1f}%")
            if result['avg_conf_wrong'] > 0:
                print(f"  错误预测置信度: {result['avg_conf_wrong']*100:.1f}%")

        results[model_name] = model_results

        # 清理
        del model
        torch.cuda.empty_cache()

    # 对比分析
    print("\n" + "=" * 70)
    print("跨模型泛化对比")
    print("=" * 70)

    print(f"\n{'模型':<25} | {'ChatGPT':>10} | {'Davinci':>10} | {'Human':>10} | {'差距':>8}")
    print("-" * 75)

    for model_name, res in results.items():
        chatgpt_acc = res['chatgpt_ai']['accuracy'] * 100
        davinci_acc = res['davinci_ai']['accuracy'] * 100
        human_acc = res['human']['accuracy'] * 100
        gap = chatgpt_acc - davinci_acc

        print(f"{model_name:<25} | {chatgpt_acc:>9.2f}% | {davinci_acc:>9.2f}% | {human_acc:>9.2f}% | {gap:>+7.2f}%")

    # 关键发现
    print("\n" + "=" * 70)
    print("关键发现")
    print("=" * 70)

    for model_name, res in results.items():
        chatgpt_acc = res['chatgpt_ai']['accuracy']
        davinci_acc = res['davinci_ai']['accuracy']
        gap = chatgpt_acc - davinci_acc

        print(f"\n[{model_name}]")
        if gap > 0.10:
            print(f"  ⚠️ 严重过拟合: ChatGPT {chatgpt_acc*100:.1f}% vs Davinci {davinci_acc*100:.1f}%")
            print(f"     模型可能学习了 ChatGPT 特有的表达模式，而非通用 AI 特征")
        elif gap > 0.05:
            print(f"  ⚡ 中度差异: ChatGPT {chatgpt_acc*100:.1f}% vs Davinci {davinci_acc*100:.1f}%")
            print(f"     模型对不同 AI 模型有一定区分度")
        else:
            print(f"  ✅ 泛化良好: ChatGPT {chatgpt_acc*100:.1f}% vs Davinci {davinci_acc*100:.1f}%")
            print(f"     模型学习到了通用的 AI 生成特征")

    # 保存结果
    output_file = PROJECT_ROOT / 'docs' / 'plans' / 'cross_model_test_result.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # 转换 defaultdict 为普通 dict
    serializable_results = {}
    for model_name, res in results.items():
        serializable_results[model_name] = {}
        for cat, cat_res in res.items():
            cat_res_copy = dict(cat_res)
            if 'length_buckets' in cat_res_copy:
                cat_res_copy['length_buckets'] = {
                    k: dict(v) for k, v in cat_res_copy['length_buckets'].items()
                }
            serializable_results[model_name][cat] = cat_res_copy

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存至: {output_file}")


if __name__ == '__main__':
    main()
