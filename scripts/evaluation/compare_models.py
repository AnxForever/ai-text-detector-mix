#!/usr/bin/env python3
"""
模型对比评估脚本
在多个高难度测试场景下对比两个模型：
  - bert_v2_overnight (旧模型，有长度偏差)
  - bert_v2_balanced  (新模型，长度平衡)

高难度测试场景：
  1. 标准测试集 (balanced test set)
  2. 长度陷阱测试 (短AI + 长Human)
  3. 跨域测试 (训练集未涉及的来源)
  4. 边界长度测试 (500-1000字符，最易混淆区间)
  5. 极短文本测试 (<100字符)
  6. 外部数据集测试 (HC3/M4 等未参与训练的样本)
"""
import os
import sys
import json
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def evaluate_model(model, tokenizer, texts, labels, device, batch_size=4):
    """评估模型，返回详细指标."""
    dataset = TextDataset(texts, labels, tokenizer, max_len=256)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch_labels.cpu().tolist())

    # 计算指标
    total = len(all_labels)
    correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
    accuracy = correct / total if total > 0 else 0

    # Human (label=0) 指标
    human_total = sum(1 for l in all_labels if l == 0)
    human_correct = sum(1 for p, l in zip(all_preds, all_labels) if l == 0 and p == 0)
    human_acc = human_correct / human_total if human_total > 0 else 0

    # AI (label=1) 指标
    ai_total = sum(1 for l in all_labels if l == 1)
    ai_correct = sum(1 for p, l in zip(all_preds, all_labels) if l == 1 and p == 1)
    ai_acc = ai_correct / ai_total if ai_total > 0 else 0

    # 精确率/召回率/F1
    tp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 1)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'total': total,
        'accuracy': accuracy,
        'human_total': human_total,
        'human_acc': human_acc,
        'ai_total': ai_total,
        'ai_acc': ai_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def load_model(model_path, device):
    """加载模型和 tokenizer."""
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(
        model_path, num_labels=2
    ).to(device)
    model.eval()
    return model, tokenizer


def build_test_scenarios(data_dir, external_path):
    """构建多个高难度测试场景."""
    scenarios = {}

    # 1. 标准平衡测试集
    test_df = pd.read_csv(f'{data_dir}/test_balanced.csv')
    test_df = test_df.dropna(subset=['text', 'label'])
    test_df['label'] = test_df['label'].astype(int)
    test_df['text_len'] = test_df['text'].astype(str).apply(len)
    scenarios['标准平衡测试集'] = {
        'texts': test_df['text'].tolist(),
        'labels': test_df['label'].tolist(),
        'description': f'平衡测试集 (Human/AI 各半, {len(test_df)} 条)'
    }

    # 2. 长度陷阱测试：短AI + 长Human（对有长度偏差的模型最难）
    short_ai = test_df[(test_df['label'] == 1) & (test_df['text_len'] < 200)]
    long_human = test_df[(test_df['label'] == 0) & (test_df['text_len'] > 500)]
    trap_df = pd.concat([short_ai, long_human])
    if len(trap_df) > 0:
        scenarios['长度陷阱 (短AI+长Human)'] = {
            'texts': trap_df['text'].tolist(),
            'labels': trap_df['label'].tolist(),
            'description': f'短AI ({len(short_ai)}) + 长Human ({len(long_human)})'
        }

    # 3. 反向长度陷阱：长AI + 短Human
    long_ai = test_df[(test_df['label'] == 1) & (test_df['text_len'] > 500)]
    short_human = test_df[(test_df['label'] == 0) & (test_df['text_len'] < 200)]
    reverse_trap = pd.concat([long_ai, short_human])
    if len(reverse_trap) > 0:
        scenarios['反向长度 (长AI+短Human)'] = {
            'texts': reverse_trap['text'].tolist(),
            'labels': reverse_trap['label'].tolist(),
            'description': f'长AI ({len(long_ai)}) + 短Human ({len(short_human)})'
        }

    # 4. 边界长度测试 (500-1000字符区间)
    boundary = test_df[(test_df['text_len'] >= 500) & (test_df['text_len'] <= 1000)]
    if len(boundary) > 0:
        scenarios['边界长度 (500-1000字符)'] = {
            'texts': boundary['text'].tolist(),
            'labels': boundary['label'].tolist(),
            'description': f'500-1000 字符区间 ({len(boundary)} 条)'
        }

    # 5. 极短文本测试 (<100字符)
    very_short = test_df[test_df['text_len'] < 100]
    if len(very_short) > 0:
        scenarios['极短文本 (<100字符)'] = {
            'texts': very_short['text'].tolist(),
            'labels': very_short['label'].tolist(),
            'description': f'极短文本 ({len(very_short)} 条)'
        }

    # 6. 外部数据集测试 (未参与训练的样本)
    if os.path.exists(external_path):
        with open(external_path, 'r', encoding='utf-8') as f:
            ext_data = json.load(f)

        if isinstance(ext_data, dict) and 'human' in ext_data and 'ai' in ext_data:
            # 从外部数据中随机采样一些未参与训练的数据
            import random
            random.seed(42)

            ext_texts = []
            ext_labels = []

            # 采样 Human
            human_samples = ext_data['human']
            random.shuffle(human_samples)
            for item in human_samples[:500]:
                text = item.get('text', '')
                if len(text) >= 50:
                    ext_texts.append(text)
                    ext_labels.append(0)

            # 采样 AI
            ai_samples = ext_data['ai']
            random.shuffle(ai_samples)
            for item in ai_samples[:500]:
                text = item.get('text', '')
                if len(text) >= 50:
                    ext_texts.append(text)
                    ext_labels.append(1)

            if ext_texts:
                scenarios['外部数据集 (随机采样)'] = {
                    'texts': ext_texts,
                    'labels': ext_labels,
                    'description': f'外部 HC3/M4/LCSTS 随机采样 ({len(ext_texts)} 条)'
                }

            # 7. 按来源分类的外部测试
            by_source = defaultdict(lambda: {'texts': [], 'labels': []})
            for item in human_samples[:2000]:
                src = item.get('source', 'unknown')
                if len(item.get('text', '')) >= 50:
                    by_source[src]['texts'].append(item['text'])
                    by_source[src]['labels'].append(0)

            for item in ai_samples[:2000]:
                src = item.get('source', 'unknown')
                if len(item.get('text', '')) >= 50:
                    by_source[src]['texts'].append(item['text'])
                    by_source[src]['labels'].append(1)

            for src, data in by_source.items():
                if len(data['texts']) >= 50:
                    scenarios[f'外部-{src}'] = {
                        'texts': data['texts'],
                        'labels': data['labels'],
                        'description': f'{src} 来源 ({len(data["texts"])} 条)'
                    }

    # 8. OOD 测试：使用 core_v1 测试集
    core_v1_path = 'datasets/active/core_v1'
    if os.path.exists(core_v1_path):
        ood_texts = []
        ood_labels = []
        for fname in os.listdir(core_v1_path):
            if fname.endswith('.jsonl'):
                fpath = os.path.join(core_v1_path, fname)
                with open(fpath, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            item = json.loads(line.strip())
                            text = item.get('text', '')
                            label = item.get('label', -1)
                            if len(text) >= 50 and label in [0, 1]:
                                ood_texts.append(text)
                                ood_labels.append(label)
                        except json.JSONDecodeError:
                            pass

        if ood_texts:
            # 随机采样最多 2000 条
            import random
            random.seed(42)
            indices = list(range(len(ood_texts)))
            random.shuffle(indices)
            indices = indices[:2000]
            scenarios['OOD-core_v1'] = {
                'texts': [ood_texts[i] for i in indices],
                'labels': [ood_labels[i] for i in indices],
                'description': f'core_v1 原始数据 OOD 测试 ({min(len(ood_texts), 2000)} 条)'
            }

    # 9. 加载预构建的高难度测试集
    difficult_dir = 'datasets/bert_v2_overnight/difficult_tests'
    if os.path.exists(difficult_dir):
        index_path = os.path.join(difficult_dir, 'index.json')
        if os.path.exists(index_path):
            with open(index_path, 'r', encoding='utf-8') as f:
                difficult_index = json.load(f)

            for item in difficult_index:  # noqa: F841
                name = item['name']
                filepath = item['path']

                if os.path.exists(filepath):
                    texts = []
                    labels = []
                    with open(filepath, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                d = json.loads(line.strip())
                                texts.append(d.get('text', ''))
                                labels.append(d.get('label', 0))
                            except json.JSONDecodeError:
                                pass

                    if texts:
                        scenarios[f'难度测试-{name}'] = {
                            'texts': texts,
                            'labels': labels,
                            'description': f'{name} ({len(texts)} 条)'
                        }

    return scenarios


def print_comparison(results, scenario_name):
    """打印对比结果."""
    old = results['old']
    new = results['new']

    print(f"\n{'─'*70}")
    print(f"场景: {scenario_name}")
    print(f"说明: {results.get('description', '')}")
    print(f"{'─'*70}")
    print(f"{'指标':<20} | {'旧模型(biased)':>15} | {'新模型(balanced)':>15} | {'差异':>10}")
    print(f"{'─'*20}─┼─{'─'*15}─┼─{'─'*15}─┼─{'─'*10}")

    metrics = [
        ('样本数', 'total', '{:d}', False),
        ('总准确率', 'accuracy', '{:.2%}', True),
        ('Human 准确率', 'human_acc', '{:.2%}', True),
        ('AI 准确率', 'ai_acc', '{:.2%}', True),
        ('精确率', 'precision', '{:.2%}', True),
        ('召回率', 'recall', '{:.2%}', True),
        ('F1', 'f1', '{:.2%}', True),
    ]

    for label, key, fmt, show_diff in metrics:
        old_val = old.get(key, 0)
        new_val = new.get(key, 0)
        old_str = fmt.format(old_val)
        new_str = fmt.format(new_val)

        if show_diff:
            diff = new_val - old_val
            if diff > 0.001:
                diff_str = f'+{diff:.2%} ↑'
            elif diff < -0.001:
                diff_str = f'{diff:.2%} ↓'
            else:
                diff_str = '持平'
        else:
            diff_str = ''

        print(f"{label:<20} | {old_str:>15} | {new_str:>15} | {diff_str:>10}")


def main():
    print("=" * 70)
    print("模型对比评估 - 旧模型(biased) vs 新模型(balanced)")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 模型路径
    old_model_path = 'models/bert_v2_overnight'
    new_model_path = 'models/bert_v2_balanced'

    # 检查模型存在
    if not os.path.exists(old_model_path):
        print(f"[ERROR] 旧模型不存在: {old_model_path}")
        sys.exit(1)
    if not os.path.exists(new_model_path):
        print(f"[ERROR] 新模型不存在: {new_model_path}")
        print("请先运行 train_balanced.py 训练新模型")
        sys.exit(1)

    # 加载模型
    print("\n加载模型...")
    print(f"  旧模型: {old_model_path}")
    old_model, old_tokenizer = load_model(old_model_path, device)
    print(f"  新模型: {new_model_path}")
    new_model, new_tokenizer = load_model(new_model_path, device)
    print("  模型加载完成")

    # 构建测试场景
    print("\n构建测试场景...")
    data_dir = 'datasets/bert_v2_overnight'
    external_path = 'datasets/external/extracted_all_for_training.json'
    scenarios = build_test_scenarios(data_dir, external_path)
    print(f"  共 {len(scenarios)} 个测试场景")

    # 逐场景评估
    all_results = {}
    for name, scenario in scenarios.items():
        texts = scenario['texts']
        labels = scenario['labels']
        desc = scenario['description']

        print(f"\n评估场景: {name} ({len(texts)} 条)...")

        old_metrics = evaluate_model(
            old_model, old_tokenizer, texts, labels, device
        )
        new_metrics = evaluate_model(
            new_model, new_tokenizer, texts, labels, device
        )

        result = {
            'old': old_metrics,
            'new': new_metrics,
            'description': desc
        }
        all_results[name] = result
        print_comparison(result, name)

        torch.cuda.empty_cache()

    # 汇总表
    print("\n\n" + "=" * 70)
    print("汇总对比表")
    print("=" * 70)
    print(f"\n{'场景':<30} | {'旧Acc':>8} | {'新Acc':>8} | {'旧F1':>8} | {'新F1':>8} | {'胜者':>6}")
    print(f"{'─'*30}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*6}")

    old_wins = 0
    new_wins = 0
    for name, r in all_results.items():
        old_acc = r['old']['accuracy']
        new_acc = r['new']['accuracy']
        old_f1 = r['old']['f1']
        new_f1 = r['new']['f1']

        if new_f1 > old_f1 + 0.001:
            winner = '新 ✓'
            new_wins += 1
        elif old_f1 > new_f1 + 0.001:
            winner = '旧 ✓'
            old_wins += 1
        else:
            winner = '平'

        short_name = name[:28] if len(name) > 28 else name
        print(f"{short_name:<30} | {old_acc:>7.2%} | {new_acc:>7.2%} | "
              f"{old_f1:>7.2%} | {new_f1:>7.2%} | {winner:>6}")

    print(f"\n总结: 新模型胜 {new_wins} 场, 旧模型胜 {old_wins} 场, "
          f"平局 {len(all_results) - old_wins - new_wins} 场")

    # 保存结果
    output_path = 'models/bert_v2_balanced/comparison_results.json'
    save_data = {}
    for name, r in all_results.items():
        save_data[name] = {
            'description': r['description'],
            'old_model': r['old'],
            'new_model': r['new']
        }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {output_path}")


if __name__ == '__main__':
    main()
