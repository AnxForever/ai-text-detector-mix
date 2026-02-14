#!/usr/bin/env python3
"""
评估指定模型组 - 被 agent 调用
用法: python eval_model_group.py <group_id> <model1> [model2] [model3]

结果保存到 datasets/eval/fair_test/results_group_<id>.json
"""
import os, sys, json, time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
os.chdir(PROJECT_ROOT)

PROBLEM_SAMPLES = [
    {"text": "温馨提示：明天上午9点将进行消防演练，请各位同事提前做好准备，保持通道畅通。", "label": 0, "cat": "正式通知"},
    {"text": "今天天气真好，中午和同事一起去公园散步了，心情舒畅。", "label": 0, "cat": "日常分享"},
    {"text": "这家店的牛肉面真的很不错，汤底浓郁，面条筋道，强烈推荐大家去尝尝。", "label": 0, "cat": "口语推荐"},
    {"text": "今天终于把报告写完了，感觉整个人都轻松了。", "label": 0, "cat": "工作日常"},
    {"text": "本文研究了深度学习在自然语言处理中的应用，通过实验验证了模型的有效性。", "label": 0, "cat": "学术摘要"},
    {"text": "实验结果表明，改进后的算法在准确率上提升了5%，验证了本文方法的可行性。", "label": 0, "cat": "实验结论"},
    {"text": "昨日，我市召开经济工作会议，市长强调要加快产业升级，推动高质量发展。", "label": 0, "cat": "地方新闻"},
    {"text": "关于2024年期末考试安排的通知：考试时间为1月15日至1月20日，请同学们认真复习。", "label": 0, "cat": "考试通知"},
    {"text": "尊敬的老师：您好！我是计算机系大三学生张明，想就毕业论文选题向您请教。", "label": 0, "cat": "学生邮件"},
    {"text": "综上所述，本研究为该领域提供了新的思路和方法，具有一定的理论和实践意义。", "label": 0, "cat": "论文结尾"},
    {"text": "梯度爆炸（gradient explosion）是指在深度神经网络训练过程中，反向传播时梯度值呈指数级增长的现象。", "label": 1, "cat": "AI技术文档"},
    {"text": "额...这个问题嘛...让我想想啊，好像是这样的...不对不对，应该是那样...", "label": 1, "cat": "犹豫式AI"},
    {"text": "老铁们这个东西绝绝子！yyds！真的太绝了，建议大家冲冲冲！", "label": 1, "cat": "网络语AI"},
    {"text": "深度学习是机器学习的一个分支，它使用多层神经网络来学习数据的层次化表示。通过自动提取特征，深度学习在图像识别、自然语言处理等领域取得了显著成果。", "label": 1, "cat": "AI百科解释"},
    {"text": "人工智能（Artificial Intelligence，简称AI）是计算机科学的一个重要分支。它致力于研究和开发能够模拟、延伸和扩展人类智能的理论、方法和技术。", "label": 1, "cat": "AI定义文本"},
]

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(str(self.texts[idx]), max_length=self.max_len,
                             padding='max_length', truncation=True, return_tensors='pt')
        return {
            'input_ids': enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def evaluate_batch(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            out = model(input_ids=ids, attention_mask=mask)
            probs = torch.softmax(out.logits, dim=-1)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['label'].numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

def compute_metrics(preds, labels):
    return {
        'accuracy': float(accuracy_score(labels, preds)),
        'human_precision': float(precision_score(labels, preds, pos_label=0, zero_division=0)),
        'human_recall': float(recall_score(labels, preds, pos_label=0, zero_division=0)),
        'human_f1': float(f1_score(labels, preds, pos_label=0, zero_division=0)),
        'ai_precision': float(precision_score(labels, preds, pos_label=1, zero_division=0)),
        'ai_recall': float(recall_score(labels, preds, pos_label=1, zero_division=0)),
        'ai_f1': float(f1_score(labels, preds, pos_label=1, zero_division=0)),
        'total': int(len(labels)),
        'human_count': int((np.array(labels) == 0).sum()),
        'ai_count': int((np.array(labels) == 1).sum()),
    }

def main():
    if len(sys.argv) < 3:
        print("用法: python eval_model_group.py <group_id> <model1> [model2] ...")
        sys.exit(1)

    group_id = sys.argv[1]
    model_names = sys.argv[2:]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Group {group_id}] 设备: {device}, 模型: {model_names}")

    # 加载测试集
    test_dir = PROJECT_ROOT / "datasets/eval/fair_test"
    test_sets = {}
    for csv_file in sorted(test_dir.glob("*.csv")):
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        if 'text' in df.columns and 'label' in df.columns:
            test_sets[csv_file.stem] = df
            print(f"  测试集 {csv_file.stem}: {len(df)} 条")

    results = {}
    batch_size = 32

    for model_name in model_names:
        model_path = PROJECT_ROOT / "models" / model_name
        if not model_path.exists():
            print(f"\n⏭️  {model_name}: 不存在")
            continue

        print(f"\n{'─'*60}")
        print(f"📊 评估: {model_name}")
        print(f"{'─'*60}")

        try:
            tokenizer = BertTokenizer.from_pretrained(str(model_path))
            model = BertForSequenceClassification.from_pretrained(str(model_path)).to(device)
            model.eval()
        except Exception as e:
            print(f"  ❌ 加载失败: {e}")
            continue

        model_results = {}

        # 各测试集
        for ts_name, ts_df in test_sets.items():
            texts = ts_df['text'].tolist()
            labels = ts_df['label'].tolist()
            dataset = TextDataset(texts, labels, tokenizer)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

            t0 = time.time()
            preds, true_labels, probs = evaluate_batch(model, loader, device)
            elapsed = time.time() - t0

            metrics = compute_metrics(preds, true_labels)
            metrics['time_seconds'] = round(elapsed, 1)
            model_results[ts_name] = metrics

            print(f"  [{ts_name}] acc={metrics['accuracy']*100:.2f}% "
                  f"H:P={metrics['human_precision']*100:.1f}/R={metrics['human_recall']*100:.1f} "
                  f"A:P={metrics['ai_precision']*100:.1f}/R={metrics['ai_recall']*100:.1f} "
                  f"({elapsed:.1f}s)")

        # 问题样本
        prob_results = []
        model.eval()
        for s in PROBLEM_SAMPLES:
            inputs = tokenizer(s['text'], return_tensors='pt', truncation=True, max_length=512, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs)
                probs = torch.softmax(out.logits, dim=-1)
                ai_prob = float(probs[0][1].item())
            pred = 1 if ai_prob >= 0.5 else 0
            correct = pred == s['label']
            prob_results.append({"cat": s['cat'], "label": s['label'], "ai_prob": round(ai_prob, 4), "correct": correct})
            mark = "✅" if correct else "❌"
            print(f"    {mark} [{s['cat']}] AI={ai_prob*100:.1f}%")

        correct_n = sum(1 for r in prob_results if r['correct'])
        model_results['problem_samples'] = prob_results
        model_results['problem_acc'] = round(correct_n / len(prob_results), 4)
        print(f"  问题样本: {correct_n}/{len(prob_results)} ({correct_n/len(prob_results)*100:.1f}%)")

        results[model_name] = model_results

        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # 保存结果
    out_path = test_dir / f"results_group_{group_id}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Group {group_id} 结果已保存: {out_path}")

if __name__ == '__main__':
    main()
