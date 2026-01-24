"""
完整评估脚本 - 生成所有评估指标和可视化
包括：准确率、F1分数、精确率、召回率、AUC、ROC曲线、混淆矩阵等
"""
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score
)
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ModelEvaluator:
    def __init__(self, model_path, device='cuda'):
        """初始化评估器"""
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {self.device}")

        # 加载模型
        print("正在加载模型...")
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        print("模型加载成功")

    def predict_with_probabilities(self, texts, batch_size=16):
        """预测并返回概率"""
        all_probs = []
        all_preds = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]

            encodings = self.tokenizer(
                batch_texts,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        return np.array(all_preds), np.array(all_probs)

    def evaluate_dataset(self, csv_path, output_dir='evaluation_results'):
        """完整评估并生成所有指标和可视化"""
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n正在评估: {csv_path}")

        # 读取数据
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        texts = df['text'].tolist()
        true_labels = df['label'].values

        print(f"样本数: {len(texts)}")

        # 预测
        print("正在预测...")
        pred_labels, pred_probs = self.predict_with_probabilities(texts)

        # 1. 基础指标
        print("\n" + "="*60)
        print("基础评估指标")
        print("="*60)

        accuracy = accuracy_score(true_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average='binary'
        )

        print(f"准确率 (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"精确率 (Precision): {precision:.4f} ({precision*100:.2f}%)")
        print(f"召回率 (Recall): {recall:.4f} ({recall*100:.2f}%)")
        print(f"F1分数: {f1:.4f}")

        # 2. AUC和ROC
        print("\n" + "="*60)
        print("ROC曲线和AUC")
        print("="*60)

        # 获取正类（AI生成，label=1）的概率
        probs_positive = pred_probs[:, 1]

        # 计算ROC曲线
        fpr, tpr, thresholds = roc_curve(true_labels, probs_positive)
        roc_auc = auc(fpr, tpr)

        print(f"AUC (Area Under Curve): {roc_auc:.4f}")

        # 绘制ROC曲线
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random Classifier (AUC = 0.5)')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)', fontsize=12)
        plt.ylabel('True Positive Rate (TPR)', fontsize=12)
        plt.title('ROC Curve - AI Text Detection', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3)

        roc_path = os.path.join(output_dir, 'roc_curve.png')
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        print(f"✓ ROC曲线已保存: {roc_path}")
        plt.close()

        # 3. 混淆矩阵
        print("\n" + "="*60)
        print("混淆矩阵")
        print("="*60)

        cm = confusion_matrix(true_labels, pred_labels)

        # 计算各个值
        tn, fp, fn, tp = cm.ravel()

        print(f"\n混淆矩阵:")
        print(f"              预测人类    预测AI")
        print(f"实际人类      {tn:6d}     {fp:6d}")
        print(f"实际AI        {fn:6d}     {tp:6d}")

        print(f"\n详细:")
        print(f"  True Negative (TN):  {tn} - 正确识别为人类")
        print(f"  False Positive (FP): {fp} - 人类误判为AI")
        print(f"  False Negative (FN): {fn} - AI误判为人类")
        print(f"  True Positive (TP):  {tp} - 正确识别为AI")

        # 绘制混淆矩阵热力图
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['人类撰写', 'AI生成'],
                   yticklabels=['人类撰写', 'AI生成'],
                   cbar_kws={'label': 'Count'},
                   annot_kws={'size': 16, 'weight': 'bold'})
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.title('Confusion Matrix - AI Text Detection', fontsize=14, fontweight='bold')

        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        print(f"✓ 混淆矩阵已保存: {cm_path}")
        plt.close()

        # 4. 分类报告
        print("\n" + "="*60)
        print("详细分类报告")
        print("="*60)

        report = classification_report(
            true_labels, pred_labels,
            target_names=['人类撰写', 'AI生成'],
            digits=4
        )
        print(report)

        # 5. 置信度分布
        print("\n" + "="*60)
        print("预测置信度分布")
        print("="*60)

        # 获取每个预测的置信度
        confidences = np.max(pred_probs, axis=1)

        print(f"平均置信度: {np.mean(confidences):.4f}")
        print(f"最小置信度: {np.min(confidences):.4f}")
        print(f"最大置信度: {np.max(confidences):.4f}")
        print(f"置信度中位数: {np.median(confidences):.4f}")

        # 绘制置信度分布直方图
        plt.figure(figsize=(12, 6))

        # 分别绘制正确和错误预测的置信度
        correct_mask = (pred_labels == true_labels)

        plt.subplot(1, 2, 1)
        plt.hist(confidences[correct_mask], bins=50, alpha=0.7,
                color='green', edgecolor='black', label='Correct')
        plt.hist(confidences[~correct_mask], bins=50, alpha=0.7,
                color='red', edgecolor='black', label='Incorrect')
        plt.xlabel('Confidence', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)

        # 按类别绘制
        plt.subplot(1, 2, 2)
        human_conf = confidences[true_labels == 0]
        ai_conf = confidences[true_labels == 1]

        plt.hist(human_conf, bins=30, alpha=0.7, color='blue',
                edgecolor='black', label='Human')
        plt.hist(ai_conf, bins=30, alpha=0.7, color='orange',
                edgecolor='black', label='AI')
        plt.xlabel('Confidence', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Confidence by True Label', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()
        conf_path = os.path.join(output_dir, 'confidence_distribution.png')
        plt.savefig(conf_path, dpi=300, bbox_inches='tight')
        print(f"✓ 置信度分布已保存: {conf_path}")
        plt.close()

        # 6. 保存完整报告
        report_path = os.path.join(output_dir, 'evaluation_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("AI文本检测模型 - 完整评估报告\n")
            f.write("="*60 + "\n\n")

            f.write("1. 基础指标\n")
            f.write("-"*60 + "\n")
            f.write(f"准确率 (Accuracy):  {accuracy:.4f} ({accuracy*100:.2f}%)\n")
            f.write(f"精确率 (Precision): {precision:.4f} ({precision*100:.2f}%)\n")
            f.write(f"召回率 (Recall):    {recall:.4f} ({recall*100:.2f}%)\n")
            f.write(f"F1分数:            {f1:.4f}\n")
            f.write(f"AUC:               {roc_auc:.4f}\n\n")

            f.write("2. 混淆矩阵\n")
            f.write("-"*60 + "\n")
            f.write(f"True Negative (TN):  {tn} (正确识别为人类)\n")
            f.write(f"False Positive (FP): {fp} (人类误判为AI)\n")
            f.write(f"False Negative (FN): {fn} (AI误判为人类)\n")
            f.write(f"True Positive (TP):  {tp} (正确识别为AI)\n\n")

            f.write("3. 置信度统计\n")
            f.write("-"*60 + "\n")
            f.write(f"平均置信度: {np.mean(confidences):.4f}\n")
            f.write(f"最小置信度: {np.min(confidences):.4f}\n")
            f.write(f"最大置信度: {np.max(confidences):.4f}\n")
            f.write(f"置信度中位数: {np.median(confidences):.4f}\n\n")

            f.write("4. 详细分类报告\n")
            f.write("-"*60 + "\n")
            f.write(report)

        print(f"\n✓ 完整评估报告已保存: {report_path}")

        # 返回结果字典
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': roc_auc,
            'confusion_matrix': cm,
            'classification_report': report
        }


def main():
    print("="*60)
    print("AI文本检测模型 - 完整评估")
    print("="*60)

    # 配置
    model_path = 'models/bert_improved/best_model'
    test_csv = 'datasets/bert_debiased/test.csv'
    output_dir = 'evaluation_results'

    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型路径不存在: {model_path}")
        return

    if not os.path.exists(test_csv):
        print(f"错误: 测试集不存在: {test_csv}")
        return

    # 创建评估器
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    evaluator = ModelEvaluator(model_path, device=device)

    # 执行评估
    results = evaluator.evaluate_dataset(test_csv, output_dir)

    print("\n" + "="*60)
    print("评估完成！")
    print("="*60)
    print(f"\n生成的文件:")
    print(f"  - {output_dir}/roc_curve.png")
    print(f"  - {output_dir}/confusion_matrix.png")
    print(f"  - {output_dir}/confidence_distribution.png")
    print(f"  - {output_dir}/evaluation_report.txt")
    print()


if __name__ == "__main__":
    main()
