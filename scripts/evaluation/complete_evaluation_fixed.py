"""
完整评估脚本（修复版）- 生成所有评估指标和可视化
解决ROC曲线和训练曲线的可视化问题
✅ 已修复：中文显示问题
"""
import os
import torch
import pandas as pd
import numpy as np
import matplotlib
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

# ==================== 中文字体配置 ====================
# 关键：必须在导入matplotlib后、绘图前配置
def setup_chinese_font():
    """配置matplotlib中文字体显示"""
    import platform
    from matplotlib.font_manager import FontProperties, fontManager

    # 尝试多种中文字体（按优先级）
    chinese_fonts = [
        'Microsoft YaHei',      # Windows
        'SimHei',               # Windows
        'WenQuanYi Micro Hei',  # Linux
        'WenQuanYi Zen Hei',    # Linux
        'Droid Sans Fallback',  # Linux
        'Arial Unicode MS',     # macOS
        'PingFang SC',          # macOS
        'STHeiti',              # macOS/Linux
    ]

    # 获取系统所有可用字体
    available_fonts = set([f.name for f in fontManager.ttflist])

    # 查找第一个可用的中文字体
    selected_font = None
    for font in chinese_fonts:
        if font in available_fonts:
            selected_font = font
            print(f"✓ 使用中文字体: {font}")
            break

    if selected_font:
        # 设置全局字体
        plt.rcParams['font.sans-serif'] = [selected_font] + plt.rcParams['font.sans-serif']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        return True
    else:
        print("⚠️ 警告: 未找到中文字体，将使用英文标签")
        print(f"   可用字体示例: {list(available_fonts)[:5]}")
        return False

# 初始化中文字体
has_chinese_font = setup_chinese_font()
# ==================== 中文字体配置结束 ====================

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

        # 2. AUC和ROC（改进版）
        print("\n" + "="*60)
        print("ROC曲线和AUC")
        print("="*60)

        # 获取正类（AI生成，label=1）的概率
        probs_positive = pred_probs[:, 1]

        # 计算ROC曲线
        fpr, tpr, thresholds = roc_curve(true_labels, probs_positive)
        roc_auc = auc(fpr, tpr)

        print(f"AUC (Area Under Curve): {roc_auc:.4f}")

        # 分析预测概率分布
        print(f"\n预测概率分析:")
        print(f"  最小概率: {probs_positive.min():.6f}")
        print(f"  最大概率: {probs_positive.max():.6f}")
        print(f"  平均概率: {probs_positive.mean():.6f}")
        print(f"  中位数: {np.median(probs_positive):.6f}")
        print(f"  标准差: {probs_positive.std():.6f}")

        # 绘制改进的ROC曲线
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 左图：标准ROC曲线
        ax1.plot(fpr, tpr, color='darkorange', lw=3,
                label=f'ROC曲线 (AUC = {roc_auc:.4f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='随机分类器 (AUC = 0.5)')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('假正率 (False Positive Rate)', fontsize=13, fontweight='bold')
        ax1.set_ylabel('真正率 (True Positive Rate)', fontsize=13, fontweight='bold')
        ax1.set_title('ROC曲线 - 完整视图', fontsize=15, fontweight='bold', pad=15)
        ax1.legend(loc="lower right", fontsize=12)
        ax1.grid(alpha=0.3, linestyle='--')

        # 右图：放大左上角（0-0.1区间）
        # 找到FPR < 0.1的点
        mask = fpr <= 0.1
        fpr_zoom = fpr[mask]
        tpr_zoom = tpr[mask]

        ax2.plot(fpr_zoom, tpr_zoom, color='darkorange', lw=3, marker='o',
                markersize=5, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
        ax2.plot([0, 0.1], [0, 0.1], color='navy', lw=2, linestyle='--',
                label='随机分类器')
        ax2.set_xlim([0.0, 0.1])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('假正率 (False Positive Rate)', fontsize=13, fontweight='bold')
        ax2.set_ylabel('真正率 (True Positive Rate)', fontsize=13, fontweight='bold')
        ax2.set_title('ROC曲线 - 左上角放大 (FPR: 0-0.1)', fontsize=15, fontweight='bold', pad=15)
        ax2.legend(loc="lower right", fontsize=12)
        ax2.grid(alpha=0.3, linestyle='--')

        # 标注关键点
        if len(fpr_zoom) > 0:
            ax2.annotate('模型在极低FPR下\n达到高TPR',
                        xy=(fpr_zoom[-1] if fpr_zoom[-1] < 0.1 else 0.05,
                            tpr_zoom[-1] if len(tpr_zoom) > 0 else 0.5),
                        xytext=(0.05, 0.5),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2),
                        fontsize=11, color='red', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

        plt.tight_layout()
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

        # 绘制混淆矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['人类撰写', 'AI生成'],
                   yticklabels=['人类撰写', 'AI生成'],
                   cbar_kws={'label': '样本数'},
                   annot_kws={'size': 18, 'weight': 'bold'})
        plt.ylabel('真实标签', fontsize=14, fontweight='bold')
        plt.xlabel('预测标签', fontsize=14, fontweight='bold')
        plt.title('混淆矩阵 - AI文本检测', fontsize=16, fontweight='bold', pad=15)

        # 添加总样本数标注
        plt.text(0.5, -0.15, f'总样本数: {len(texts)} | 准确率: {accuracy*100:.2f}%',
                ha='center', transform=plt.gca().transAxes, fontsize=12, fontweight='bold')

        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        print(f"✓ 混淆矩阵已保存: {cm_path}")
        plt.close()

        # 4. 详细分类报告
        print("\n" + "="*60)
        print("详细分类报告")
        print("="*60)
        print(classification_report(true_labels, pred_labels,
                                   target_names=['人类撰写', 'AI生成']))

        # 5. 预测置信度分布
        print("\n" + "="*60)
        print("预测置信度分布")
        print("="*60)

        max_probs = np.max(pred_probs, axis=1)
        print(f"平均置信度: {max_probs.mean():.4f}")
        print(f"最小置信度: {max_probs.min():.4f}")
        print(f"最大置信度: {max_probs.max():.4f}")
        print(f"置信度中位数: {np.median(max_probs):.4f}")

        # 绘制置信度分布
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 左图：所有样本的置信度分布
        ax1.hist(max_probs, bins=50, color='skyblue', edgecolor='black', alpha=0.7, linewidth=1.5)
        ax1.axvline(max_probs.mean(), color='red', linestyle='--', linewidth=3,
                   label=f'平均值: {max_probs.mean():.4f}')
        ax1.set_xlabel('预测置信度', fontsize=13, fontweight='bold')
        ax1.set_ylabel('样本数', fontsize=13, fontweight='bold')
        ax1.set_title('预测置信度分布 (所有样本)', fontsize=15, fontweight='bold', pad=15)
        ax1.legend(fontsize=12, loc='upper left')
        ax1.grid(alpha=0.3, linestyle='--')

        # 右图：正确vs错误预测的置信度对比
        correct_mask = (pred_labels == true_labels)
        correct_conf = max_probs[correct_mask]
        wrong_conf = max_probs[~correct_mask]

        ax2.hist(correct_conf, bins=30, color='green', alpha=0.6,
                label=f'正确预测 (n={len(correct_conf)})', edgecolor='black', linewidth=1.5)
        if len(wrong_conf) > 0:
            ax2.hist(wrong_conf, bins=30, color='red', alpha=0.7,
                    label=f'错误预测 (n={len(wrong_conf)})', edgecolor='black', linewidth=1.5)
        ax2.set_xlabel('预测置信度', fontsize=13, fontweight='bold')
        ax2.set_ylabel('样本数', fontsize=13, fontweight='bold')
        ax2.set_title('正确 vs 错误预测的置信度对比', fontsize=15, fontweight='bold', pad=15)
        ax2.legend(fontsize=12, loc='upper left')
        ax2.grid(alpha=0.3, linestyle='--')

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
            f.write(f"平均置信度: {max_probs.mean():.4f}\n")
            f.write(f"最小置信度: {max_probs.min():.4f}\n")
            f.write(f"最大置信度: {max_probs.max():.4f}\n")
            f.write(f"置信度中位数: {np.median(max_probs):.4f}\n\n")

            f.write("4. 详细分类报告\n")
            f.write("-"*60 + "\n")
            f.write(classification_report(true_labels, pred_labels,
                                         target_names=['人类撰写', 'AI生成']))

        print(f"\n✓ 完整评估报告已保存: {report_path}")

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': roc_auc,
            'confusion_matrix': cm
        }


def main():
    print("="*60)
    print("AI文本检测模型 - 完整评估（修复版）")
    print("="*60)
    print()

    # 配置
    model_path = "models/bert_improved/best_model"
    test_csv = "datasets/bert_debiased/test.csv"
    output_dir = "evaluation_results"

    # 评估
    evaluator = ModelEvaluator(model_path)
    results = evaluator.evaluate_dataset(test_csv, output_dir)

    print("\n" + "="*60)
    print("评估完成！")
    print("="*60)
    print("\n生成的文件:")
    print(f"  - {output_dir}/roc_curve.png")
    print(f"  - {output_dir}/confusion_matrix.png")
    print(f"  - {output_dir}/confidence_distribution.png")
    print(f"  - {output_dir}/evaluation_report.txt")


if __name__ == "__main__":
    main()
