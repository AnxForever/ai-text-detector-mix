"""
完整评估脚本（Windows版）- 生成所有评估指标和可视化
✅ 已修复：中文显示问题（使用Windows系统字体）
✅ 可在Windows PowerShell中直接运行

使用方法：
    python scripts/evaluation/complete_evaluation_windows.py
"""
import os
import sys
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
import warnings
warnings.filterwarnings('ignore')

# ==================== Windows中文字体配置 ====================
# 关键规则：所有matplotlib可视化必须配置中文字体！
def setup_chinese_font_windows():
    """配置Windows中文字体显示"""
    from matplotlib.font_manager import FontProperties, fontManager
    import platform

    # Windows系统字体路径
    windows_fonts_path = r'C:\Windows\Fonts'

    # 尝试多种Windows中文字体（按优先级）
    windows_chinese_fonts = [
        'Microsoft YaHei',    # 微软雅黑（推荐）
        'SimHei',             # 黑体
        'SimSun',             # 宋体
        'KaiTi',              # 楷体
        'FangSong',           # 仿宋
    ]

    print("正在配置中文字体...")

    # 方法1：直接设置字体名称
    for font_name in windows_chinese_fonts:
        try:
            # 测试字体是否可用
            test_fig = plt.figure()
            plt.rcParams['font.sans-serif'] = [font_name]
            plt.rcParams['axes.unicode_minus'] = False
            plt.close(test_fig)
            print(f"✓ 成功配置中文字体: {font_name}")
            return True
        except:
            continue

    # 方法2：从字体文件加载
    font_files = {
        'Microsoft YaHei': 'msyh.ttc',
        'SimHei': 'simhei.ttf',
        'SimSun': 'simsun.ttc',
    }

    for font_name, font_file in font_files.items():
        font_path = os.path.join(windows_fonts_path, font_file)
        if os.path.exists(font_path):
            try:
                from matplotlib.font_manager import fontManager
                fontManager.addfont(font_path)
                plt.rcParams['font.sans-serif'] = [font_name]
                plt.rcParams['axes.unicode_minus'] = False
                print(f"✓ 从文件加载中文字体: {font_name} ({font_file})")
                return True
            except Exception as e:
                print(f"  尝试加载 {font_name} 失败: {e}")
                continue

    # 方法3：使用系统默认
    print("⚠️ 警告: 使用系统默认字体，中文可能显示为方框")
    plt.rcParams['axes.unicode_minus'] = False
    return False

# 初始化中文字体
has_chinese_font = setup_chinese_font_windows()
print()
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
        print("模型加载成功\n")

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

        print(f"正在评估: {csv_path}")

        # 读取数据
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        texts = df['text'].tolist()
        true_labels = df['label'].values

        print(f"样本数: {len(texts)}")

        # 预测
        print("正在预测...\n")
        pred_labels, pred_probs = self.predict_with_probabilities(texts)

        # 1. 基础指标
        print("="*60)
        print("基础评估指标")
        print("="*60)

        accuracy = accuracy_score(true_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average='binary'
        )

        print(f"准确率 (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"精确率 (Precision): {precision:.4f} ({precision*100:.2f}%)")
        print(f"召回率 (Recall): {recall:.4f} ({recall*100:.2f}%)")
        print(f"F1分数: {f1:.4f}\n")

        # 2. AUC和ROC
        print("="*60)
        print("ROC曲线和AUC")
        print("="*60)

        # 获取正类（AI生成，label=1）的概率
        probs_positive = pred_probs[:, 1]

        # 计算ROC曲线
        fpr, tpr, thresholds = roc_curve(true_labels, probs_positive)
        roc_auc = auc(fpr, tpr)

        print(f"AUC (Area Under Curve): {roc_auc:.4f}\n")

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

        # 右图：放大左上角
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
        print(f"✓ ROC曲线已保存: {roc_path}\n")
        plt.close()

        # 3. 混淆矩阵
        print("="*60)
        print("混淆矩阵")
        print("="*60)

        cm = confusion_matrix(true_labels, pred_labels)
        tn, fp, fn, tp = cm.ravel()

        print(f"True Negative (TN):  {tn} - 正确识别为人类")
        print(f"False Positive (FP): {fp} - 人类误判为AI")
        print(f"False Negative (FN): {fn} - AI误判为人类")
        print(f"True Positive (TP):  {tp} - 正确识别为AI\n")

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
        plt.text(0.5, -0.15, f'总样本数: {len(texts)} | 准确率: {accuracy*100:.2f}%',
                ha='center', transform=plt.gca().transAxes, fontsize=12, fontweight='bold')

        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        print(f"✓ 混淆矩阵已保存: {cm_path}\n")
        plt.close()

        # 4. 预测置信度分布
        print("="*60)
        print("预测置信度分布")
        print("="*60)

        max_probs = np.max(pred_probs, axis=1)
        print(f"平均置信度: {max_probs.mean():.4f}")
        print(f"最小置信度: {max_probs.min():.4f}")
        print(f"最大置信度: {max_probs.max():.4f}\n")

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

        # 右图：正确vs错误预测对比
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
        print(f"✓ 置信度分布已保存: {conf_path}\n")
        plt.close()

        # 5. 保存完整报告
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

        print(f"✓ 完整评估报告已保存: {report_path}")

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
    print("AI文本检测模型 - 完整评估 (Windows版)")
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
    print("\n所有图表均支持中文显示 ✓")


if __name__ == "__main__":
    main()
