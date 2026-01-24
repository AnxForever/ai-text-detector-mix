"""
训练过程可视化脚本
根据training_history.json生成训练曲线
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_training_curves(history_path, output_dir='evaluation_results'):
    """绘制训练曲线"""
    os.makedirs(output_dir, exist_ok=True)

    # 读取训练历史
    with open(history_path, 'r', encoding='utf-8') as f:
        history = json.load(f)

    epochs = list(range(1, len(history['train_loss']) + 1))

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training Process - AI Text Detection Model',
                fontsize=16, fontweight='bold')

    # 1. 损失曲线
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], 'o-', color='#1f77b4',
            linewidth=2, markersize=8, label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 's-', color='#ff7f0e',
            linewidth=2, markersize=8, label='Validation Loss')
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training and Validation Loss', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.set_xticks(epochs)

    # 标注最佳epoch
    best_epoch = np.argmin(history['val_loss']) + 1
    ax1.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5,
               label=f'Best Epoch ({best_epoch})')
    ax1.legend(fontsize=11)

    # 2. 准确率曲线
    ax2 = axes[0, 1]
    ax2.plot(epochs, [acc*100 for acc in history['train_acc']], 'o-',
            color='#2ca02c', linewidth=2, markersize=8, label='Training Accuracy')
    ax2.plot(epochs, [acc*100 for acc in history['val_acc']], 's-',
            color='#d62728', linewidth=2, markersize=8, label='Validation Accuracy')
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Training and Validation Accuracy', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    ax2.set_xticks(epochs)
    ax2.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5,
               label=f'Best Epoch ({best_epoch})')
    ax2.legend(fontsize=11)

    # 3. F1分数曲线
    ax3 = axes[1, 0]
    ax3.plot(epochs, history['val_f1'], 'o-', color='#9467bd',
            linewidth=2, markersize=8, label='Validation F1 Score')
    ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax3.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax3.set_title('Validation F1 Score', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(alpha=0.3)
    ax3.set_xticks(epochs)
    ax3.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5,
               label=f'Best Epoch ({best_epoch})')
    ax3.legend(fontsize=11)

    # 4. 验证集损失vs准确率对比
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()

    line1 = ax4.plot(epochs, history['val_loss'], 'o-', color='#ff7f0e',
                    linewidth=2, markersize=8, label='Validation Loss')
    line2 = ax4_twin.plot(epochs, [acc*100 for acc in history['val_acc']], 's-',
                         color='#2ca02c', linewidth=2, markersize=8,
                         label='Validation Accuracy')

    ax4.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Loss', fontsize=12, fontweight='bold', color='#ff7f0e')
    ax4_twin.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold', color='#2ca02c')
    ax4.set_title('Validation Loss vs Accuracy', fontsize=13, fontweight='bold')
    ax4.tick_params(axis='y', labelcolor='#ff7f0e')
    ax4_twin.tick_params(axis='y', labelcolor='#2ca02c')
    ax4.grid(alpha=0.3)
    ax4.set_xticks(epochs)

    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='center right', fontsize=11)

    ax4.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()

    # 保存
    output_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 训练曲线已保存: {output_path}")
    plt.close()

    # 打印训练统计
    print("\n" + "="*60)
    print("训练过程统计")
    print("="*60)
    print(f"\n最佳Epoch: {best_epoch}")
    print(f"  验证损失: {history['val_loss'][best_epoch-1]:.6f}")
    print(f"  验证准确率: {history['val_acc'][best_epoch-1]*100:.2f}%")
    print(f"  验证F1分数: {history['val_f1'][best_epoch-1]:.4f}")

    print(f"\n最终Epoch: {len(epochs)}")
    print(f"  训练准确率: {history['train_acc'][-1]*100:.2f}%")
    print(f"  验证准确率: {history['val_acc'][-1]*100:.2f}%")

    # 创建简单的数据表
    print("\n" + "="*60)
    print("逐Epoch训练数据")
    print("="*60)
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12} {'Val F1':<10}")
    print("-"*60)

    for i in range(len(epochs)):
        print(f"{epochs[i]:<8} "
              f"{history['train_loss'][i]:<12.6f} "
              f"{history['train_acc'][i]*100:<12.2f} "
              f"{history['val_loss'][i]:<12.6f} "
              f"{history['val_acc'][i]*100:<12.2f} "
              f"{history['val_f1'][i]:<10.4f}")


def main():
    print("="*60)
    print("训练过程可视化")
    print("="*60)

    history_path = 'models/bert_improved/best_model/training_history.json'
    output_dir = 'evaluation_results'

    if not os.path.exists(history_path):
        print(f"错误: 训练历史文件不存在: {history_path}")
        return

    plot_training_curves(history_path, output_dir)

    print("\n" + "="*60)
    print("可视化完成！")
    print("="*60)


if __name__ == "__main__":
    main()
