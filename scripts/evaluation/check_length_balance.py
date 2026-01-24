"""
验证数据集长度平衡性
"""
import pandas as pd
import numpy as np

# 读取数据集
train = pd.read_csv('datasets/bert/train.csv', encoding='utf-8-sig')
val = pd.read_csv('datasets/bert/val.csv', encoding='utf-8-sig')
test = pd.read_csv('datasets/bert/test.csv', encoding='utf-8-sig')

bins = [0, 300, 600, 1000, 1500, 2000, 3000, 10000]

print("=" * 70)
print("数据集长度平衡性验证")
print("=" * 70)

for name, df in [('训练集', train), ('验证集', val), ('测试集', test)]:
    print(f"\n{name} 长度分布平衡性:")
    print("=" * 70)

    df_copy = df.copy()
    df_copy['length_bin'] = pd.cut(df_copy['length'], bins=bins)

    total_ai = 0
    total_human = 0

    for bin_range in df_copy['length_bin'].cat.categories:
        bin_df = df_copy[df_copy['length_bin'] == bin_range]

        if len(bin_df) == 0:
            continue

        ai_count = (bin_df['label'] == 1).sum()
        human_count = (bin_df['label'] == 0).sum()

        total_ai += ai_count
        total_human += human_count

        if ai_count > 0 and human_count > 0:
            ratio = human_count / ai_count
            if 0.8 <= ratio <= 1.2:
                balance_status = "✓ 平衡"
            else:
                balance_status = "⚠ 不平衡"
        elif ai_count == 0:
            ratio = 0
            balance_status = "⚠ 无AI样本"
        elif human_count == 0:
            ratio = 0
            balance_status = "⚠ 无人类样本"
        else:
            ratio = 0
            balance_status = "⚠ 缺失"

        print(f"  {bin_range}: AI={ai_count}, 人类={human_count}, 比例={ratio:.2f} {balance_status}")

    print(f"\n  总计: AI={total_ai}, 人类={total_human}")
    print(f"  总体平衡比: {total_human/total_ai:.3f}")

print("\n" + "=" * 70)
print("改进效果对比")
print("=" * 70)

# 读取旧数据集（如果存在）
import glob
backup_dirs = sorted(glob.glob('datasets/bert_backup_*'), reverse=True)

if backup_dirs:
    backup_dir = backup_dirs[0]
    print(f"\n对比备份: {backup_dir}")

    try:
        old_full = pd.read_csv(f'{backup_dir}/full_dataset_labeled.csv', encoding='utf-8-sig')

        print(f"\n改进前:")
        print(f"  总样本数: {len(old_full)}")
        old_ai = (old_full['label'] == 1).sum()
        old_human = (old_full['label'] == 0).sum()
        print(f"  AI样本: {old_ai} (平均{old_full[old_full['label']==1]['length'].mean():.0f}字符)")
        print(f"  人类样本: {old_human} (平均{old_full[old_full['label']==0]['length'].mean():.0f}字符)")

        old_length_diff = abs(old_full[old_full['label']==1]['length'].mean() -
                             old_full[old_full['label']==0]['length'].mean())
        print(f"  长度差异: {old_length_diff:.0f}字符")

        # 短AI文本占比
        old_ai_df = old_full[old_full['label']==1]
        old_short_ai = (old_ai_df['length'] < 600).sum()
        print(f"  短AI文本(<600字): {old_short_ai} ({old_short_ai/len(old_ai_df)*100:.1f}%)")

    except Exception as e:
        print(f"  无法加载旧数据: {e}")

# 新数据集
new_full = pd.concat([train, val, test], ignore_index=True)
print(f"\n改进后:")
print(f"  总样本数: {len(new_full)}")
new_ai = (new_full['label'] == 1).sum()
new_human = (new_full['label'] == 0).sum()
print(f"  AI样本: {new_ai} (平均{new_full[new_full['label']==1]['length'].mean():.0f}字符)")
print(f"  人类样本: {new_human} (平均{new_full[new_full['label']==0]['length'].mean():.0f}字符)")

new_length_diff = abs(new_full[new_full['label']==1]['length'].mean() -
                     new_full[new_full['label']==0]['length'].mean())
print(f"  长度差异: {new_length_diff:.0f}字符")

# 短AI文本占比
new_ai_df = new_full[new_full['label']==1]
new_short_ai = (new_ai_df['length'] < 600).sum()
print(f"  短AI文本(<600字): {new_short_ai} ({new_short_ai/len(new_ai_df)*100:.1f}%)")

if backup_dirs:
    print(f"\n改进幅度:")
    if 'old_length_diff' in locals():
        improvement = (old_length_diff - new_length_diff) / old_length_diff * 100
        print(f"  长度差异减少: {improvement:.1f}%")

        short_ai_improvement = (new_short_ai/len(new_ai_df) - old_short_ai/len(old_ai_df)) * 100
        print(f"  短AI文本占比提升: {short_ai_improvement:.1f}个百分点")

print("\n" + "=" * 70)
print("✓ 验证完成")
print("=" * 70)
