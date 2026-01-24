"""
数据集划分脚本（长度分层版本）
将完整数据集划分为train/val/test三个子集
使用长度分层抽样，确保每个长度区间内AI和人类文本平衡
"""

import sys
import io
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except:
        pass

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from datetime import datetime
import argparse


def split_dataset_with_length_stratification(
    df,
    length_bins=[300, 600, 1000, 1500, 2000, 3000, 10000],
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    balance_within_bins=True,
    random_state=42
):
    """
    按长度分层划分数据集

    Args:
        df: 完整数据集
        length_bins: 长度区间边界
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        balance_within_bins: 是否在每个长度区间内平衡AI和人类样本
        random_state: 随机种子

    Returns:
        train_df, val_df, test_df
    """
    print("\n[长度分层采样策略]", flush=True)
    print(f"  长度区间: {length_bins}", flush=True)
    print(f"  区间内平衡: {'是' if balance_within_bins else '否'}", flush=True)

    # 添加长度区间列
    df['length_bin'] = pd.cut(df['length'], bins=length_bins, include_lowest=True)

    train_dfs = []
    val_dfs = []
    test_dfs = []

    # 统计信息
    bin_stats = []

    print("\n  各长度区间处理：", flush=True)

    for bin_range in df['length_bin'].cat.categories:
        bin_df = df[df['length_bin'] == bin_range].copy()

        if len(bin_df) == 0:
            continue

        print(f"\n  区间 {bin_range}:", flush=True)
        print(f"    原始样本数: {len(bin_df)}", flush=True)

        # 获取AI和人类文本
        ai_df = bin_df[bin_df['label'] == 1]
        human_df = bin_df[bin_df['label'] == 0]

        print(f"    AI: {len(ai_df)}, 人类: {len(human_df)}", flush=True)

        if balance_within_bins and len(ai_df) > 0 and len(human_df) > 0:
            # 欠采样到相同数量
            min_count = min(len(ai_df), len(human_df))

            if min_count < 10:  # 样本太少，跳过该区间
                print(f"    ⚠ 样本太少(<10)，跳过此区间", flush=True)
                continue

            ai_sampled = ai_df.sample(n=min_count, random_state=random_state)
            human_sampled = human_df.sample(n=min_count, random_state=random_state)

            balanced_bin = pd.concat([ai_sampled, human_sampled], ignore_index=True)

            print(f"    平衡后: AI={min_count}, 人类={min_count}, 总计={len(balanced_bin)}", flush=True)
        else:
            balanced_bin = bin_df
            print(f"    未平衡: 总计={len(balanced_bin)}", flush=True)

        if len(balanced_bin) < 10:
            print(f"    ⚠ 样本太少，跳过此区间", flush=True)
            continue

        # 在该区间内进行70/15/15划分
        train_bin, temp_bin = train_test_split(
            balanced_bin,
            test_size=(val_ratio + test_ratio),
            stratify=balanced_bin['label'],
            random_state=random_state
        )

        val_bin, test_bin = train_test_split(
            temp_bin,
            test_size=test_ratio / (val_ratio + test_ratio),
            stratify=temp_bin['label'],
            random_state=random_state
        )

        train_dfs.append(train_bin)
        val_dfs.append(val_bin)
        test_dfs.append(test_bin)

        # 统计信息
        bin_stats.append({
            'bin': str(bin_range),
            'original_count': len(bin_df),
            'balanced_count': len(balanced_bin),
            'train_count': len(train_bin),
            'val_count': len(val_bin),
            'test_count': len(test_bin)
        })

        print(f"    划分: train={len(train_bin)}, val={len(val_bin)}, test={len(test_bin)}", flush=True)

    # 合并所有区间
    train_df = pd.concat(train_dfs, ignore_index=True).sample(frac=1, random_state=random_state)
    val_df = pd.concat(val_dfs, ignore_index=True).sample(frac=1, random_state=random_state)
    test_df = pd.concat(test_dfs, ignore_index=True).sample(frac=1, random_state=random_state)

    # 移除临时列
    train_df = train_df.drop('length_bin', axis=1)
    val_df = val_df.drop('length_bin', axis=1)
    test_df = test_df.drop('length_bin', axis=1)

    print("\n  ✓ 长度分层采样完成", flush=True)

    return train_df, val_df, test_df, bin_stats


def split_dataset(input_file="datasets/bert/full_dataset_labeled.csv", output_dir="datasets/bert", seed=42):
    """划分数据集"""

    print("=" * 60, flush=True)
    print("BERT数据集划分程序", flush=True)
    print("=" * 60, flush=True)
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    # 1. 加载完整数据集
    print("\n[1/4] 加载数据集...", flush=True)
    full_dataset = pd.read_csv(input_file, encoding='utf-8-sig')
    print(f"  ✓ 总样本数: {len(full_dataset)}", flush=True)
    print(f"    AI样本: {(full_dataset['label']==1).sum()}", flush=True)
    print(f"    人类样本: {(full_dataset['label']==0).sum()}", flush=True)

    # 2. 按长度分层划分
    print("\n[2/4] 按长度分层划分数据集...", flush=True)
    print("  划分比例: train=70%, val=15%, test=15%", flush=True)
    print("  策略: 在每个长度区间内确保AI和人类样本平衡", flush=True)

    # 使用长度分层采样
    train_df, val_df, test_df, bin_stats = split_dataset_with_length_stratification(
        full_dataset,
        length_bins=[300, 600, 1000, 1500, 2000, 3000, 10000],
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        balance_within_bins=True,
        random_state=seed
    )

    print(f"  ✓ 训练集: {len(train_df)} 条 ({len(train_df)/len(full_dataset)*100:.1f}%)", flush=True)
    print(f"  ✓ 验证集: {len(val_df)} 条 ({len(val_df)/len(full_dataset)*100:.1f}%)", flush=True)
    print(f"  ✓ 测试集: {len(test_df)} 条 ({len(test_df)/len(full_dataset)*100:.1f}%)", flush=True)

    # 3. 验证类别平衡
    print("\n[3/4] 验证类别平衡...", flush=True)

    for name, df in [('训练集', train_df), ('验证集', val_df), ('测试集', test_df)]:
        ai_count = (df['label']==1).sum()
        human_count = (df['label']==0).sum()
        ai_ratio = ai_count / len(df) * 100

        print(f"\n  {name}:", flush=True)
        print(f"    总计: {len(df)}", flush=True)
        print(f"    AI样本: {ai_count} ({ai_ratio:.1f}%)", flush=True)
        print(f"    人类样本: {human_count} ({100-ai_ratio:.1f}%)", flush=True)

        # 检查平衡性
        balance_ratio = human_count / ai_count
        if 0.9 <= balance_ratio <= 1.1:
            print(f"    ✓ 类别平衡良好 (比例: {balance_ratio:.3f})", flush=True)
        else:
            print(f"    ⚠ 类别可能不平衡 (比例: {balance_ratio:.3f})", flush=True)

    # 属性分布检查
    print(f"\n  属性分布检查:", flush=True)
    for name, df in [('训练集', train_df), ('验证集', val_df), ('测试集', test_df)]:
        print(f"\n  {name}属性分布:", flush=True)
        attr_dist = df['attribute'].value_counts(normalize=True) * 100
        for attr, pct in attr_dist.items():
            print(f"    {attr}: {pct:.1f}%", flush=True)

    # 4. 保存数据集
    print("\n[4/4] 保存数据集...", flush=True)
    print("=" * 60, flush=True)

    os.makedirs(output_dir, exist_ok=True)

    # 保存为CSV
    train_df.to_csv(f"{output_dir}/train.csv", index=False, encoding='utf-8-sig')
    val_df.to_csv(f"{output_dir}/val.csv", index=False, encoding='utf-8-sig')
    test_df.to_csv(f"{output_dir}/test.csv", index=False, encoding='utf-8-sig')

    print(f"✓ 训练集已保存: {output_dir}/train.csv", flush=True)
    print(f"✓ 验证集已保存: {output_dir}/val.csv", flush=True)
    print(f"✓ 测试集已保存: {output_dir}/test.csv", flush=True)

    # 保存划分统计信息
    split_stats = {
        'split_date': datetime.now().isoformat(),
        'split_method': 'length_stratified',
        'length_bins': [300, 600, 1000, 1500, 2000, 3000, 10000],
        'random_seed': seed,
        'balance_within_bins': True,
        'bin_statistics': bin_stats,
        'train': {
            'count': len(train_df),
            'percentage': float(len(train_df)/len(full_dataset)*100),
            'ai_samples': int((train_df['label']==1).sum()),
            'human_samples': int((train_df['label']==0).sum()),
            'balance_ratio': float((train_df['label']==0).sum() / (train_df['label']==1).sum()),
            'avg_length': float(train_df['length'].mean()),
            'length_std': float(train_df['length'].std())
        },
        'val': {
            'count': len(val_df),
            'percentage': float(len(val_df)/len(full_dataset)*100),
            'ai_samples': int((val_df['label']==1).sum()),
            'human_samples': int((val_df['label']==0).sum()),
            'balance_ratio': float((val_df['label']==0).sum() / (val_df['label']==1).sum()),
            'avg_length': float(val_df['length'].mean()),
            'length_std': float(val_df['length'].std())
        },
        'test': {
            'count': len(test_df),
            'percentage': float(len(test_df)/len(full_dataset)*100),
            'ai_samples': int((test_df['label']==1).sum()),
            'human_samples': int((test_df['label']==0).sum()),
            'balance_ratio': float((test_df['label']==0).sum() / (test_df['label']==1).sum()),
            'avg_length': float(test_df['length'].mean()),
            'length_std': float(test_df['length'].std())
        }
    }

    stats_file = f"{output_dir}/split_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(split_stats, f, ensure_ascii=False, indent=2)
    print(f"✓ 统计信息已保存: {stats_file}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("数据集划分完成！", flush=True)
    print("=" * 60, flush=True)

    # 打印文件清单
    print("\n已生成文件:", flush=True)
    print(f"  1. train.csv ({len(train_df):,} 条)", flush=True)
    print(f"  2. val.csv ({len(val_df):,} 条)", flush=True)
    print(f"  3. test.csv ({len(test_df):,} 条)", flush=True)
    print(f"  4. split_stats.json (统计信息)", flush=True)

    print("\n下一步:", flush=True)
    print("  1. 转换为BERT输入格式（Tokenization）", flush=True)
    print("  2. 创建PyTorch Dataset和DataLoader", flush=True)
    print("  3. 加载BERT模型 (bert-base-chinese)", flush=True)
    print("  4. 开始训练", flush=True)

    return train_df, val_df, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test with length stratification.")
    parser.add_argument("--input", default="datasets/bert/full_dataset_labeled.csv", help="Full labeled dataset CSV")
    parser.add_argument("--output-dir", default="datasets/bert", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train, val, test = split_dataset(
        input_file=args.input,
        output_dir=args.output_dir,
        seed=args.seed
    )
