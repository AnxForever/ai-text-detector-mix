"""
去除格式偏差：让AI和人类文本的格式分布一致

策略B（全部纯文本）- 当前使用：
1. 去除所有AI文本的markdown格式（AI markdown率降至0%）
2. 保持人类文本不变（人类markdown率保持~0%）
3. 彻底消除格式偏差，强制模型学习语义特征

原策略A（双向平衡）- 已弃用：
1. 随机去除50% AI文本的markdown格式（使AI markdown率降至~32%）
2. 为50%的人类文本添加markdown格式（使人类markdown率提升至~32%）
3. 确保两类文本的格式分布接近1:1
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
import re
import random
from datetime import datetime

# 导入增强的格式处理函数
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from format_handler import (
    remove_markdown_comprehensive,
    has_markdown,
    has_markdown_detailed,
    batch_remove_markdown,
    get_format_statistics
)

random.seed(42)

# 注意：remove_markdown, add_markdown, has_markdown 函数现在从 format_handler 导入
# 这里保留旧函数作为备份，但不再使用


def remove_all_markdown_strategy(df):
    """
    策略B：全部纯文本

    核心思路：
    - 去除所有AI文本的markdown格式
    - 保持人类文本不变（已经是纯文本）
    - 彻底消除格式偏差

    Args:
        df: 数据框

    Returns:
        处理后的数据框
    """

    print("="*60)
    print("策略B：全部纯文本去偏")
    print("="*60)
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 分离AI和人类文本
    ai_df = df[df['label'] == 1].copy()
    human_df = df[df['label'] == 0].copy()

    # 统计当前状态
    print("【当前状态】")
    ai_stats = get_format_statistics(ai_df['text'].tolist())
    human_stats = get_format_statistics(human_df['text'].tolist())

    print(f"AI文本: {len(ai_df)}条")
    print(f"  包含markdown: {ai_stats['has_markdown_count']}条 ({ai_stats['markdown_rate']*100:.1f}%)")
    print(f"  纯文本: {len(ai_df) - ai_stats['has_markdown_count']}条 ({(1-ai_stats['markdown_rate'])*100:.1f}%)")

    print(f"\n人类文本: {len(human_df)}条")
    print(f"  包含markdown: {human_stats['has_markdown_count']}条 ({human_stats['markdown_rate']*100:.1f}%)")
    print(f"  纯文本: {len(human_df) - human_stats['has_markdown_count']}条 ({(1-human_stats['markdown_rate'])*100:.1f}%)")

    current_bias = abs(ai_stats['markdown_rate'] - human_stats['markdown_rate'])
    print(f"\n格式偏差: {current_bias*100:.1f}%")

    # 处理AI文本：去除所有markdown
    print(f"\n[1/2] 去除所有AI文本的markdown...")
    ai_with_md = ai_df[ai_df['text'].apply(has_markdown)].copy()
    ai_without_md = ai_df[~ai_df['text'].apply(has_markdown)].copy()

    if len(ai_with_md) > 0:
        print(f"  发现 {len(ai_with_md)} 条包含markdown的AI文本")

        # 批量去除markdown（使用增强函数）
        print(f"  正在去除markdown格式...", flush=True)
        cleaned_texts = batch_remove_markdown(ai_with_md['text'].tolist(), show_progress=True)
        ai_with_md.loc[:, 'text'] = cleaned_texts

        # 合并
        ai_df = pd.concat([ai_with_md, ai_without_md])
        print(f"  ✓ 成功去除了 {len(ai_with_md)} 条AI文本的markdown格式")
    else:
        print(f"  ✓ 没有包含markdown的AI文本，无需处理")

    # 人类文本保持不变
    print(f"\n[2/2] 保持人类文本不变...")
    print(f"  ✓ 人类文本已经是纯文本（markdown率: {human_stats['markdown_rate']*100:.1f}%）")

    # 合并结果
    result_df = pd.concat([ai_df, human_df])

    # 验证结果
    print("\n【处理后状态】")
    ai_stats_new = get_format_statistics(ai_df['text'].tolist())
    human_stats_new = get_format_statistics(human_df['text'].tolist())

    print(f"AI文本包含markdown: {ai_stats_new['has_markdown_count']}条 ({ai_stats_new['markdown_rate']*100:.1f}%)")
    print(f"人类文本包含markdown: {human_stats_new['has_markdown_count']}条 ({human_stats_new['markdown_rate']*100:.1f}%)")

    new_bias = abs(ai_stats_new['markdown_rate'] - human_stats_new['markdown_rate'])
    print(f"\n格式偏差: {new_bias*100:.1f}%")

    if new_bias < 0.05:
        print("✅ 格式偏差<5%，已彻底消除！")
    elif new_bias < 0.10:
        print("✅ 格式偏差<10%，分布已平衡！")
    elif new_bias < 0.20:
        print("⚠️ 格式偏差10-20%，基本平衡")
    else:
        print("❌ 格式偏差仍然较大，需要进一步调整")

    # 格式统计对比
    print("\n【格式类型统计变化】")
    print("AI文本格式类型出现率:")
    for fmt_type in ['title', 'bold', 'list', 'code_block']:
        old_pct = ai_stats['format_type_percentages'].get(fmt_type, 0)
        new_pct = ai_stats_new['format_type_percentages'].get(fmt_type, 0)
        if old_pct > 0:
            print(f"  {fmt_type}: {old_pct:.1f}% → {new_pct:.1f}% (减少 {old_pct - new_pct:.1f}%)")

    return result_df


def balance_format_distribution(df, target_markdown_rate=0.35):
    """
    平衡AI和人类文本的格式分布

    Args:
        df: 数据框
        target_markdown_rate: 目标markdown比例（默认35%）

    Returns:
        处理后的数据框
    """

    print("="*60)
    print("去除格式偏差")
    print("="*60)
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 分离AI和人类文本
    ai_df = df[df['label'] == 1].copy()
    human_df = df[df['label'] == 0].copy()

    # 统计当前状态
    ai_md_count = ai_df['text'].apply(has_markdown).sum()
    human_md_count = human_df['text'].apply(has_markdown).sum()

    print("【当前状态】")
    print(f"AI文本: {len(ai_df)}条")
    print(f"  包含markdown: {ai_md_count}条 ({ai_md_count/len(ai_df)*100:.1f}%)")
    print(f"  纯文本: {len(ai_df) - ai_md_count}条 ({(1-ai_md_count/len(ai_df))*100:.1f}%)")

    print(f"\n人类文本: {len(human_df)}条")
    print(f"  包含markdown: {human_md_count}条 ({human_md_count/len(human_df)*100:.1f}%)")
    print(f"  纯文本: {len(human_df) - human_md_count}条 ({(1-human_md_count/len(human_df))*100:.1f}%)")

    print(f"\n格式偏差: {abs(ai_md_count/len(ai_df) - human_md_count/len(human_df))*100:.1f}%")

    # 计算目标数量
    target_ai_md = int(len(ai_df) * target_markdown_rate)
    target_human_md = int(len(human_df) * target_markdown_rate)

    print(f"\n【目标状态】（markdown率={target_markdown_rate*100:.0f}%）")
    print(f"AI文本需要包含markdown: {target_ai_md}条")
    print(f"人类文本需要包含markdown: {target_human_md}条")

    # 处理AI文本：去除部分markdown
    print(f"\n[1/2] 处理AI文本...")
    ai_with_md = ai_df[ai_df['text'].apply(has_markdown)].copy()
    ai_without_md = ai_df[~ai_df['text'].apply(has_markdown)].copy()

    remove_count = ai_md_count - target_ai_md

    if remove_count > 0:
        # 随机选择需要去除markdown的文本
        to_remove_md = ai_with_md.sample(n=min(remove_count, len(ai_with_md)), random_state=42)
        to_keep_md = ai_with_md[~ai_with_md.index.isin(to_remove_md.index)]

        # 去除markdown
        to_remove_md['text'] = to_remove_md['text'].apply(remove_markdown)

        # 合并
        ai_df = pd.concat([to_keep_md, to_remove_md, ai_without_md])
        print(f"  ✓ 去除了{len(to_remove_md)}条AI文本的markdown格式")
    else:
        print(f"  ✓ AI文本markdown比例已合适，无需调整")

    # 处理人类文本：添加markdown
    print(f"\n[2/2] 处理人类文本...")
    human_with_md = human_df[human_df['text'].apply(has_markdown)].copy()
    human_without_md = human_df[~human_df['text'].apply(has_markdown)].copy()

    add_count = target_human_md - human_md_count

    if add_count > 0:
        # 随机选择需要添加markdown的文本
        to_add_md = human_without_md.sample(n=min(add_count, len(human_without_md)), random_state=42)
        to_keep_plain = human_without_md[~human_without_md.index.isin(to_add_md.index)]

        # 添加markdown
        to_add_md['text'] = to_add_md['text'].apply(add_markdown)

        # 合并
        human_df = pd.concat([human_with_md, to_add_md, to_keep_plain])
        print(f"  ✓ 为{len(to_add_md)}条人类文本添加了markdown格式")
    else:
        print(f"  ✓ 人类文本markdown比例已合适，无需调整")

    # 合并并验证
    result_df = pd.concat([ai_df, human_df])

    # 验证结果
    print("\n【处理后状态】")
    ai_md_count_new = ai_df['text'].apply(has_markdown).sum()
    human_md_count_new = human_df['text'].apply(has_markdown).sum()

    print(f"AI文本包含markdown: {ai_md_count_new}条 ({ai_md_count_new/len(ai_df)*100:.1f}%)")
    print(f"人类文本包含markdown: {human_md_count_new}条 ({human_md_count_new/len(human_df)*100:.1f}%)")

    new_bias = abs(ai_md_count_new/len(ai_df) - human_md_count_new/len(human_df))
    print(f"\n格式偏差: {new_bias*100:.1f}%")

    if new_bias < 0.10:
        print("✅ 格式偏差<10%，分布已平衡！")
    elif new_bias < 0.20:
        print("⚠️ 格式偏差10-20%，基本平衡")
    else:
        print("❌ 格式偏差仍然较大，需要进一步调整")

    return result_df


def main():
    print("="*60)
    print("格式偏差去除工具 - 策略B")
    print("="*60)
    print("策略：去除所有AI文本的markdown，保持人类文本不变\n")

    # 读取原始数据集
    print("[1/4] 读取数据集...")
    train_df = pd.read_csv('datasets/bert/train.csv', encoding='utf-8-sig')
    val_df = pd.read_csv('datasets/bert/val.csv', encoding='utf-8-sig')
    test_df = pd.read_csv('datasets/bert/test.csv', encoding='utf-8-sig')

    print(f"  训练集: {len(train_df)}条")
    print(f"  验证集: {len(val_df)}条")
    print(f"  测试集: {len(test_df)}条")

    # 处理每个数据集（使用策略B）
    print("\n[2/4] 处理训练集（策略B：全部纯文本）...")
    train_debiased = remove_all_markdown_strategy(train_df)

    print("\n[3/4] 处理验证集（策略B：全部纯文本）...")
    val_debiased = remove_all_markdown_strategy(val_df)

    print("\n[4/4] 处理测试集（策略B：全部纯文本）...")
    test_debiased = remove_all_markdown_strategy(test_df)

    # 保存结果
    print("\n" + "="*60)
    print("保存去偏后的数据集")
    print("="*60)

    os.makedirs('datasets/bert_debiased', exist_ok=True)

    train_debiased.to_csv('datasets/bert_debiased/train.csv', index=False, encoding='utf-8-sig')
    val_debiased.to_csv('datasets/bert_debiased/val.csv', index=False, encoding='utf-8-sig')
    test_debiased.to_csv('datasets/bert_debiased/test.csv', index=False, encoding='utf-8-sig')

    print(f"✓ 训练集已保存: datasets/bert_debiased/train.csv")
    print(f"✓ 验证集已保存: datasets/bert_debiased/val.csv")
    print(f"✓ 测试集已保存: datasets/bert_debiased/test.csv")

    # 总结统计
    print("\n" + "="*60)
    print("总体统计")
    print("="*60)

    total_original = len(train_df) + len(val_df) + len(test_df)
    total_debiased = len(train_debiased) + len(val_debiased) + len(test_debiased)

    print(f"总样本数: {total_debiased}条 (原始: {total_original}条)")

    # 最终格式偏差统计
    all_debiased = pd.concat([train_debiased, val_debiased, test_debiased])
    ai_debiased = all_debiased[all_debiased['label'] == 1]
    human_debiased = all_debiased[all_debiased['label'] == 0]

    ai_md_rate = ai_debiased['text'].apply(has_markdown).sum() / len(ai_debiased)
    human_md_rate = human_debiased['text'].apply(has_markdown).sum() / len(human_debiased)
    final_bias = abs(ai_md_rate - human_md_rate)

    print(f"\n最终格式偏差: {final_bias*100:.2f}%")
    print(f"  AI markdown率: {ai_md_rate*100:.2f}%")
    print(f"  人类 markdown率: {human_md_rate*100:.2f}%")

    if final_bias < 0.05:
        print("\n✅ 格式偏差<5%，彻底消除！模型将被迫学习语义特征。")
    else:
        print(f"\n⚠️ 格式偏差={final_bias*100:.1f}%，可能需要检查")

    print("\n" + "="*60)
    print("完成！")
    print("="*60)
    print("\n下一步：")
    print("1. 使用 format_bias_check.py 验证格式偏差")
    print("2. 使用去偏后的数据集重新训练模型")
    print("   命令: python scripts/training/train_bert_improved.py --use_debiased")



if __name__ == "__main__":
    main()
