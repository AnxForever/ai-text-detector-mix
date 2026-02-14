#!/usr/bin/env python3
"""
构建答辩聚焦数据集 - 从 core_v2 筛选中等长度(100-500字)高质量样本

策略：
1. 长度过滤：100-500字（自然长度，不截断）
2. 场景关键词筛选
3. 质量控制：去重、完整句子
4. 平衡控制：Human:AI ≈ 1:1
"""
import os
import sys
import json
import random
import pandas as pd
from pathlib import Path
from collections import defaultdict
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
random.seed(42)

# 场景关键词配置
SCENARIO_FILTERS = {
    'academic': {
        'keywords': ['论文', '研究', '分析', '方法', '结果', '实验', '数据', '模型', '本文', '探讨', '综述'],
        'target_human': 1000,
        'target_ai': 1000,
        'length_range': (200, 400),
        'priority': 'P0',
    },
    'student_work': {
        'keywords': ['我认为', '我觉得', '读后感', '总结', '心得体会', '学习', '思考', '理解'],
        'target_human': 1000,
        'target_ai': 500,
        'length_range': (150, 300),
        'priority': 'P0',
    },
    'news_comment': {
        'keywords': ['觉得', '个人观点', '评价', '推荐', '感觉', '挺好', '不错'],
        'target_human': 1000,
        'target_ai': 1000,
        'length_range': (100, 300),
        'priority': 'P1',
    },
    'formal_notice': {
        'keywords': ['通知', '公告', '提示', '注意', '规定', '要求', '请', '温馨提示'],
        'target_human': 800,
        'target_ai': 200,
        'length_range': (150, 400),
        'priority': 'P1',
    },
    'casual_ai': {
        'category_match': 'casual_ai',  # 使用 category 列匹配
        'target_human': 0,
        'target_ai': 1000,
        'length_range': (150, 300),
        'priority': 'P1',
    },
    'news_report': {
        'keywords': ['报道', '记者', '据悉', '消息', '发布', '表示', '宣布', '新闻'],
        'target_human': 750,
        'target_ai': 750,
        'length_range': (300, 500),
        'priority': 'P2',
    },
}

# 全局长度范围
MIN_LENGTH = 100
MAX_LENGTH = 500


def load_data():
    """加载 core_v2 数据"""
    data_path = PROJECT_ROOT / 'datasets' / 'active' / 'core_v2' / 'train.csv'
    if not data_path.exists():
        data_path = PROJECT_ROOT / 'datasets' / 'active' / 'core_v1' / 'train.csv'
        print(f"使用 core_v1: {data_path}")
    else:
        print(f"使用 core_v2: {data_path}")

    df = pd.read_csv(data_path, encoding='utf-8-sig')
    df['length'] = df['text'].str.len()
    df['label'] = df['label'].astype(int)

    print(f"加载数据: {len(df)} 条")
    print(f"  Human: {(df['label']==0).sum()}")
    print(f"  AI: {(df['label']==1).sum()}")

    return df


def filter_by_length(df, min_len=MIN_LENGTH, max_len=MAX_LENGTH):
    """按长度过滤"""
    filtered = df[(df['length'] >= min_len) & (df['length'] <= max_len)].copy()
    print(f"\n长度过滤 ({min_len}-{max_len}字): {len(filtered)} 条")
    return filtered


def quality_filter(df):
    """质量过滤"""
    original_len = len(df)

    # 去除截断文本（以...结尾或开头）
    df = df[~df['text'].str.strip().str.endswith('...')]
    df = df[~df['text'].str.strip().str.startswith('...')]

    # 去除过短句子（至少包含2个句号或问号或感叹号）
    def has_complete_sentences(text):
        punct_count = text.count('。') + text.count('？') + text.count('！')
        return punct_count >= 1

    df = df[df['text'].apply(has_complete_sentences)]

    # 去重
    df = df.drop_duplicates(subset=['text'])

    print(f"质量过滤: {original_len} -> {len(df)} 条 (去除 {original_len - len(df)})")
    return df


def match_scenario(row, scenario_config):
    """检查行是否匹配场景"""
    text = str(row['text'])
    length = row['length']

    # 长度范围检查
    min_len, max_len = scenario_config['length_range']
    if not (min_len <= length <= max_len):
        return False

    # 类别匹配（优先）
    if 'category_match' in scenario_config:
        if 'category' in row.index and row['category'] == scenario_config['category_match']:
            return True
        return False

    # 关键词匹配
    keywords = scenario_config.get('keywords', [])
    for kw in keywords:
        if kw in text:
            return True

    return False


def sample_scenario(df, scenario_name, config):
    """为指定场景采样数据"""
    target_human = config['target_human']
    target_ai = config['target_ai']

    # 筛选匹配的行
    matches = df.apply(lambda row: match_scenario(row, config), axis=1)
    matched_df = df[matches].copy()

    # 分离 Human 和 AI
    human_df = matched_df[matched_df['label'] == 0]
    ai_df = matched_df[matched_df['label'] == 1]

    # 采样
    sampled_human = human_df.sample(n=min(target_human, len(human_df)), random_state=42) if target_human > 0 and len(human_df) > 0 else pd.DataFrame()
    sampled_ai = ai_df.sample(n=min(target_ai, len(ai_df)), random_state=42) if target_ai > 0 and len(ai_df) > 0 else pd.DataFrame()

    # 添加场景标签
    if len(sampled_human) > 0:
        sampled_human = sampled_human.copy()
        sampled_human['scenario'] = scenario_name
    if len(sampled_ai) > 0:
        sampled_ai = sampled_ai.copy()
        sampled_ai['scenario'] = scenario_name

    result = pd.concat([sampled_human, sampled_ai], ignore_index=True)

    print(f"  {scenario_name}: {len(result)} 条 (H:{len(sampled_human)}, AI:{len(sampled_ai)})")
    print(f"    目标: H:{target_human}, AI:{target_ai}")
    print(f"    可用: H:{len(human_df)}, AI:{len(ai_df)}")

    return result


def build_dataset():
    """构建完整数据集"""
    print("=" * 60)
    print("构建答辩聚焦数据集")
    print("=" * 60)

    # 加载数据
    df = load_data()

    # 长度过滤
    df = filter_by_length(df)

    # 质量过滤
    df = quality_filter(df)

    # 按场景采样
    print("\n按场景采样:")
    all_samples = []
    used_indices = set()

    for scenario_name, config in SCENARIO_FILTERS.items():
        # 排除已使用的样本
        available_df = df[~df.index.isin(used_indices)]

        sampled = sample_scenario(available_df, scenario_name, config)
        all_samples.append(sampled)
        used_indices.update(sampled.index.tolist())

    # 合并所有样本
    final_df = pd.concat(all_samples, ignore_index=True)

    # 打乱顺序
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n总计: {len(final_df)} 条")
    print(f"  Human: {(final_df['label']==0).sum()}")
    print(f"  AI: {(final_df['label']==1).sum()}")

    return final_df


def split_dataset(df, train_ratio=0.8, val_ratio=0.1):
    """拆分数据集"""
    # 按场景分层拆分
    train_list = []
    val_list = []
    test_list = []

    for scenario in df['scenario'].unique():
        scenario_df = df[df['scenario'] == scenario]
        n = len(scenario_df)

        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        shuffled = scenario_df.sample(frac=1, random_state=42)
        train_list.append(shuffled.iloc[:n_train])
        val_list.append(shuffled.iloc[n_train:n_train+n_val])
        test_list.append(shuffled.iloc[n_train+n_val:])

    train_df = pd.concat(train_list, ignore_index=True).sample(frac=1, random_state=42)
    val_df = pd.concat(val_list, ignore_index=True).sample(frac=1, random_state=42)
    test_df = pd.concat(test_list, ignore_index=True).sample(frac=1, random_state=42)

    return train_df, val_df, test_df


def save_dataset(train_df, val_df, test_df):
    """保存数据集"""
    output_dir = PROJECT_ROOT / 'datasets' / 'defense_focused'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存 CSV
    train_df.to_csv(output_dir / 'train.csv', index=False, encoding='utf-8-sig')
    val_df.to_csv(output_dir / 'val.csv', index=False, encoding='utf-8-sig')
    test_df.to_csv(output_dir / 'test.csv', index=False, encoding='utf-8-sig')

    print(f"\n保存至 {output_dir}:")
    print(f"  train.csv: {len(train_df)} 条")
    print(f"  val.csv: {len(val_df)} 条")
    print(f"  test.csv: {len(test_df)} 条")

    # 保存元数据
    metadata = {
        'created_at': datetime.now().isoformat(),
        'source': 'core_v2',
        'length_range': [MIN_LENGTH, MAX_LENGTH],
        'scenarios': list(SCENARIO_FILTERS.keys()),
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'train_human': int((train_df['label']==0).sum()),
        'train_ai': int((train_df['label']==1).sum()),
        'scenario_distribution': train_df['scenario'].value_counts().to_dict(),
    }

    with open(output_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"  metadata.json: 已保存")

    return output_dir


def print_statistics(df, name="数据集"):
    """打印数据集统计"""
    print(f"\n{name} 统计:")
    print(f"  总条数: {len(df)}")
    print(f"  Human: {(df['label']==0).sum()} ({(df['label']==0).sum()/len(df)*100:.1f}%)")
    print(f"  AI: {(df['label']==1).sum()} ({(df['label']==1).sum()/len(df)*100:.1f}%)")

    print(f"\n  场景分布:")
    for scenario, count in df['scenario'].value_counts().items():
        h = ((df['scenario']==scenario) & (df['label']==0)).sum()
        a = ((df['scenario']==scenario) & (df['label']==1)).sum()
        print(f"    {scenario}: {count} 条 (H:{h}, AI:{a})")

    print(f"\n  长度分布:")
    for low, high in [(100, 200), (200, 300), (300, 400), (400, 500)]:
        subset = df[(df['length'] >= low) & (df['length'] < high)]
        print(f"    {low}-{high}字: {len(subset)} 条")


def main():
    # 构建数据集
    df = build_dataset()

    # 打印统计
    print_statistics(df, "筛选后数据集")

    # 拆分
    train_df, val_df, test_df = split_dataset(df)

    print("\n" + "=" * 60)
    print("数据集拆分")
    print("=" * 60)
    print(f"训练集: {len(train_df)} 条")
    print(f"验证集: {len(val_df)} 条")
    print(f"测试集: {len(test_df)} 条")

    # 保存
    output_dir = save_dataset(train_df, val_df, test_df)

    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)

    return output_dir


if __name__ == '__main__':
    main()
