"""
数据集重建脚本（长度平衡版本）
合并所有AI文本（原始 + 新生成短文本 + 提取片段）
与人类文本合并后进行长度分层划分
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
import json
import shutil
from datetime import datetime
from pathlib import Path


def backup_existing_dataset():
    """备份现有数据集"""
    bert_dir = Path("datasets/bert")

    if bert_dir.exists() and any(bert_dir.glob("*.csv")):
        backup_dir = Path(f"datasets/bert_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        backup_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n备份现有数据集到: {backup_dir}", flush=True)

        for file in bert_dir.glob("*.csv"):
            shutil.copy(file, backup_dir / file.name)
            print(f"  ✓ 已备份: {file.name}", flush=True)

        for file in bert_dir.glob("*.json"):
            shutil.copy(file, backup_dir / file.name)
            print(f"  ✓ 已备份: {file.name}", flush=True)

        print(f"  ✓ 备份完成", flush=True)
        return backup_dir
    else:
        print("\n无需备份（未找到现有数据集）", flush=True)
        return None


def load_ai_texts():
    """加载所有AI文本"""

    print("\n" + "=" * 70, flush=True)
    print("加载AI文本数据", flush=True)
    print("=" * 70, flush=True)

    ai_datasets = []

    # 1. 原始AI文本（9,170条）
    print("\n[1/3] 加载原始AI文本...", flush=True)
    original_file = "datasets/final/parallel_dataset_cleaned.csv"

    if not os.path.exists(original_file):
        print(f"  ✗ 文件不存在: {original_file}", flush=True)
        raise FileNotFoundError(f"未找到原始AI文本: {original_file}")

    df_original = pd.read_csv(original_file, encoding='utf-8-sig')

    # 标准化列名
    if 'text_content' in df_original.columns:
        df_original = df_original.rename(columns={'text_content': 'text'})

    df_original['source_type'] = 'original'
    ai_datasets.append(df_original)

    print(f"  ✓ 原始AI文本: {len(df_original)} 条", flush=True)
    print(f"    平均长度: {df_original['length'].mean():.0f} 字符", flush=True)

    # 2. 新生成的短AI文本（3,000条）
    print("\n[2/3] 加载新生成的短AI文本...", flush=True)
    short_text_file = "datasets/short_ai_texts/short_ai_texts_3000.csv"

    if os.path.exists(short_text_file):
        df_short = pd.read_csv(short_text_file, encoding='utf-8-sig')

        # 标准化列名
        if 'text_content' in df_short.columns:
            df_short = df_short.rename(columns={'text_content': 'text'})

        df_short['source_type'] = 'generated_short'

        # 确保有必要的列
        if 'topic' not in df_short.columns:
            df_short['topic'] = df_short.get('topic', '未知')
        if 'quality' not in df_short.columns:
            df_short['quality'] = 1.0

        ai_datasets.append(df_short)

        print(f"  ✓ 短AI文本: {len(df_short)} 条", flush=True)
        print(f"    平均长度: {df_short['length'].mean():.0f} 字符", flush=True)
    else:
        print(f"  ⚠ 未找到短AI文本文件: {short_text_file}", flush=True)
        print(f"    请先运行 generate_short_ai_texts.py", flush=True)

    # 3. 提取的短片段（2,000条）
    print("\n[3/3] 加载提取的短片段...", flush=True)
    segments_file = "datasets/short_ai_texts/extracted_segments_2000.csv"

    if os.path.exists(segments_file):
        df_segments = pd.read_csv(segments_file, encoding='utf-8-sig')

        # 标准化列名
        if 'text_content' in df_segments.columns:
            df_segments = df_segments.rename(columns={'text_content': 'text'})

        df_segments['source_type'] = 'extracted_segment'

        # 确保有必要的列
        if 'topic' not in df_segments.columns:
            df_segments['topic'] = df_segments.get('original_topic', '未知')
        if 'quality' not in df_segments.columns:
            df_segments['quality'] = 1.0

        ai_datasets.append(df_segments)

        print(f"  ✓ 提取片段: {len(df_segments)} 条", flush=True)
        print(f"    平均长度: {df_segments['length'].mean():.0f} 字符", flush=True)
    else:
        print(f"  ⚠ 未找到片段文件: {segments_file}", flush=True)
        print(f"    请先运行 extract_segments.py", flush=True)

    # 合并所有AI文本
    print("\n合并所有AI文本...", flush=True)
    df_ai_all = pd.concat(ai_datasets, ignore_index=True)

    # 确保所有必要的列都存在
    required_columns = ['text', 'length', 'source_api', 'attribute', 'topic', 'quality', 'source_type']
    for col in required_columns:
        if col not in df_ai_all.columns:
            if col == 'text':
                raise ValueError("缺少必要的'text'列")
            elif col == 'length':
                df_ai_all['length'] = df_ai_all['text'].str.len()
            elif col == 'source_api':
                df_ai_all['source_api'] = 'unknown'
            elif col == 'attribute':
                df_ai_all['attribute'] = '说明'
            elif col == 'topic':
                df_ai_all['topic'] = '未知'
            elif col == 'quality':
                df_ai_all['quality'] = 1.0

    print(f"  ✓ 合并完成: {len(df_ai_all)} 条AI文本", flush=True)
    print(f"    平均长度: {df_ai_all['length'].mean():.0f} ± {df_ai_all['length'].std():.0f}", flush=True)

    # 长度分布统计
    print(f"\n  长度分布:", flush=True)
    bins = [0, 300, 600, 1000, 1500, 2000, 3000, 10000]
    hist = pd.cut(df_ai_all['length'], bins=bins).value_counts().sort_index()
    for interval, count in hist.items():
        print(f"    {interval}: {count} ({count/len(df_ai_all)*100:.1f}%)", flush=True)

    # 来源统计
    print(f"\n  来源分布:", flush=True)
    for source, count in df_ai_all['source_type'].value_counts().items():
        print(f"    {source}: {count} ({count/len(df_ai_all)*100:.1f}%)", flush=True)

    return df_ai_all


def load_human_texts():
    """加载人类文本"""

    print("\n" + "=" * 70, flush=True)
    print("加载人类文本数据", flush=True)
    print("=" * 70, flush=True)

    human_file = "datasets/human_texts/thucnews_real_human_9000.csv"

    if not os.path.exists(human_file):
        print(f"  ✗ 文件不存在: {human_file}", flush=True)
        raise FileNotFoundError(f"未找到人类文本: {human_file}")

    df_human = pd.read_csv(human_file, encoding='utf-8-sig')

    print(f"  ✓ 人类文本: {len(df_human)} 条", flush=True)
    print(f"    平均长度: {df_human['length'].mean():.0f} 字符", flush=True)

    # 长度分布统计
    print(f"\n  长度分布:", flush=True)
    bins = [0, 300, 600, 1000, 1500, 2000, 3000, 10000]
    hist = pd.cut(df_human['length'], bins=bins).value_counts().sort_index()
    for interval, count in hist.items():
        print(f"    {interval}: {count} ({count/len(df_human)*100:.1f}%)", flush=True)

    return df_human


def merge_and_label_dataset(df_ai, df_human):
    """合并并标注数据集"""

    print("\n" + "=" * 70, flush=True)
    print("合并并标注数据集", flush=True)
    print("=" * 70, flush=True)

    # AI文本 -> label=1
    ai_labeled = pd.DataFrame({
        'text': df_ai['text'],
        'label': 1,  # AI生成
        'source': 'ai_generated',
        'source_detail': df_ai['source_api'],
        'source_type': df_ai['source_type'],
        'attribute': df_ai['attribute'],
        'topic': df_ai['topic'],
        'length': df_ai['length'],
        'quality': df_ai.get('quality', 1.0)
    })

    # 人类文本 -> label=0
    human_labeled = pd.DataFrame({
        'text': df_human['text'],
        'label': 0,  # 人类撰写
        'source': 'human_written',
        'source_detail': df_human['source'],
        'source_type': 'real_news',
        'attribute': df_human['attribute'],
        'topic': df_human['topic'],
        'length': df_human['length'],
        'quality': 1.0
    })

    print(f"  ✓ AI文本标注: label=1, {len(ai_labeled)} 条", flush=True)
    print(f"  ✓ 人类文本标注: label=0, {len(human_labeled)} 条", flush=True)

    # 合并
    full_dataset = pd.concat([ai_labeled, human_labeled], ignore_index=True)

    # 打乱顺序
    full_dataset = full_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n  ✓ 合并完成: {len(full_dataset)} 条", flush=True)

    # 统计信息
    print(f"\n总样本数: {len(full_dataset)}", flush=True)
    print(f"  AI样本 (label=1): {(full_dataset['label']==1).sum()} 条 ({(full_dataset['label']==1).sum()/len(full_dataset)*100:.1f}%)", flush=True)
    print(f"  人类样本 (label=0): {(full_dataset['label']==0).sum()} 条 ({(full_dataset['label']==0).sum()/len(full_dataset)*100:.1f}%)", flush=True)

    # 类别平衡检查
    balance_ratio = (full_dataset['label']==0).sum() / (full_dataset['label']==1).sum()
    print(f"\n类别平衡比: {balance_ratio:.3f}", flush=True)

    # 长度分布对比
    print(f"\n长度分布对比:", flush=True)
    print(f"  AI文本平均长度: {full_dataset[full_dataset['label']==1]['length'].mean():.0f} 字符", flush=True)
    print(f"  人类文本平均长度: {full_dataset[full_dataset['label']==0]['length'].mean():.0f} 字符", flush=True)

    length_diff = abs(full_dataset[full_dataset['label']==1]['length'].mean() -
                     full_dataset[full_dataset['label']==0]['length'].mean())
    print(f"  长度差异: {length_diff:.0f} 字符", flush=True)

    # 每个长度区间的平衡性
    print(f"\n各长度区间的AI/人类比例:", flush=True)
    bins = [0, 300, 600, 1000, 1500, 2000, 3000, 10000]
    full_dataset['length_bin'] = pd.cut(full_dataset['length'], bins=bins)

    for bin_range in full_dataset['length_bin'].cat.categories:
        bin_df = full_dataset[full_dataset['length_bin'] == bin_range]
        if len(bin_df) > 0:
            ai_count = (bin_df['label'] == 1).sum()
            human_count = (bin_df['label'] == 0).sum()
            ratio = human_count / ai_count if ai_count > 0 else 0
            print(f"  {bin_range}: AI={ai_count}, 人类={human_count}, 比例={ratio:.2f}", flush=True)

    full_dataset = full_dataset.drop('length_bin', axis=1)

    return full_dataset


def save_full_dataset(df):
    """保存完整数据集"""

    print("\n" + "=" * 70, flush=True)
    print("保存完整数据集", flush=True)
    print("=" * 70, flush=True)

    output_dir = "datasets/bert"
    os.makedirs(output_dir, exist_ok=True)

    # 保存CSV
    filename = f"{output_dir}/full_dataset_labeled.csv"
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"  ✓ 已保存: {filename}", flush=True)

    # 保存元数据
    metadata = {
        'name': 'BERT二分类数据集（AI vs 人类）- 长度平衡版',
        'created_at': datetime.now().isoformat(),
        'version': '2.0_length_balanced',
        'total_samples': len(df),
        'ai_samples': int((df['label']==1).sum()),
        'human_samples': int((df['label']==0).sum()),
        'balance_ratio': float((df['label']==0).sum() / (df['label']==1).sum()),
        'avg_length_ai': float(df[df['label']==1]['length'].mean()),
        'avg_length_human': float(df[df['label']==0]['length'].mean()),
        'length_difference': float(abs(df[df['label']==1]['length'].mean() - df[df['label']==0]['length'].mean())),
        'ai_source_distribution': df[df['label']==1]['source_type'].value_counts().to_dict(),
        'improvement': 'Added 3000 short AI texts and 2000 extracted segments to balance length distribution'
    }

    metadata_file = f"{output_dir}/dataset_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"  ✓ 元数据已保存: {metadata_file}", flush=True)


def rebuild_dataset():
    """重建数据集的主函数"""

    print("\n" + "=" * 70, flush=True)
    print("数据集重建程序（长度平衡版本）", flush=True)
    print("=" * 70, flush=True)
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    # 1. 备份现有数据集
    backup_dir = backup_existing_dataset()

    try:
        # 2. 加载所有AI文本
        df_ai = load_ai_texts()

        # 3. 加载人类文本
        df_human = load_human_texts()

        # 4. 合并并标注
        full_dataset = merge_and_label_dataset(df_ai, df_human)

        # 5. 保存完整数据集
        save_full_dataset(full_dataset)

        print("\n" + "=" * 70, flush=True)
        print("数据集重建完成！", flush=True)
        print("=" * 70, flush=True)

        print("\n下一步:", flush=True)
        print("  1. 运行 split_dataset.py 进行长度分层划分", flush=True)
        print("  2. 检查各长度区间的平衡性", flush=True)
        print("  3. 开始BERT训练", flush=True)

        return full_dataset

    except Exception as e:
        print(f"\n✗ 错误: {e}", flush=True)
        if backup_dir:
            print(f"\n备份位置: {backup_dir}", flush=True)
            print("如需恢复，请将备份文件复制回 datasets/bert/", flush=True)
        raise


if __name__ == "__main__":
    dataset = rebuild_dataset()
