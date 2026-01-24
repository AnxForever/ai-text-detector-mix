"""
数据标注和合并脚本
将AI生成文本和人类文本合并，添加标签
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
import argparse
from datetime import datetime

def _select_text_column(df: pd.DataFrame, fallback_label: str) -> pd.Series:
    if 'text_content' in df.columns:
        return df['text_content']
    if 'text' in df.columns:
        return df['text']
    raise ValueError(f"{fallback_label} missing text column")


def load_and_label_datasets(ai_file: str, human_file: str, output_dir: str, seed: int = 42):
    """加载并标注数据集"""

    print("=" * 60, flush=True)
    print("BERT二分类数据集构建程序", flush=True)
    print("=" * 60, flush=True)
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    # 1. 加载AI生成文本
    print("\n[1/5] 加载AI生成文本...", flush=True)
    ai_df = pd.read_csv(ai_file, encoding='utf-8-sig')
    print(f"  ✓ AI文本: {len(ai_df)} 条", flush=True)
    print(f"    平均长度: {ai_df['length'].mean():.0f} 字符", flush=True)
    print(f"    长度范围: {ai_df['length'].min()} - {ai_df['length'].max()}", flush=True)

    # 2. 加载人类文本（真实数据）
    print("\n[2/5] 加载人类文本（THUCNews真实数据）...", flush=True)
    human_df = pd.read_csv(human_file, encoding='utf-8-sig')
    print(f"  ✓ 人类文本: {len(human_df)} 条", flush=True)
    print(f"    平均长度: {human_df['length'].mean():.0f} 字符", flush=True)
    print(f"    长度范围: {human_df['length'].min()} - {human_df['length'].max()}", flush=True)

    # 3. 标注数据
    print("\n[3/5] 添加标签...", flush=True)

    # AI文本 -> label=1
    ai_text = _select_text_column(ai_df, "AI dataset")
    ai_quality = ai_df['generation_quality'] if 'generation_quality' in ai_df.columns else 1.0
    ai_labeled = pd.DataFrame({
        'text': ai_text,
        'label': 1,  # AI生成
        'source': 'ai_generated',
        'source_detail': ai_df.get('source_api', 'unknown'),
        'attribute': ai_df.get('attribute', '说明'),
        'topic': ai_df.get('topic', '未知'),
        'length': ai_df.get('length', ai_text.astype(str).str.len()),
        'quality': ai_quality
    })

    # 人类文本 -> label=0
    human_text = _select_text_column(human_df, "Human dataset")
    human_labeled = pd.DataFrame({
        'text': human_text,
        'label': 0,  # 人类撰写
        'source': 'human_written',
        'source_detail': human_df.get('source', 'unknown'),
        'attribute': human_df.get('attribute', '说明'),
        'topic': human_df.get('topic', '未知'),
        'length': human_df.get('length', human_text.astype(str).str.len()),
        'quality': 1.0  # 默认高质量
    })

    print(f"  ✓ AI文本标注: label=1", flush=True)
    print(f"  ✓ 人类文本标注: label=0", flush=True)

    # 4. 合并数据集
    print("\n[4/5] 合并数据集...", flush=True)
    full_dataset = pd.concat([ai_labeled, human_labeled], ignore_index=True)

    # 打乱顺序
    full_dataset = full_dataset.sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"  ✓ 合并完成: {len(full_dataset)} 条", flush=True)

    # 5. 统计信息
    print("\n[5/5] 数据集统计...", flush=True)
    print(f"\n总样本数: {len(full_dataset)}", flush=True)
    print(f"  AI样本 (label=1): {(full_dataset['label']==1).sum()} 条 ({(full_dataset['label']==1).sum()/len(full_dataset)*100:.1f}%)", flush=True)
    print(f"  人类样本 (label=0): {(full_dataset['label']==0).sum()} 条 ({(full_dataset['label']==0).sum()/len(full_dataset)*100:.1f}%)", flush=True)

    # 类别平衡检查
    balance_ratio = (full_dataset['label']==0).sum() / (full_dataset['label']==1).sum()
    print(f"\n类别平衡比: {balance_ratio:.3f} (推荐范围: 0.8-1.2)", flush=True)
    if 0.8 <= balance_ratio <= 1.2:
        print("  ✓ 类别平衡良好", flush=True)
    else:
        print("  ⚠ 类别可能存在不平衡", flush=True)

    # 长度分布对比
    print(f"\n长度分布:", flush=True)
    print(f"  AI文本平均长度: {full_dataset[full_dataset['label']==1]['length'].mean():.0f} 字符", flush=True)
    print(f"  人类文本平均长度: {full_dataset[full_dataset['label']==0]['length'].mean():.0f} 字符", flush=True)

    length_diff = abs(full_dataset[full_dataset['label']==1]['length'].mean() -
                     full_dataset[full_dataset['label']==0]['length'].mean())
    print(f"  长度差异: {length_diff:.0f} 字符", flush=True)

    if length_diff < 200:
        print("  ✓ 长度分布相近，利于模型学习语义特征", flush=True)
    elif length_diff < 500:
        print("  ⚠ 长度分布有一定差异", flush=True)
    else:
        print("  ⚠ 长度分布差异较大，模型可能会过度依赖长度特征", flush=True)

    # 属性分布对比
    print(f"\n属性分布:", flush=True)
    print(f"  AI文本:", flush=True)
    ai_attr = full_dataset[full_dataset['label']==1]['attribute'].value_counts()
    for attr, count in ai_attr.items():
        print(f"    {attr}: {count} ({count/len(full_dataset[full_dataset['label']==1])*100:.1f}%)", flush=True)

    print(f"  人类文本:", flush=True)
    human_attr = full_dataset[full_dataset['label']==0]['attribute'].value_counts()
    for attr, count in human_attr.items():
        print(f"    {attr}: {count} ({count/len(full_dataset[full_dataset['label']==0])*100:.1f}%)", flush=True)

    # 6. 保存合并后的数据集
    print("\n" + "=" * 60, flush=True)
    print("保存数据集...", flush=True)
    print("=" * 60, flush=True)

    os.makedirs(output_dir, exist_ok=True)

    filename = f"{output_dir}/full_dataset_labeled.csv"
    full_dataset.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"✓ 已保存到: {filename}", flush=True)

    # 保存元数据
    metadata = {
        'name': 'BERT二分类数据集（AI vs 人类）',
        'created_at': datetime.now().isoformat(),
        'total_samples': len(full_dataset),
        'ai_samples': int((full_dataset['label']==1).sum()),
        'human_samples': int((full_dataset['label']==0).sum()),
        'balance_ratio': float(balance_ratio),
        'avg_length_ai': float(full_dataset[full_dataset['label']==1]['length'].mean()),
        'avg_length_human': float(full_dataset[full_dataset['label']==0]['length'].mean()),
        'length_difference': float(length_diff),
        'attributes': {
            'ai': ai_attr.to_dict(),
            'human': human_attr.to_dict()
        }
    }

    metadata_file = f"{output_dir}/dataset_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"✓ 元数据已保存到: {metadata_file}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("数据标注和合并完成！", flush=True)
    print("=" * 60, flush=True)

    print("\n下一步:", flush=True)
    print("  1. 划分train/val/test集（70%/15%/15%）", flush=True)
    print("  2. 转换为BERT输入格式", flush=True)
    print("  3. 创建PyTorch DataLoader", flush=True)
    print("  4. 开始模型训练", flush=True)

    return full_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label and merge AI + human datasets.")
    parser.add_argument("--ai-file", default="datasets/final/parallel_dataset_cleaned.csv", help="AI dataset CSV")
    parser.add_argument("--human-file", default="datasets/human_texts/thucnews_real_human_9000.csv",
                        help="Human dataset CSV")
    parser.add_argument("--output-dir", default="datasets/bert", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    args = parser.parse_args()

    dataset = load_and_label_datasets(
        ai_file=args.ai_file,
        human_file=args.human_file,
        output_dir=args.output_dir,
        seed=args.seed
    )
