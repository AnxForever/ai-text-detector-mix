#!/usr/bin/env python3
"""
合并 core_v3 数据集（精简版）

只添加真实Human数据，废弃模板生成数据：
1. core_v2 原有数据 (64,614)
2. + M4-qazh 真实Human (3,000)
3. + VCSum 真实会议摘要 (239)
"""
import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def main():
    print("=" * 60)
    print("合并 core_v3 数据集（精简版 - 仅真实数据）")
    print("=" * 60)

    # 1. 加载 core_v2
    print("\n1. 加载 core_v2...")
    core_v2_dir = Path('datasets/active/core_v2')
    train_v2 = pd.read_csv(core_v2_dir / 'train.csv')
    val_v2 = pd.read_csv(core_v2_dir / 'val.csv')
    full_v2 = pd.concat([train_v2, val_v2], ignore_index=True)
    print(f"   total: {len(full_v2)}")

    # 2. 从外部数据提取真实Human（仅M4-qazh + VCSum）
    print("\n2. 提取真实外部Human数据...")
    ext_path = Path('datasets/external/extracted_all_for_training.json')
    with open(ext_path, 'r', encoding='utf-8') as f:
        ext_data = json.load(f)

    real_human = []
    for item in ext_data['human']:
        source = item.get('source', '')
        # 只取M4-qazh和VCSum（真实数据），排除HC3和LCSTS（已在core_v2中）
        if source in ['M4-qazh', 'VCSum']:
            real_human.append({
                'text': item['text'],
                'label': 0,
                'source': f"external_{source.lower().replace('-', '_')}",
                'category': 'Human',
                'length': len(item['text']),
                'origin_dataset': 'external_supplement',
                'origin_split': 'supplement'
            })

    real_human_df = pd.DataFrame(real_human)
    print(f"   M4-qazh + VCSum: {len(real_human_df)} 条")
    for src, cnt in real_human_df['source'].value_counts().items():
        print(f"     {src}: {cnt}")

    # 3. 合并
    print("\n3. 合并...")
    cols = ['text', 'label', 'source', 'length', 'category', 'origin_dataset', 'origin_split']
    for col in cols:
        if col not in full_v2.columns:
            full_v2[col] = ''
        if col not in real_human_df.columns:
            real_human_df[col] = ''

    full_v3 = pd.concat([full_v2[cols], real_human_df[cols]], ignore_index=True)
    print(f"   合并后: {len(full_v3)}")

    # 4. 去重
    print("\n4. 去重...")
    before = len(full_v3)
    full_v3 = full_v3.drop_duplicates(subset=['text'], keep='first')
    print(f"   {before} -> {len(full_v3)} (去除 {before - len(full_v3)})")

    # 5. 统计
    print("\n" + "=" * 60)
    print("数据统计:")
    print("=" * 60)

    print(f"\n标签分布:")
    for label, count in full_v3['label'].value_counts().items():
        name = 'Human' if label == 0 else 'AI'
        print(f"  {name}: {count}")

    print(f"\nHuman来源分布:")
    human_data = full_v3[full_v3['label'] == 0]
    for src, cnt in human_data['source'].value_counts().items():
        pct = cnt / len(human_data) * 100
        print(f"  {src}: {cnt} ({pct:.1f}%)")

    # 6. 分割
    print("\n6. 分割 train/val...")
    train_df, val_df = train_test_split(
        full_v3, test_size=0.1, random_state=42, stratify=full_v3['label']
    )
    print(f"   train: {len(train_df)}")
    print(f"   val: {len(val_df)}")

    # 7. 保存
    output_dir = Path('datasets/active/core_v3')
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_dir / 'train.csv', index=False, encoding='utf-8-sig')
    val_df.to_csv(output_dir / 'val.csv', index=False, encoding='utf-8-sig')

    # 保存日志
    merge_log = {
        'version': 'core_v3',
        'date': '2026-01-30',
        'strategy': '仅添加真实数据，废弃模板生成数据',
        'sources': {
            'core_v2': len(full_v2),
            'M4_qazh': int((real_human_df['source'] == 'external_m4_qazh').sum()),
            'VCSum': int((real_human_df['source'] == 'external_vcsum').sum()),
        },
        'total': len(full_v3),
        'train_size': len(train_df),
        'val_size': len(val_df),
        'label_distribution': {
            'Human': int((full_v3['label'] == 0).sum()),
            'AI': int((full_v3['label'] == 1).sum()),
        }
    }
    with open(output_dir / 'merge_log.json', 'w', encoding='utf-8') as f:
        json.dump(merge_log, f, ensure_ascii=False, indent=2)

    print(f"\n保存到: {output_dir}")
    print(f"core_v3 总计: {len(full_v3)} 条")


if __name__ == '__main__':
    main()
