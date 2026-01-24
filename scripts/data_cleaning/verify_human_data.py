#!/usr/bin/env python3
"""
验证并清理数据集 - 只保留真实人类文本
移除任何AI生成的"人类风格"文本
"""

import pandas as pd
from pathlib import Path

def verify_human_data():
    """验证人类数据的真实性"""
    
    print("="*60)
    print("数据集真实性验证")
    print("="*60)
    
    # 1. 检查thucnews（真实人类文本）
    print("\n[1] THUCNews真实人类文本:")
    thucnews_file = "datasets/human_texts/thucnews_real_human_9000.csv"
    if Path(thucnews_file).exists():
        df_thucnews = pd.read_csv(thucnews_file, encoding='utf-8-sig')
        print(f"  ✓ 文件存在: {len(df_thucnews)} 条")
        print(f"  平均长度: {df_thucnews['text'].str.len().mean():.0f}")
        print(f"  来源标记: {df_thucnews['source'].unique()}")
    else:
        print(f"  ✗ 文件不存在")
        df_thucnews = None
    
    # 2. 检查human_style（可疑 - 可能是AI生成）
    print("\n[2] Human Style文本（可疑）:")
    human_style_file = "datasets/human_texts/human_style_texts_9000.csv"
    if Path(human_style_file).exists():
        df_style = pd.read_csv(human_style_file, encoding='utf-8-sig')
        print(f"  ⚠️  文件存在: {len(df_style)} 条")
        print(f"  平均长度: {df_style['text'].str.len().mean():.0f}")
        print(f"  来源标记: {df_style['source'].unique()}")
        print(f"  ⚠️  标记为 'generated_human_style' - 这是AI生成的！")
        print(f"  ❌ 不应该用作人类数据")
    else:
        print(f"  文件不存在")
    
    # 3. 检查当前使用的数据集
    print("\n[3] 当前训练数据集:")
    current_file = "datasets/bert_v2/full_dataset_labeled.csv"
    if Path(current_file).exists():
        df_current = pd.read_csv(current_file, encoding='utf-8-sig')
        human_data = df_current[df_current['label'] == 0]
        print(f"  人类数据: {len(human_data)} 条")
        print(f"  平均长度: {human_data['text'].str.len().mean():.0f}")
        
        # 检查是否包含可疑数据
        if 'source' in df_current.columns:
            sources = df_current[df_current['label']==0]['source'].unique()
            print(f"  来源: {sources}")
            
            if 'generated_human_style' in sources:
                print(f"  ❌ 警告：包含AI生成的'人类风格'文本！")
            elif 'human_written' in sources:
                print(f"  ⚠️  标记为'human_written'，需要确认来源")
            elif 'thucnews_real' in sources:
                print(f"  ✓ 来自THUCNews，真实人类文本")
    
    # 4. 建议
    print("\n" + "="*60)
    print("建议:")
    print("="*60)
    
    if df_thucnews is not None:
        print(f"✓ 使用 THUCNews 作为人类数据（{len(df_thucnews)} 条真实文本）")
        print(f"✓ 平均长度: {df_thucnews['text'].str.len().mean():.0f}")
        print(f"✓ 这是真实的人类撰写的新闻文本")
    else:
        print("✗ 需要收集真实的人类文本数据集")
        print("  推荐: HC3, THUCNews, Wikipedia等")
    
    print("\n❌ 不要使用:")
    print("  - human_style_texts_9000.csv (AI生成的)")
    print("  - 任何标记为'generated'的数据")
    
    return df_thucnews


if __name__ == "__main__":
    verify_human_data()
