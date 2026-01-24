#!/usr/bin/env python3
"""
批量提取图特征
为整个数据集添加图特征列
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from text_graph_builder import TextGraphBuilder

def extract_graph_features_batch(input_file, output_file):
    """批量提取图特征"""
    print(f"读取数据: {input_file}")
    df = pd.read_csv(input_file, encoding='utf-8-sig')
    
    print(f"数据量: {len(df)} 条")
    
    # 初始化图构建器
    builder = TextGraphBuilder()
    
    # 提取特征
    graph_features = []
    
    print("提取图特征...")
    for text in tqdm(df['text'], desc="Processing"):
        try:
            graph = builder.build_graph(text)
            features = builder.get_graph_features(graph)
            graph_features.append([
                features['num_nodes'],
                features['num_edges'],
                features['density'],
                features['avg_degree'],
                features['clustering'],
                features['avg_path_length']
            ])
        except Exception as e:
            # 出错时使用零向量
            graph_features.append([0, 0, 0, 0, 0, 0])
    
    # 添加到DataFrame
    graph_feature_names = [
        'graph_num_nodes',
        'graph_num_edges', 
        'graph_density',
        'graph_avg_degree',
        'graph_clustering',
        'graph_avg_path_length'
    ]
    
    graph_df = pd.DataFrame(graph_features, columns=graph_feature_names)
    result_df = pd.concat([df, graph_df], axis=1)
    
    # 保存
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✓ 已保存到: {output_file}")
    
    # 统计
    print(f"\n图特征统计:")
    print(graph_df.describe())
    
    # 按标签分组统计
    if 'label' in result_df.columns:
        print(f"\n按标签分组:")
        for label in result_df['label'].unique():
            subset = result_df[result_df['label'] == label]
            label_name = "AI" if label == 1 else "Human"
            print(f"\n{label_name} (label={label}):")
            print(subset[graph_feature_names].describe().loc[['mean', 'std']])


def main():
    parser = argparse.ArgumentParser(description='批量提取图特征')
    parser.add_argument('--input', required=True, help='输入CSV文件')
    parser.add_argument('--output', required=True, help='输出CSV文件')
    
    args = parser.parse_args()
    
    extract_graph_features_batch(args.input, args.output)


if __name__ == "__main__":
    main()
