#!/usr/bin/env python3
"""
文本实体关系图构建
提取文本中的实体和关系，构建图结构
"""

import jieba.posseg as pseg
import networkx as nx
from typing import List, Dict, Tuple
import re

class TextGraphBuilder:
    """文本图构建器"""
    
    def __init__(self):
        # 实体词性标记
        self.entity_pos = {'n', 'nr', 'ns', 'nt', 'nz', 'v', 'vn'}
        
    def build_graph(self, text: str) -> nx.Graph:
        """构建文本的实体关系图"""
        # 1. 实体识别
        entities = self._extract_entities(text)
        
        # 2. 构建图
        G = nx.Graph()
        
        # 添加节点
        for i, entity in enumerate(entities):
            G.add_node(i, word=entity['word'], pos=entity['pos'])
        
        # 3. 添加边（基于共现和距离）
        for i in range(len(entities)):
            for j in range(i+1, len(entities)):
                # 距离权重：越近权重越大
                distance = abs(entities[i]['position'] - entities[j]['position'])
                if distance < 50:  # 50字符内认为有关联
                    weight = 1.0 / (1 + distance / 10)
                    G.add_edge(i, j, weight=weight)
        
        return G
    
    def _extract_entities(self, text: str) -> List[Dict]:
        """提取实体"""
        entities = []
        position = 0
        
        words = pseg.cut(text)
        for word, pos in words:
            # 只保留实体类词汇
            if pos in self.entity_pos and len(word) > 1:
                entities.append({
                    'word': word,
                    'pos': pos,
                    'position': position
                })
            position += len(word)
        
        return entities
    
    def get_graph_features(self, G: nx.Graph) -> Dict[str, float]:
        """提取图的统计特征"""
        if len(G.nodes) == 0:
            return {
                'num_nodes': 0,
                'num_edges': 0,
                'density': 0,
                'avg_degree': 0,
                'clustering': 0,
                'avg_path_length': 0
            }
        
        features = {
            'num_nodes': len(G.nodes),
            'num_edges': len(G.edges),
            'density': nx.density(G),
            'avg_degree': sum(dict(G.degree()).values()) / len(G.nodes) if len(G.nodes) > 0 else 0,
        }
        
        # 聚类系数
        try:
            features['clustering'] = nx.average_clustering(G)
        except:
            features['clustering'] = 0
        
        # 平均路径长度
        try:
            if nx.is_connected(G):
                features['avg_path_length'] = nx.average_shortest_path_length(G)
            else:
                # 对于非连通图，计算最大连通分量的平均路径长度
                largest_cc = max(nx.connected_components(G), key=len)
                subgraph = G.subgraph(largest_cc)
                features['avg_path_length'] = nx.average_shortest_path_length(subgraph)
        except:
            features['avg_path_length'] = 0
        
        return features


def main():
    """测试图构建"""
    builder = TextGraphBuilder()
    
    test_texts = [
        "人工智能技术在医疗领域的应用越来越广泛。深度学习算法可以帮助医生诊断疾病。机器学习模型能够分析患者的病历数据。",
        "今天天气很好，我去公园散步。看到很多人在锻炼身体。"
    ]
    
    for i, text in enumerate(test_texts):
        print(f"\n文本 {i+1}: {text[:50]}...")
        
        G = builder.build_graph(text)
        features = builder.get_graph_features(G)
        
        print(f"图特征:")
        for key, val in features.items():
            print(f"  {key}: {val:.4f}")


if __name__ == "__main__":
    main()
