#!/usr/bin/env python3
"""
图卷积网络（GCN）模块
用于提取文本图的结构特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch

class TextGCN(nn.Module):
    """文本图卷积网络"""
    
    def __init__(self, 
                 node_feature_dim=768,  # BERT embedding维度
                 hidden_dim=128,
                 output_dim=64,
                 num_layers=2,
                 dropout=0.3):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GCN层
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_feature_dim, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # 输出层
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, edge_index, batch):
        """
        Args:
            x: [num_nodes, node_feature_dim] 节点特征
            edge_index: [2, num_edges] 边索引
            batch: [num_nodes] 批次索引
        Returns:
            graph_embedding: [batch_size, output_dim]
        """
        # GCN层
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 图级别池化
        x = global_mean_pool(x, batch)
        
        # 输出
        x = self.fc(x)
        
        return x


class GraphFeatureExtractor:
    """图特征提取器（结合BERT和GCN）"""
    
    def __init__(self, bert_model, gcn_model, device='cuda'):
        self.bert_model = bert_model
        self.gcn_model = gcn_model
        self.device = device
        
    def extract_graph_features(self, text: str, graph_data: Data) -> torch.Tensor:
        """
        提取文本的图特征
        
        Args:
            text: 原始文本
            graph_data: 图数据（包含节点、边信息）
        Returns:
            graph_embedding: [output_dim]
        """
        # 1. 使用BERT为每个实体节点生成embedding
        node_embeddings = self._get_node_embeddings(text, graph_data)
        
        # 2. 使用GCN提取图结构特征
        graph_data.x = node_embeddings
        graph_data = graph_data.to(self.device)
        
        with torch.no_grad():
            graph_embedding = self.gcn_model(
                graph_data.x, 
                graph_data.edge_index,
                torch.zeros(graph_data.x.size(0), dtype=torch.long, device=self.device)
            )
        
        return graph_embedding.squeeze(0)
    
    def _get_node_embeddings(self, text: str, graph_data: Data) -> torch.Tensor:
        """为图中的每个节点生成BERT embedding"""
        # 简化版：使用整个文本的BERT embedding作为所有节点的初始特征
        # 实际应用中可以为每个实体单独编码
        
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        
        inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            text_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # 为每个节点复制相同的embedding（简化版）
        num_nodes = graph_data.num_nodes
        node_embeddings = text_embedding.repeat(num_nodes, 1)
        
        return node_embeddings


def test_gcn():
    """测试GCN模块"""
    print("测试图卷积网络...")
    
    # 创建模型
    gcn = TextGCN(
        node_feature_dim=768,
        hidden_dim=128,
        output_dim=64
    )
    
    # 模拟图数据
    num_nodes = 10
    num_edges = 15
    
    x = torch.randn(num_nodes, 768)  # 节点特征
    edge_index = torch.randint(0, num_nodes, (2, num_edges))  # 边
    batch = torch.zeros(num_nodes, dtype=torch.long)  # 单个图
    
    # 前向传播
    output = gcn(x, edge_index, batch)
    
    print(f"✓ GCN测试成功")
    print(f"  输入: {num_nodes}个节点, {num_edges}条边")
    print(f"  输出: {output.shape}")
    print(f"  参数量: {sum(p.numel() for p in gcn.parameters()):,}")


if __name__ == "__main__":
    test_gcn()
