#!/usr/bin/env python3
"""
完整的图增强检测模型
BERT + 统计特征 + 图神经网络
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch_geometric.data import Data
import sys
sys.path.append('scripts/features')

class GraphEnhancedDetectionModel(nn.Module):
    """图增强的AI文本检测模型"""
    
    def __init__(self,
                 bert_model_name='bert-base-chinese',
                 stat_feature_dim=10,
                 graph_feature_dim=6,  # 图统计特征
                 gcn_output_dim=64,
                 hidden_dim=256,
                 dropout=0.3):
        super().__init__()
        
        # 1. BERT编码器
        self.bert = BertModel.from_pretrained(bert_model_name)
        bert_dim = self.bert.config.hidden_size  # 768
        
        # 2. 统计特征处理
        self.stat_fc = nn.Sequential(
            nn.Linear(stat_feature_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 3. 图统计特征处理
        self.graph_stat_fc = nn.Sequential(
            nn.Linear(graph_feature_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 4. GCN图结构特征（可选，需要torch_geometric）
        self.use_gcn = False  # 默认关闭，避免依赖问题
        if self.use_gcn:
            from scripts.features.graph_neural_network import TextGCN
            self.gcn = TextGCN(
                node_feature_dim=bert_dim,
                hidden_dim=128,
                output_dim=gcn_output_dim
            )
            gcn_dim = gcn_output_dim
        else:
            gcn_dim = 0
        
        # 5. 特征融合分类器
        fusion_dim = bert_dim + 32 + 32 + gcn_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )
    
    def forward(self, input_ids, attention_mask, stat_features, graph_stat_features, graph_data=None):
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            stat_features: [batch, stat_dim]
            graph_stat_features: [batch, graph_stat_dim]
            graph_data: PyG Data对象（可选）
        """
        batch_size = input_ids.size(0)
        
        # 1. BERT语义特征
        bert_output = self.bert(input_ids, attention_mask)
        semantic_features = bert_output.pooler_output  # [batch, 768]
        
        # 2. 统计特征
        stat_embed = self.stat_fc(stat_features)  # [batch, 32]
        
        # 3. 图统计特征
        graph_stat_embed = self.graph_stat_fc(graph_stat_features)  # [batch, 32]
        
        # 4. GCN图结构特征（可选）
        if self.use_gcn and graph_data is not None:
            gcn_features = self.gcn(
                graph_data.x,
                graph_data.edge_index,
                graph_data.batch
            )  # [batch, gcn_dim]
            fused = torch.cat([semantic_features, stat_embed, graph_stat_embed, gcn_features], dim=1)
        else:
            fused = torch.cat([semantic_features, stat_embed, graph_stat_embed], dim=1)
        
        # 5. 分类
        logits = self.classifier(fused)
        
        return logits


class GraphEnhancedTrainer:
    """图增强模型训练器"""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        
        # 图构建器
        from scripts.features.text_graph_builder import TextGraphBuilder
        self.graph_builder = TextGraphBuilder()
    
    def prepare_batch(self, texts, labels):
        """准备训练批次（包含图特征）"""
        from scripts.features.statistical_features import StatisticalFeatureExtractor
        
        stat_extractor = StatisticalFeatureExtractor()
        
        batch_data = {
            'texts': texts,
            'labels': labels,
            'stat_features': [],
            'graph_stat_features': []
        }
        
        # 提取特征
        for text in texts:
            # 统计特征
            stat_feat = stat_extractor.extract_features(text)
            batch_data['stat_features'].append(stat_feat)
            
            # 图特征
            graph = self.graph_builder.build_graph(text)
            graph_feat = self.graph_builder.get_graph_features(graph)
            graph_feat_array = [
                graph_feat['num_nodes'],
                graph_feat['num_edges'],
                graph_feat['density'],
                graph_feat['avg_degree'],
                graph_feat['clustering'],
                graph_feat['avg_path_length']
            ]
            batch_data['graph_stat_features'].append(graph_feat_array)
        
        return batch_data
    
    def train_epoch(self, train_data, optimizer, tokenizer):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        criterion = nn.CrossEntropyLoss()
        
        for batch in train_data:
            # 准备数据
            texts = batch['texts']
            labels = torch.tensor(batch['labels']).to(self.device)
            
            # Tokenize
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            
            # 特征
            stat_features = torch.tensor(batch['stat_features']).to(self.device)
            graph_stat_features = torch.tensor(batch['graph_stat_features']).to(self.device)
            
            # 前向传播
            optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask, stat_features, graph_stat_features)
            loss = criterion(logits, labels)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_data)


def test_graph_enhanced_model():
    """测试图增强模型"""
    print("测试图增强检测模型...")
    
    model = GraphEnhancedDetectionModel()
    
    # 模拟输入
    batch_size = 4
    input_ids = torch.randint(0, 21128, (batch_size, 128))
    attention_mask = torch.ones(batch_size, 128)
    stat_features = torch.randn(batch_size, 10)
    graph_stat_features = torch.randn(batch_size, 6)
    
    # 前向传播
    logits = model(input_ids, attention_mask, stat_features, graph_stat_features)
    
    print(f"✓ 图增强模型测试成功")
    print(f"  输入维度:")
    print(f"    - BERT: {input_ids.shape}")
    print(f"    - 统计特征: {stat_features.shape}")
    print(f"    - 图特征: {graph_stat_features.shape}")
    print(f"  输出: {logits.shape}")
    print(f"  总参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 特征融合维度
    print(f"\n特征融合:")
    print(f"  BERT: 768维")
    print(f"  统计特征: 10维 -> 32维")
    print(f"  图特征: 6维 -> 32维")
    print(f"  融合后: 832维 -> 256维 -> 128维 -> 2维")


if __name__ == "__main__":
    test_graph_enhanced_model()
