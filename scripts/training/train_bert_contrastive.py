"""
BERT + 对比学习的AI文本检测训练脚本

核心创新:
1. 监督对比学习 (Supervised Contrastive Learning)
2. 双任务学习: 分类损失 + 对比损失
3. 硬负样本挖掘 (Hard Negative Mining)
4. 场景感知训练 (可选)

理论依据:
- DeTeCtive (NeurIPS 2024): 多级对比学习检测AI文本
- Genre-Aware CL (PAN@CLEF 2025): 体裁感知对比学习
"""

import os
import sys
import json
import argparse
import random
from typing import Optional, List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertModel,
    BertTokenizer,
    BertPreTrainedModel,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


# ==================== 对比学习损失函数 ====================

class SupConLoss(nn.Module):
    """
    监督对比学习损失 (Supervised Contrastive Loss)

    论文: Supervised Contrastive Learning (NeurIPS 2020)

    与普通对比学习的区别:
    - 同类样本作为正样本 (不只是数据增强)
    - 不同类样本作为负样本

    对于AI检测任务:
    - Human文本相互拉近
    - AI文本相互拉近
    - Human与AI相互推远
    """
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels, mask=None):
        """
        Args:
            features: [batch_size, hidden_dim] 归一化后的特征向量
            labels: [batch_size] 标签 (0=human, 1=ai)
            mask: 可选的对比掩码

        Returns:
            对比损失值
        """
        device = features.device
        batch_size = features.shape[0]

        # 确保标签是正确形状
        labels = labels.contiguous().view(-1, 1)

        # 创建标签掩码: 同类为1，异类为0
        # [batch_size, batch_size]
        label_mask = torch.eq(labels, labels.T).float().to(device)

        # 计算相似度矩阵 [batch_size, batch_size]
        # 使用点积计算（特征已经归一化）
        similarity_matrix = torch.matmul(features, features.T)

        # 对角线掩码（排除自己）
        logits_mask = torch.scatter(
            torch.ones_like(label_mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )

        # 应用掩码（移除自己）
        mask = label_mask * logits_mask

        # 计算log概率
        # exp_logits: [batch_size, batch_size]
        exp_logits = torch.exp(similarity_matrix / self.temperature) * logits_mask

        # 分母：所有样本的exp(sim)之和（排除自己）
        log_prob = similarity_matrix / self.temperature - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # 只保留正样本对的log概率，然后求平均
        # 正样本对数量
        pos_count = mask.sum(1)
        pos_count = torch.clamp(pos_count, min=1)  # 避免除零

        # 计算每个anchor的对比损失
        mean_log_prob_pos = (mask * log_prob).sum(1) / pos_count

        # 最终损失（取负）
        loss = -mean_log_prob_pos.mean()

        return loss


class HardNegativeSupConLoss(nn.Module):
    """
    带硬负样本挖掘的对比损失

    策略:
    - 在每个batch中找到最相似的负样本
    - 加大这些样本的权重
    - 帮助模型学习更精细的区分特征
    """
    def __init__(self, temperature=0.07, hard_negative_weight=0.5):
        super().__init__()
        self.temperature = temperature
        self.hard_negative_weight = hard_negative_weight

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)

        # 标签掩码
        pos_mask = torch.eq(labels, labels.T).float().to(device)
        neg_mask = 1 - pos_mask

        # 对角线掩码
        diag_mask = torch.scatter(
            torch.ones(batch_size, batch_size).to(device),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )

        # 相似度矩阵
        sim_matrix = torch.matmul(features, features.T) / self.temperature

        # 硬负样本: 找到每个anchor最相似的负样本
        neg_sim = sim_matrix * neg_mask
        neg_sim[neg_mask == 0] = float('-inf')
        hardest_neg_sim = neg_sim.max(dim=1, keepdim=True)[0]

        # 对硬负样本加权
        hard_neg_weight_mask = (neg_sim == hardest_neg_sim).float() * self.hard_negative_weight
        neg_weight = neg_mask + hard_neg_weight_mask

        # 计算损失
        exp_sim = torch.exp(sim_matrix) * diag_mask

        # 分母: 正样本 + 加权负样本
        denominator = (exp_sim * pos_mask).sum(1, keepdim=True) + (exp_sim * neg_weight).sum(1, keepdim=True)

        # 分子: 正样本
        numerator = exp_sim * pos_mask * diag_mask

        # Log概率
        log_prob = torch.log(numerator / (denominator + 1e-8) + 1e-8)
        log_prob = log_prob * pos_mask * diag_mask

        # 每个anchor的平均
        pos_count = (pos_mask * diag_mask).sum(1)
        pos_count = torch.clamp(pos_count, min=1)

        loss = -(log_prob.sum(1) / pos_count).mean()

        return loss


# ==================== 模型定义 ====================

class BertForContrastiveClassification(BertPreTrainedModel):
    """
    BERT + 对比学习 + 分类的统一模型

    架构:
        BERT -> [CLS] embedding -> Projection Head -> 归一化特征 (用于对比)
                                -> Classifier Head -> 分类 logits

    训练:
        Loss = α * 分类损失 + (1-α) * 对比损失
    """
    def __init__(self, config, projection_dim=128, dropout=0.1):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=True)

        hidden_size = config.hidden_size

        # 投影头 (用于对比学习)
        # 将768维映射到128维，更容易计算对比
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, projection_dim)
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2)  # binary classification
        )

        # 初始化
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        return_features=False
    ):
        """
        前向传播

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            token_type_ids: [batch, seq_len]
            return_features: 是否返回对比特征

        Returns:
            logits: [batch, 2] 分类logits
            features: [batch, projection_dim] 归一化特征 (如果return_features=True)
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # [CLS] token的表示
        pooled_output = outputs.pooler_output  # [batch, hidden_size]

        # 分类
        logits = self.classifier(pooled_output)  # [batch, 2]

        if return_features:
            # 投影 + L2归一化
            projected = self.projection_head(pooled_output)  # [batch, projection_dim]
            features = F.normalize(projected, p=2, dim=1)  # L2归一化
            return logits, features

        return logits


# ==================== 数据集 ====================

class ContrastiveAIDetectionDataset(Dataset):
    """
    支持对比学习的AI检测数据集

    特性:
    - 支持场景标签(可选)
    - 动态padding
    - 文本对采样(用于对比)
    """
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: BertTokenizer,
        max_length: int = 512,
        scenarios: Optional[List[str]] = None
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.scenarios = scenarios

        # 按标签分组索引 (用于采样正负样本对)
        self.label_to_indices = {0: [], 1: []}
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,  # 动态padding在collate_fn中处理
            return_tensors=None
        )

        item = {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'token_type_ids': encoding.get('token_type_ids', [0] * len(encoding['input_ids'])),
            'label': label,
            'idx': idx
        }

        if self.scenarios:
            item['scenario'] = self.scenarios[idx]

        return item


def contrastive_collate_fn(batch):
    """
    对比学习专用的collate函数

    - 动态padding到batch内最大长度
    - 保留索引信息(用于采样)
    """
    # 找到batch内最大长度
    max_len = max(len(item['input_ids']) for item in batch)

    # Padding
    input_ids = []
    attention_masks = []
    token_type_ids = []
    labels = []
    indices = []

    for item in batch:
        seq_len = len(item['input_ids'])
        pad_len = max_len - seq_len

        input_ids.append(item['input_ids'] + [0] * pad_len)
        attention_masks.append(item['attention_mask'] + [0] * pad_len)
        token_type_ids.append(item['token_type_ids'] + [0] * pad_len)
        labels.append(item['label'])
        indices.append(item['idx'])

    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(attention_masks, dtype=torch.long),
        'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long),
        'indices': torch.tensor(indices, dtype=torch.long)
    }


# ==================== 训练器 ====================

class ContrastiveBERTTrainer:
    """
    对比学习BERT训练器

    参数:
        contrastive_weight: 对比损失权重 (推荐0.1-0.3)
        temperature: 对比学习温度 (推荐0.05-0.1)
        use_hard_negative: 是否使用硬负样本挖掘
    """
    def __init__(
        self,
        model_name='bert-base-chinese',
        max_length=512,
        batch_size=32,  # 对比学习需要较大batch
        learning_rate=2e-5,
        num_epochs=5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        contrastive_weight=0.2,  # 对比损失权重
        temperature=0.07,        # 对比温度
        projection_dim=128,      # 投影维度
        use_hard_negative=True,  # 硬负样本挖掘
        device=None,
        output_dir='models/bert_contrastive'
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        self.projection_dim = projection_dim
        self.use_hard_negative = use_hard_negative
        self.output_dir = output_dir

        # 设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        print(f"使用设备: {self.device}")

        # 创建目录
        os.makedirs(output_dir, exist_ok=True)

        # 初始化
        self._init_model()
        self._init_loss()

        # 历史记录
        self.history = {
            'train_loss': [],
            'train_cls_loss': [],
            'train_con_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': []
        }

    def _init_model(self):
        """初始化模型"""
        print("\n正在加载模型...")

        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)

        # 使用自定义的对比学习模型
        from transformers import BertConfig
        config = BertConfig.from_pretrained(self.model_name)
        self.model = BertForContrastiveClassification(
            config,
            projection_dim=self.projection_dim
        )

        # 加载预训练权重 (只加载BERT部分)
        pretrained_model = BertModel.from_pretrained(self.model_name)
        self.model.bert.load_state_dict(pretrained_model.state_dict())
        del pretrained_model

        self.model.to(self.device)
        print(f"✓ 模型加载成功: {self.model_name}")
        print(f"  - 投影维度: {self.projection_dim}")

    def _init_loss(self):
        """初始化损失函数"""
        # 分类损失
        self.cls_criterion = nn.CrossEntropyLoss()

        # 对比损失
        if self.use_hard_negative:
            self.con_criterion = HardNegativeSupConLoss(temperature=self.temperature)
            print(f"✓ 使用硬负样本对比损失 (temperature={self.temperature})")
        else:
            self.con_criterion = SupConLoss(temperature=self.temperature)
            print(f"✓ 使用监督对比损失 (temperature={self.temperature})")

        print(f"✓ 对比损失权重: {self.contrastive_weight}")

    def load_data(self, train_csv, val_csv, test_csv):
        """加载数据"""
        print("\n正在加载数据...")

        train_df = pd.read_csv(train_csv, encoding='utf-8-sig')
        val_df = pd.read_csv(val_csv, encoding='utf-8-sig')
        test_df = pd.read_csv(test_csv, encoding='utf-8-sig')

        print(f"  训练集: {len(train_df)} 条")
        print(f"  验证集: {len(val_df)} 条")
        print(f"  测试集: {len(test_df)} 条")

        # 检查是否有场景标签
        has_scenario = 'scenario' in train_df.columns

        # 创建数据集
        self.train_dataset = ContrastiveAIDetectionDataset(
            train_df['text'].tolist(),
            train_df['label'].tolist(),
            self.tokenizer,
            self.max_length,
            scenarios=train_df['scenario'].tolist() if has_scenario else None
        )

        self.val_dataset = ContrastiveAIDetectionDataset(
            val_df['text'].tolist(),
            val_df['label'].tolist(),
            self.tokenizer,
            self.max_length
        )

        self.test_dataset = ContrastiveAIDetectionDataset(
            test_df['text'].tolist(),
            test_df['label'].tolist(),
            self.tokenizer,
            self.max_length
        )

        # DataLoader
        # 注意: 对比学习需要较大的batch size来提供足够的正负样本
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=contrastive_collate_fn,
            drop_last=True  # 确保每个batch都是完整的
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=contrastive_collate_fn
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=contrastive_collate_fn
        )

        print(f"✓ 数据加载完成")
        print(f"✓ 每batch样本数: {self.batch_size} (对比学习推荐32+)")

    def _setup_optimizer(self):
        """设置优化器"""
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        total_steps = len(self.train_loader) * self.num_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        print(f"✓ 优化器配置完成")
        print(f"  - 总步数: {total_steps}")
        print(f"  - 预热步数: {warmup_steps}")

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()

        total_loss = 0
        total_cls_loss = 0
        total_con_loss = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        for batch in pbar:
            # 移动到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            token_type_ids = batch['token_type_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            # 前向传播 (获取logits和特征)
            logits, features = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_features=True
            )

            # 分类损失
            cls_loss = self.cls_criterion(logits, labels)

            # 对比损失
            con_loss = self.con_criterion(features, labels)

            # 总损失
            loss = (1 - self.contrastive_weight) * cls_loss + self.contrastive_weight * con_loss

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            # 统计
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_con_loss += con_loss.item()

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cls': f'{cls_loss.item():.4f}',
                'con': f'{con_loss.item():.4f}'
            })

        # 计算平均
        n_batches = len(self.train_loader)
        avg_loss = total_loss / n_batches
        avg_cls_loss = total_cls_loss / n_batches
        avg_con_loss = total_con_loss / n_batches
        accuracy = accuracy_score(all_labels, all_preds)

        return avg_loss, avg_cls_loss, avg_con_loss, accuracy

    def evaluate(self, data_loader, desc='评估中'):
        """评估"""
        self.model.eval()

        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc=desc):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                # 评估时不需要对比特征
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    return_features=False
                )

                loss = self.cls_criterion(logits, labels)
                total_loss += loss.item()

                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1).cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())

        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary'
        )

        return avg_loss, accuracy, precision, recall, f1, all_preds, all_labels

    def train(self):
        """完整训练"""
        print("\n" + "=" * 70)
        print("🚀 开始对比学习训练")
        print("=" * 70)

        self._setup_optimizer()

        best_val_f1 = 0

        for epoch in range(self.num_epochs):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            print('=' * 70)

            # 训练
            train_loss, cls_loss, con_loss, train_acc = self.train_epoch(epoch)

            # 验证
            val_loss, val_acc, val_prec, val_recall, val_f1, _, _ = self.evaluate(
                self.val_loader, '验证中'
            )

            # 记录
            self.history['train_loss'].append(train_loss)
            self.history['train_cls_loss'].append(cls_loss)
            self.history['train_con_loss'].append(con_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)

            # 打印
            print(f"\n📊 训练指标:")
            print(f"   总损失: {train_loss:.4f} | 分类: {cls_loss:.4f} | 对比: {con_loss:.4f}")
            print(f"   准确率: {train_acc:.4f}")
            print(f"\n📊 验证指标:")
            print(f"   损失: {val_loss:.4f} | 准确率: {val_acc:.4f}")
            print(f"   Precision: {val_prec:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f}")

            # 保存最佳
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                self.save_model('best_model')
                print(f"\n✅ 保存最佳模型 (F1={val_f1:.4f})")

        self.save_model('final_model')
        print("\n✅ 训练完成!")

    def test(self):
        """测试"""
        print("\n" + "=" * 70)
        print("📊 测试集评估")
        print("=" * 70)

        # 加载最佳模型
        self.load_model('best_model')

        test_loss, test_acc, test_prec, test_recall, test_f1, preds, labels = self.evaluate(
            self.test_loader, '测试中'
        )

        print(f"\n🎯 测试结果:")
        print(f"   准确率: {test_acc:.4f}")
        print(f"   Precision: {test_prec:.4f}")
        print(f"   Recall: {test_recall:.4f}")
        print(f"   F1 Score: {test_f1:.4f}")

        print(f"\n📋 详细分类报告:")
        print(classification_report(
            labels, preds,
            target_names=['人类文本', 'AI文本'],
            digits=4
        ))

        # 保存结果
        results = {
            'test_accuracy': float(test_acc),
            'test_precision': float(test_prec),
            'test_recall': float(test_recall),
            'test_f1': float(test_f1),
            'contrastive_weight': self.contrastive_weight,
            'temperature': self.temperature,
            'use_hard_negative': self.use_hard_negative
        }

        with open(os.path.join(self.output_dir, 'test_results.json'), 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return results

    def save_model(self, name='model'):
        """保存模型"""
        save_path = os.path.join(self.output_dir, name)
        os.makedirs(save_path, exist_ok=True)

        # 保存模型
        torch.save(self.model.state_dict(), os.path.join(save_path, 'model.pt'))
        self.tokenizer.save_pretrained(save_path)

        # 保存配置
        config = {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'projection_dim': self.projection_dim,
            'contrastive_weight': self.contrastive_weight,
            'temperature': self.temperature,
            'use_hard_negative': self.use_hard_negative,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs
        }

        with open(os.path.join(save_path, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        # 保存历史
        with open(os.path.join(save_path, 'history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)

    def load_model(self, name='model'):
        """加载模型"""
        load_path = os.path.join(self.output_dir, name)

        # 加载配置
        with open(os.path.join(load_path, 'config.json'), 'r') as f:
            config = json.load(f)

        # 加载模型
        self.model.load_state_dict(
            torch.load(os.path.join(load_path, 'model.pt'), map_location=self.device)
        )
        self.tokenizer = BertTokenizer.from_pretrained(load_path)

        print(f"✓ 模型已加载: {load_path}")


def main():
    parser = argparse.ArgumentParser(description="BERT + Contrastive Learning Training")

    # 数据
    parser.add_argument("--train-csv", default="datasets/active/core_v1/train.csv")
    parser.add_argument("--val-csv", default="datasets/active/core_v1/val.csv")
    parser.add_argument("--test-csv", default="datasets/active/core_v1/test.csv")
    parser.add_argument("--output-dir", default="models/bert_contrastive")

    # 模型
    parser.add_argument("--model-name", default="bert-base-chinese")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--projection-dim", type=int, default=128)

    # 训练
    parser.add_argument("--batch-size", type=int, default=32,
                        help="对比学习推荐32+以提供足够正负样本")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)

    # 对比学习
    parser.add_argument("--contrastive-weight", type=float, default=0.2,
                        help="对比损失权重 (推荐0.1-0.3)")
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="对比学习温度 (推荐0.05-0.1)")
    parser.add_argument("--no-hard-negative", action="store_true",
                        help="禁用硬负样本挖掘")

    args = parser.parse_args()

    print("=" * 70)
    print("🔥 BERT + 对比学习 AI文本检测训练")
    print("=" * 70)
    print("\n核心创新:")
    print("  ✓ 监督对比学习 - 学习更鲁棒的特征表示")
    print("  ✓ 硬负样本挖掘 - 关注难分样本")
    print("  ✓ 双任务学习 - 分类+对比联合优化")
    print("\n配置:")
    print(f"  对比权重: {args.contrastive_weight}")
    print(f"  温度参数: {args.temperature}")
    print(f"  Batch大小: {args.batch_size}")

    # 创建训练器
    trainer = ContrastiveBERTTrainer(
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        contrastive_weight=args.contrastive_weight,
        temperature=args.temperature,
        projection_dim=args.projection_dim,
        use_hard_negative=not args.no_hard_negative,
        output_dir=args.output_dir
    )

    # 加载数据
    trainer.load_data(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv
    )

    # 训练
    trainer.train()

    # 测试
    results = trainer.test()

    print("\n" + "=" * 70)
    print("✅ 训练完成!")
    print("=" * 70)
    print(f"\n模型保存位置: {args.output_dir}")
    print(f"最终F1: {results['test_f1']:.4f}")

    print("\n💡 后续实验建议:")
    print("  1. 调整对比权重: --contrastive-weight 0.1/0.2/0.3")
    print("  2. 调整温度: --temperature 0.05/0.07/0.1")
    print("  3. 对比baseline: python scripts/training/train_bert_improved.py")


if __name__ == '__main__':
    main()
