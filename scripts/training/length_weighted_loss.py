"""
长度加权损失函数
给短文本更高的学习权重，防止模型过度依赖长度特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LengthWeightedLoss(nn.Module):
    """
    长度加权交叉熵损失

    原理：短文本获得更高的权重，强迫模型学习语义特征而非长度特征

    Args:
        alpha: 长度权重强度，范围[0, 1]
            - 0: 无权重（等同于普通交叉熵）
            - 1: 完全基于长度权重
            - 推荐：0.3-0.5
        max_length: 最大长度（用于归一化）
        reduction: 'mean', 'sum', 或 'none'
    """

    def __init__(self, alpha=0.3, max_length=512, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.max_length = max_length
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, labels, lengths=None, attention_masks=None):
        """
        计算长度加权损失

        Args:
            logits: 模型输出 [batch_size, num_classes]
            labels: 真实标签 [batch_size]
            lengths: 实际长度（可选） [batch_size]
                如果提供，直接使用
                如果不提供，从attention_mask计算
            attention_masks: 注意力掩码（可选） [batch_size, seq_len]

        Returns:
            加权后的损失（标量或向量，取决于reduction）
        """
        # 计算基础损失
        base_loss = self.ce_loss(logits, labels)  # [batch_size]

        # 计算实际长度
        if lengths is None:
            if attention_masks is not None:
                lengths = attention_masks.sum(dim=1).float()  # [batch_size]
            else:
                # 如果都没提供，返回无权重损失
                if self.reduction == 'mean':
                    return base_loss.mean()
                elif self.reduction == 'sum':
                    return base_loss.sum()
                else:
                    return base_loss

        # 归一化长度到[0, 1]
        norm_lengths = lengths.float() / self.max_length
        norm_lengths = torch.clamp(norm_lengths, 0.0, 1.0)  # 确保在[0,1]范围内

        # 计算权重：短文本权重更高
        # weight = 1.0 + alpha * (1.0 - norm_length)
        # 例如：
        #   - 长度 = max_length(512): weight = 1.0 + 0.3 * 0.0 = 1.0
        #   - 长度 = max_length/2(256): weight = 1.0 + 0.3 * 0.5 = 1.15
        #   - 长度 = 0: weight = 1.0 + 0.3 * 1.0 = 1.3
        weights = 1.0 + self.alpha * (1.0 - norm_lengths)

        # 应用权重
        weighted_loss = base_loss * weights

        # 应用reduction
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss


class AdaptiveLengthWeightedLoss(nn.Module):
    """
    自适应长度加权损失

    根据预测难度自适应调整权重：
    - 短且难预测的样本：最高权重
    - 短但易预测的样本：中等权重
    - 长且难预测的样本：中等权重
    - 长且易预测的样本：最低权重
    """

    def __init__(self, alpha_length=0.3, alpha_conf=0.2, max_length=512, reduction='mean'):
        """
        Args:
            alpha_length: 长度权重强度
            alpha_conf: 置信度权重强度
            max_length: 最大长度
            reduction: 损失聚合方式
        """
        super().__init__()
        self.alpha_length = alpha_length
        self.alpha_conf = alpha_conf
        self.max_length = max_length
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, labels, lengths=None, attention_masks=None):
        """
        计算自适应长度加权损失

        Args:
            logits: 模型输出 [batch_size, num_classes]
            labels: 真实标签 [batch_size]
            lengths: 实际长度（可选）
            attention_masks: 注意力掩码（可选）

        Returns:
            加权损失
        """
        # 计算基础损失
        base_loss = self.ce_loss(logits, labels)  # [batch_size]

        # 计算置信度（正确类别的概率）
        probs = F.softmax(logits, dim=1)  # [batch_size, num_classes]
        confidence = probs[range(len(labels)), labels]  # [batch_size]

        # 长度权重（短文本权重高）
        if lengths is None and attention_masks is not None:
            lengths = attention_masks.sum(dim=1).float()

        if lengths is not None:
            norm_lengths = torch.clamp(lengths.float() / self.max_length, 0.0, 1.0)
            length_weights = 1.0 + self.alpha_length * (1.0 - norm_lengths)
        else:
            length_weights = torch.ones_like(base_loss)

        # 置信度权重（低置信度/难样本权重高）
        conf_weights = 1.0 + self.alpha_conf * (1.0 - confidence)

        # 综合权重
        total_weights = length_weights * conf_weights

        # 应用权重
        weighted_loss = base_loss * total_weights

        # 应用reduction
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss


class FocalLengthWeightedLoss(nn.Module):
    """
    结合Focal Loss和长度权重

    Focal Loss关注难分样本，长度权重关注短文本
    两者结合可以更好地平衡学习
    """

    def __init__(self, alpha_length=0.3, gamma=2.0, max_length=512, reduction='mean'):
        """
        Args:
            alpha_length: 长度权重强度
            gamma: Focal Loss的focusing参数
            max_length: 最大长度
            reduction: 损失聚合方式
        """
        super().__init__()
        self.alpha_length = alpha_length
        self.gamma = gamma
        self.max_length = max_length
        self.reduction = reduction

    def forward(self, logits, labels, lengths=None, attention_masks=None):
        """
        计算Focal + 长度加权损失

        Args:
            logits: 模型输出 [batch_size, num_classes]
            labels: 真实标签 [batch_size]
            lengths: 实际长度（可选）
            attention_masks: 注意力掩码（可选）

        Returns:
            加权损失
        """
        # 计算概率
        probs = F.softmax(logits, dim=1)  # [batch_size, num_classes]
        probs_t = probs[range(len(labels)), labels]  # 正确类别的概率 [batch_size]

        # Focal Loss: FL(p_t) = -(1 - p_t)^gamma * log(p_t)
        focal_weight = (1.0 - probs_t) ** self.gamma
        base_loss = -torch.log(probs_t + 1e-8)
        focal_loss = focal_weight * base_loss

        # 长度权重
        if lengths is None and attention_masks is not None:
            lengths = attention_masks.sum(dim=1).float()

        if lengths is not None:
            norm_lengths = torch.clamp(lengths.float() / self.max_length, 0.0, 1.0)
            length_weights = 1.0 + self.alpha_length * (1.0 - norm_lengths)
        else:
            length_weights = torch.ones_like(focal_loss)

        # 综合损失
        weighted_loss = focal_loss * length_weights

        # 应用reduction
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss


# 使用示例
def example_usage():
    """演示如何使用长度加权损失"""

    # 创建损失函数
    criterion = LengthWeightedLoss(alpha=0.3, max_length=512, reduction='mean')

    # 模拟数据
    batch_size = 16
    num_classes = 2

    # 模型输出
    logits = torch.randn(batch_size, num_classes)  # [16, 2]

    # 真实标签
    labels = torch.randint(0, num_classes, (batch_size,))  # [16]

    # 实际长度（从attention_mask计算）
    attention_mask = torch.ones(batch_size, 512)
    # 模拟不同长度
    for i in range(batch_size):
        length = torch.randint(100, 512, (1,)).item()
        attention_mask[i, length:] = 0

    # 计算损失
    loss = criterion(logits, labels, attention_masks=attention_mask)

    print(f"Loss: {loss.item():.4f}")

    # 也可以直接传递长度
    lengths = attention_mask.sum(dim=1)  # [16]
    loss2 = criterion(logits, labels, lengths=lengths)

    print(f"Loss (with lengths): {loss2.item():.4f}")

    # 使用自适应版本
    adaptive_criterion = AdaptiveLengthWeightedLoss(
        alpha_length=0.3,
        alpha_conf=0.2,
        max_length=512
    )

    loss_adaptive = adaptive_criterion(logits, labels, lengths=lengths)
    print(f"Adaptive Loss: {loss_adaptive.item():.4f}")

    # 使用Focal版本
    focal_criterion = FocalLengthWeightedLoss(
        alpha_length=0.3,
        gamma=2.0,
        max_length=512
    )

    loss_focal = focal_criterion(logits, labels, lengths=lengths)
    print(f"Focal Loss: {loss_focal.item():.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("长度加权损失函数 - 使用示例")
    print("=" * 60)

    example_usage()

    print("\n" + "=" * 60)
    print("在训练脚本中使用：")
    print("=" * 60)
    print("""
# 导入
from scripts.training.length_weighted_loss import LengthWeightedLoss

# 创建损失函数
criterion = LengthWeightedLoss(
    alpha=0.3,  # 长度权重强度
    max_length=512,
    reduction='mean'
)

# 在训练循环中
for batch in train_loader:
    # 前向传播
    outputs = model(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        token_type_ids=batch['token_type_ids']
    )

    # 计算损失
    loss = criterion(
        logits=outputs.logits,
        labels=batch['labels'],
        attention_masks=batch['attention_mask']  # 用于计算实际长度
    )

    # 反向传播
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    """)

    print("\n推荐配置：")
    print("  - alpha=0.3: 适度的长度权重（推荐）")
    print("  - alpha=0.5: 较强的长度权重")
    print("  - alpha=0.1: 轻微的长度权重")
    print("\n优势：")
    print("  ✓ 强迫模型学习短文本的语义特征")
    print("  ✓ 防止过度依赖文本长度")
    print("  ✓ 提高对不同长度文本的泛化能力")
