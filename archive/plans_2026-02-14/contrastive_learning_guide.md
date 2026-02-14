# BERT + 对比学习改进方案

## 🎯 方案概述

基于最新研究（DeTeCtive NeurIPS 2024, Genre-Aware CL PAN@CLEF 2025），为你的毕设设计的对比学习改进方案。

### 核心创新点

| 技术 | 说明 | 预期效果 |
|-----|------|---------|
| **监督对比学习** | 同类样本拉近，异类样本推远 | 学习更本质的风格特征 |
| **硬负样本挖掘** | 重点关注最难区分的样本 | 提升边界情况性能 |
| **双任务学习** | 分类损失 + 对比损失联合优化 | 更好的泛化能力 |
| **投影头** | 768维→128维，更高效的对比计算 | 减少计算量 |

## 🚀 快速开始

```bash
cd /mnt/c/datacollection
source .venv/bin/activate

# 基础训练 (使用默认参数)
python scripts/training/train_bert_contrastive.py \
  --train-csv datasets/active/core_v1/train.csv \
  --val-csv datasets/active/core_v1/val.csv \
  --test-csv datasets/active/core_v1/test.csv \
  --output-dir models/bert_contrastive

# 推荐配置 (对比权重0.2, batch=32)
python scripts/training/train_bert_contrastive.py \
  --batch-size 32 \
  --contrastive-weight 0.2 \
  --temperature 0.07 \
  --num-epochs 5
```

## 📊 参数调优指南

### 关键参数

| 参数 | 推荐范围 | 说明 |
|-----|---------|------|
| `--contrastive-weight` | 0.1-0.3 | 对比损失权重，越大越注重特征学习 |
| `--temperature` | 0.05-0.1 | 越小越"尖锐"，对相似度更敏感 |
| `--batch-size` | 32-64 | **重要**：对比学习需要大batch |
| `--projection-dim` | 64-256 | 投影空间维度 |

### 实验配置建议

```bash
# 配置1: 保守配置 (低对比权重)
python scripts/training/train_bert_contrastive.py \
  --contrastive-weight 0.1 \
  --temperature 0.07 \
  --batch-size 32

# 配置2: 推荐配置 (平衡)
python scripts/training/train_bert_contrastive.py \
  --contrastive-weight 0.2 \
  --temperature 0.07 \
  --batch-size 32

# 配置3: 激进配置 (强对比)
python scripts/training/train_bert_contrastive.py \
  --contrastive-weight 0.3 \
  --temperature 0.05 \
  --batch-size 48

# 配置4: 禁用硬负样本 (对比baseline)
python scripts/training/train_bert_contrastive.py \
  --contrastive-weight 0.2 \
  --no-hard-negative
```

## 🔬 技术原理

### 1. 监督对比学习

```
Loss_con = -log(正样本相似度 / (正样本 + 负样本相似度))
```

- **正样本**：同一类别的其他样本
  - Human文本 ↔ 其他Human文本
  - AI文本 ↔ 其他AI文本
- **负样本**：不同类别的样本
  - Human文本 ↔ AI文本

### 2. 硬负样本挖掘

```python
# 在batch中找到最相似的负样本
hardest_neg = argmax(similarity(anchor, all_negatives))
# 对这些样本加权
weighted_neg = neg_samples + extra_weight * hardest_neg
```

### 3. 总损失

```
Total_Loss = (1-α) × 分类损失 + α × 对比损失

其中 α = contrastive_weight (默认0.2)
```

## 📈 预期效果

基于论文结果和我们的数据特点：

| 场景 | baseline (BERT) | 对比学习改进 |
|-----|-----------------|-------------|
| 整体准确率 | ~98% | 持平或小幅提升 |
| **OOD泛化** | 下降明显 | **显著改善** |
| **新模型检测** | 需重训 | **更好适应** |
| C2(续写)检测 | ~94% | 潜在提升 |

## 🧪 消融实验

建议进行以下对比实验：

```bash
# 实验1: Baseline (无对比学习)
python scripts/training/train_bert_improved.py --output-dir models/exp1_baseline

# 实验2: 对比学习 (权重0.1)
python scripts/training/train_bert_contrastive.py \
  --contrastive-weight 0.1 --output-dir models/exp2_con01

# 实验3: 对比学习 (权重0.2)
python scripts/training/train_bert_contrastive.py \
  --contrastive-weight 0.2 --output-dir models/exp3_con02

# 实验4: 对比学习 (权重0.3)
python scripts/training/train_bert_contrastive.py \
  --contrastive-weight 0.3 --output-dir models/exp4_con03

# 实验5: 无硬负样本
python scripts/training/train_bert_contrastive.py \
  --contrastive-weight 0.2 --no-hard-negative --output-dir models/exp5_no_hard
```

## 📝 毕设写作要点

### 可以在论文中强调的创新

1. **方法创新**：首次将监督对比学习应用于中文AI文本检测
2. **技术贡献**：设计了硬负样本挖掘策略
3. **实验验证**：在多场景(A-F)数据集上验证泛化性

### 实验设计建议

| 实验内容 | 目的 |
|---------|------|
| 不同对比权重对比 | 找到最优权重 |
| 有/无硬负样本对比 | 验证硬负样本有效性 |
| 跨场景泛化测试 | 验证OOD性能 |
| 跨模型泛化测试 | 验证对新LLM的适应性 |

## ⚠️ 注意事项

1. **Batch Size很重要**：对比学习需要大batch才能有足够的正负样本
   - 如果GPU显存不够，可以使用梯度累积

2. **温度参数敏感**：
   - 太小：训练不稳定
   - 太大：对比效果减弱
   - 推荐从0.07开始

3. **训练时间**：比baseline稍长（需要计算对比损失）

## 🔗 相关文件

```
scripts/training/
├── train_bert_contrastive.py   # ⭐ 新增：对比学习训练
├── train_bert_improved.py      # 原baseline
└── ...
```

---

*创建时间: 2026-01-28*
*基于: DeTeCtive (NeurIPS 2024), Genre-Aware CL (PAN@CLEF 2025)*
