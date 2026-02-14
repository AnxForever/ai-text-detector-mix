# 训练方法分析与改进方案

**日期**: 2026-02-09

## 一、当前训练集问题

### 1.1 数据质量问题

| 问题 | 数量 | 影响 |
|-----|------|-----|
| **长度偏差** | AI均值965字, Human均值413字 | 模型可能学习"长=AI"捷径 |
| 过短文本 (<50字符) | 1,980条 (3.28%) | 信息量不足 |
| 重复文本 | 188条 | 过拟合风险 |
| AI模板短语 | 9条 | 泄露标签 |

### 1.2 长度分布对比

```
Human: 均值=413, 中位数=203, std=503
AI:    均值=965, 中位数=634, std=922
比值:  AI/Human = 2.34x
```

**问题**: 模型可能仅通过文本长度就能达到较高准确率，而非学习到真正的语言特征差异。

## 二、前沿技术调研

### 2.1 DeTeCtive (NeurIPS 2024) ⭐ 最新SOTA

**论文**: [Detecting AI-generated Text via Multi-Level Contrastive Learning](https://arxiv.org/abs/2410.20964)

**核心思想**:
- 将AI文本检测重新定义为"区分不同作者写作风格"的任务
- 每个LLM视为一个特定"作者"，有独特的写作风格
- 多层次对比学习捕获细粒度特征

**关键技术**:
1. **Multi-Level Contrastive Learning**: 在不同粒度级别学习对比特征
2. **Multi-Task Auxiliary**: 辅助任务增强主任务
3. **TFIA (Training-Free Incremental Adaptation)**: 无需重训练的领域适应

**GitHub**: https://github.com/heyongxin233/DeTeCtive

### 2.2 其他前沿方法

| 方法 | 来源 | 特点 |
|-----|------|-----|
| LLM-conditional Feature Alignment | ICIG 2025 | 学习领域不变特征 |
| Dynamic Contrastive Learning | ICIG 2025 | 增强扰动鲁棒性 |
| RoBERTa-based Detection | Frontiers 2025 | 比BERT更强的基线 |
| CoCo | EMNLP 2023 | 低资源下的对比学习 |

### 2.3 关键发现

1. **RoBERTa > BERT**: RoBERTa在此任务上表现更好
2. **对比学习有效**: 多层次对比学习显著提升泛化能力
3. **风格建模**: 将问题建模为风格区分比二分类更有效
4. **跨域鲁棒性**: OOD (Out-of-Distribution) 泛化是关键挑战

## 三、改进方案

### 3.1 短期改进 (已实现)

**脚本**: `scripts/training/train_bert_improved_v2.py`

| 改进项 | 说明 |
|-------|------|
| 数据清洗 | 移除过短/重复/模板文本 |
| 长度感知损失 | 降低极端长度样本的权重 |
| 长度平衡采样 | 可选，确保各长度区间平衡 |
| 按长度评估 | 分析不同长度区间的准确率 |

### 3.2 中期改进 (建议)

1. **换用RoBERTa**: `hfl/chinese-roberta-wwm-ext`
2. **对比学习预训练**: 参考DeTeCtive的对比损失
3. **数据增强**: 随机删词、同义词替换

### 3.3 长期改进 (可选)

1. **完整实现DeTeCtive**: 多层次对比学习
2. **多模型检测**: 针对不同LLM训练专门检测器
3. **对抗训练**: 使用改写/润色样本增强鲁棒性

## 四、推荐训练命令

### 方案A: 快速训练 (当前方法改进)

```bash
python scripts/training/train_bert_improved_v2.py \
    --data datasets/merged_v2 \
    --output models/bert_v7_improved \
    --epochs 3 \
    --length-penalty 0.1
```

### 方案B: 启用长度平衡采样

```bash
python scripts/training/train_bert_improved_v2.py \
    --data datasets/merged_v2 \
    --output models/bert_v7_balanced \
    --epochs 3 \
    --length-penalty 0.1 \
    --use-balanced-sampler
```

## 五、评估重点

训练后重点测试:

1. **技术文档**: 之前漏检的"梯度爆炸"类文本
2. **短文本**: <200字符的Human/AI文本
3. **长文本**: >1000字符的Human/AI文本
4. **跨领域**: 训练集未见过的领域

## 六、参考资源

- [DeTeCtive GitHub](https://github.com/heyongxin233/DeTeCtive)
- [Awesome LLM Detection Papers](https://github.com/Xianjun-Yang/Awesome_papers_on_LLMs_detection)
- [DeTeCtive论文](https://arxiv.org/abs/2410.20964)
- [AI Text Detection Survey (2025)](https://www.sciencedirect.com/science/article/abs/pii/S0957417425003161)

---

*报告生成时间: 2026-02-09*
