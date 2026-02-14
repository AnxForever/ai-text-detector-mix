# 第2章 相关工作（可直接填充版）

> 📝 使用说明：补充具体引用和技术细节

---

## 2.1 大语言模型概述

### 2.1.1 Transformer架构

2017年，Vaswani等人提出了Transformer架构[1]，成为现代大语言模型的基础。Transformer的核心创新是**自注意力机制（Self-Attention）**，能够有效捕捉序列中的长距离依赖关系。

**自注意力机制**

给定输入序列 $X = \{x_1, x_2, ..., x_n\}$，自注意力计算如下：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中，$Q$、$K$、$V$ 分别为查询、键、值矩阵，$d_k$ 为键向量的维度。

**多头注意力**

为增强模型的表示能力，Transformer采用多头注意力：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

其中每个头独立计算注意力，最后拼接并线性变换。

### 2.1.2 GPT系列发展

GPT（Generative Pre-trained Transformer）系列是OpenAI推出的自回归语言模型：

**表2-1 GPT系列发展历程**

| 模型 | 发布时间 | 参数量 | 主要特点 |
|-----|---------|-------|---------|
| GPT-1 | 2018.06 | 117M | 首次提出预训练+微调范式 |
| GPT-2 | 2019.02 | 1.5B | 展示零样本学习能力 |
| GPT-3 | 2020.06 | 175B | 涌现能力，少样本学习 |
| GPT-3.5 | 2022.11 | ~175B | ChatGPT基座，RLHF对齐 |
| GPT-4 | 2023.03 | 未公开 | 多模态，更强推理能力 |

### 2.1.3 中文大语言模型

国内也涌现出多个优秀的中文大语言模型：

**表2-2 主流中文大语言模型**

| 模型 | 机构 | 特点 |
|-----|------|------|
| 文心一言 | 百度 | 中文理解能力强 |
| 通义千问 | 阿里 | 开源版本可用 |
| GLM | 智谱AI | 双向自回归架构 |
| DeepSeek | 深度求索 | 代码能力突出 |
| Kimi | Moonshot | 长文本处理能力 |

---

## 2.2 AI生成文本检测方法

### 2.2.1 基于统计特征的方法

早期的AI文本检测方法主要基于统计特征：

**困惑度（Perplexity）**

困惑度衡量语言模型对文本的"惊讶程度"：

$$PPL(x) = \exp\left(-\frac{1}{n}\sum_{i=1}^{n}\log P(x_i|x_{<i})\right)$$

AI生成的文本通常具有较低的困惑度，因为它们更符合模型的概率分布。

**词汇多样性**

计算文本中不同词汇的比例（Type-Token Ratio）。研究表明，AI生成文本的词汇多样性通常低于人类文本。

**突发度（Burstiness）**

衡量文本中词汇使用的不均匀程度。人类写作往往表现出更强的突发性（某些词集中出现）。

**局限性**

- 对短文本效果差
- 容易被简单改写绕过
- 不同领域阈值难以统一

### 2.2.2 基于深度学习的方法

近年来，基于深度学习的检测方法成为主流：

**RoBERTa分类器**

OpenAI的官方检测器使用RoBERTa模型进行微调，但因准确率不足于2023年下线。

**DetectGPT** (Mitchell et al., 2023)

提出基于困惑度扰动的零样本检测方法：
1. 对原文进行多次随机扰动
2. 计算扰动文本的困惑度变化
3. AI文本的扰动敏感度更高

**DNA-GPT** (Yang et al., 2024)

利用n-gram分析进行检测，关注AI文本中重复的局部模式。

**DeTeCtive** (NeurIPS 2024)

采用多级对比学习：
- Token级对比
- 句子级对比
- 文档级对比

提升了检测的鲁棒性和泛化能力。

### 2.2.3 基于水印的方法

另一类方法是在AI生成时嵌入水印：

**软水印**

通过轻微调整词汇分布，在生成文本中嵌入统计可检测的模式。

**硬水印**

在特定位置插入可识别的标记或序列。

**局限性**

- 需要模型方配合
- 可能影响生成质量
- 后处理可能破坏水印

---

## 2.3 预训练语言模型

### 2.3.1 BERT模型原理

BERT（Bidirectional Encoder Representations from Transformers）是Google于2018年提出的双向预训练语言模型[2]。

**预训练任务**

BERT采用两个预训练任务：

（1）**掩码语言模型（MLM）**：随机掩盖输入中15%的token，让模型预测被掩盖的词。

$$\mathcal{L}_{MLM} = -\sum_{i \in M}\log P(x_i|x_{\backslash M})$$

（2）**下一句预测（NSP）**：判断两个句子是否连续。

**模型结构**

| 参数 | BERT-base | BERT-large |
|-----|-----------|------------|
| 层数 | 12 | 24 |
| 隐藏维度 | 768 | 1024 |
| 注意力头 | 12 | 16 |
| 参数量 | 110M | 340M |

### 2.3.2 中文BERT变体

针对中文，研究者提出了多种BERT变体：

**表2-3 中文BERT变体对比**

| 模型 | 机构 | 特点 |
|-----|------|------|
| bert-base-chinese | Google | 基于字的中文BERT |
| chinese-roberta-wwm | 哈工大讯飞 | 全词掩码，RoBERTa优化 |
| ERNIE | 百度 | 知识增强 |
| MacBERT | 哈工大讯飞 | MLM as Correction |

**全词掩码（Whole Word Masking）**

中文分词后，将整个词一起掩盖，避免仅掩盖词的一部分：

```
原始: 使用语言模型
标准MLM: 使用语[MASK]模型
WWM: 使用[MASK][MASK]模型
```

### 2.3.3 微调策略

预训练模型的微调策略：

**分类任务微调**

在BERT输出的[CLS]向量上添加分类头：

$$P(y|x) = \text{softmax}(W \cdot h_{[CLS]} + b)$$

**学习率设置**

- 预训练层：较小学习率（如1e-5）
- 新增层：较大学习率（如1e-4）

**冻结策略**

- 全参数微调：所有参数可训练
- 部分冻结：仅训练顶层或分类头
- Adapter：插入轻量级适配模块

---

## 2.4 本章小结

本章综述了与中文AI文本检测相关的技术背景：

（1）**大语言模型**：从Transformer架构到GPT系列，以及国内主流中文大模型，这些模型的强大生成能力是AI文本检测研究的源动力。

（2）**检测方法**：从基于统计特征的方法到基于深度学习的方法，检测技术不断进步，但仍面临准确率和泛化性的挑战。

（3）**预训练语言模型**：BERT及其中文变体为文本分类任务提供了强大的语义表示能力，是本文方法的技术基础。

综合现有研究，本文选择基于中文BERT的分类方法，并针对多场景、混合文本等实际需求进行改进。

---

## 参考文献（部分）

[1] Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need[C]//NeurIPS. 2017: 5998-6008.

[2] Devlin J, Chang M W, Lee K, et al. BERT: Pre-training of deep bidirectional transformers for language understanding[C]//NAACL. 2019: 4171-4186.

[3] Guo B, Zhang X, Wang Z, et al. How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection[J]. arXiv preprint arXiv:2301.07597, 2023.

[4] Mitchell E, Lee Y, Khazatsky A, et al. DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature[C]//ICML. 2023.

【TODO: 补充完整参考文献列表】

---

*最后更新: 2026-01-28*
*提示: 根据学校要求调整参考文献格式*
