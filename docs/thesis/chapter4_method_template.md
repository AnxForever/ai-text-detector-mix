# 第4章 方法设计（可直接填充版）

> 📝 使用说明：核心内容已根据代码填充，部分细节需要补充

---

## 4.1 问题定义

中文AI文本检测任务可形式化定义如下：

给定一段中文文本 $x = \{x_1, x_2, ..., x_n\}$，其中 $x_i$ 表示第 $i$ 个字符，$n$ 为文本长度。目标是学习一个分类函数 $f: X \rightarrow Y$，将文本映射到标签空间 $Y = \{0, 1\}$，其中：

- $y = 0$ 表示文本由人类撰写
- $y = 1$ 表示文本由AI生成

形式化地，对于输入文本 $x$，模型输出预测标签：

$$\hat{y} = \arg\max_{c \in \{0,1\}} P(y=c|x)$$

---

## 4.2 模型架构

### 4.2.1 整体架构

本文提出的中文AI文本检测模型基于BERT预训练语言模型，整体架构如图4-1所示。

```
                    输入文本
                       ↓
              ┌────────────────┐
              │   Tokenizer    │
              │ (bert-base-    │
              │   chinese)     │
              └───────┬────────┘
                      ↓
         [CLS] token₁ token₂ ... tokenₙ [SEP]
                      ↓
              ┌────────────────┐
              │  BERT Encoder  │
              │  (12层 × 768维) │
              └───────┬────────┘
                      ↓
              [CLS]表示向量 (768维)
                      ↓
              ┌────────────────┐
              │  分类器头      │
              │  (Linear→Tanh  │
              │   →Dropout→    │
              │   Linear)      │
              └───────┬────────┘
                      ↓
              二分类输出 (Human/AI)
```

**图4-1 模型整体架构**

### 4.2.2 BERT编码器

模型采用bert-base-chinese作为预训练语言模型，其主要配置如表4-1所示。

**表4-1 BERT编码器配置**

| 参数 | 值 | 说明 |
|-----|-----|------|
| 隐藏层维度 | 768 | 每层Transformer的输出维度 |
| 注意力头数 | 12 | 多头自注意力的头数 |
| 隐藏层数 | 12 | Transformer编码器层数 |
| 最大位置编码 | 512 | 支持的最大序列长度 |
| 词表大小 | 21,128 | 中文词表包含的token数 |
| 参数量 | ~110M | 模型总参数数量 |

**输入表示**

输入文本经过分词后，每个token的表示由三部分相加得到：

$$E_{input} = E_{token} + E_{position} + E_{segment}$$

其中：
- $E_{token}$：Token嵌入，从词表中查找
- $E_{position}$：位置嵌入，编码token的位置信息
- $E_{segment}$：段落嵌入，区分不同句子

**[CLS]表示提取**

BERT编码后，取第一个位置（[CLS] token）的输出作为整个文本的语义表示：

$$h_{[CLS]} = \text{BERT}(x)[0] \in \mathbb{R}^{768}$$

### 4.2.3 分类器头设计

分类器头将[CLS]表示映射到二分类输出，结构如下：

```python
classifier = nn.Sequential(
    nn.Linear(768, 768),      # 全连接层
    nn.Tanh(),                 # 激活函数
    nn.Dropout(0.1),           # Dropout防止过拟合
    nn.Linear(768, 2)          # 输出层，2类
)
```

输出logits经过Softmax得到概率分布：

$$P(y|x) = \text{Softmax}(W_2 \cdot \text{Tanh}(W_1 \cdot h_{[CLS]} + b_1) + b_2)$$

---

## 4.3 训练策略

### 4.3.1 损失函数

**标准交叉熵损失**

基础训练使用交叉熵损失函数：

$$\mathcal{L}_{CE} = -\frac{1}{N}\sum_{i=1}^{N}[y_i\log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)]$$

其中：
- $N$ 为样本数
- $y_i$ 为真实标签
- $\hat{p}_i$ 为模型预测的AI文本概率

**长度加权损失（可选）**

为缓解不同长度文本的学习难度差异，可采用长度加权损失：

$$\mathcal{L}_{LW} = -\frac{1}{N}\sum_{i=1}^{N}w_i[y_i\log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)]$$

其中权重 $w_i$ 与文本长度相关：

$$w_i = 1 + \alpha \cdot \frac{len_i}{max\_length}$$

$\alpha$ 为超参数，默认值0.3。

### 4.3.2 优化器配置

采用AdamW优化器，配置如表4-2所示。

**表4-2 优化器配置**

| 参数 | 值 | 说明 |
|-----|-----|------|
| 优化器 | AdamW | 带权重衰减的Adam |
| 学习率 | 2e-5 | 初始学习率 |
| 权重衰减 | 0.01 | L2正则化系数 |
| 预热步数 | 500 | 学习率预热步数 |
| 调度策略 | 线性衰减 | 预热后线性衰减至0 |

**学习率调度**

采用带预热的线性衰减策略：

$$lr(t) = \begin{cases}
lr_{base} \cdot \frac{t}{t_{warmup}} & t < t_{warmup} \\
lr_{base} \cdot \frac{T - t}{T - t_{warmup}} & t \geq t_{warmup}
\end{cases}$$

其中 $T$ 为总训练步数。

### 4.3.3 训练技巧

**动态Padding**

为提高训练效率，采用batch级别的动态padding策略：

```python
def dynamic_collate_fn(batch):
    # 找到batch中最长序列
    max_len = max(len(x['input_ids']) for x in batch)
    # 将所有序列填充到相同长度
    padded_batch = pad_to_length(batch, max_len)
    return padded_batch
```

**梯度裁剪**

为防止梯度爆炸，训练时采用梯度裁剪：

$$g = \begin{cases}
g & \|g\| \leq 1.0 \\
\frac{g}{\|g\|} & \|g\| > 1.0
\end{cases}$$

**早停策略**

监控验证集F1分数，连续N个epoch无提升则停止训练，保存最佳模型。

---

## 4.4 对比学习增强（可选模块）

### 4.4.1 动机

标准分类方法仅学习决策边界，可能导致：
- 特征空间不够鲁棒
- 对对抗样本敏感
- 跨域泛化能力有限

对比学习通过学习样本间的相对关系，能够获得更鲁棒的特征表示。

### 4.4.2 监督对比损失

采用监督对比学习损失（Supervised Contrastive Loss）：

$$\mathcal{L}_{con} = -\frac{1}{|P(i)|}\sum_{p \in P(i)}\log\frac{\exp(z_i \cdot z_p / \tau)}{\sum_{a \in A(i)}\exp(z_i \cdot z_a / \tau)}$$

其中：
- $z_i$ 为样本 $i$ 的归一化特征向量
- $P(i)$ 为与样本 $i$ 同类的正样本集合
- $A(i)$ 为所有非自身样本的集合
- $\tau$ 为温度参数，默认0.07

### 4.4.3 硬负样本挖掘

为进一步提升性能，引入硬负样本挖掘策略：

1. 对每个样本，找到最相似的负样本（硬负样本）
2. 增加硬负样本的损失权重

$$w_{hard} = 1 + \beta \cdot \mathbf{1}[\text{is\_hardest\_negative}]$$

其中 $\beta$ 为硬负样本权重，默认0.5。

### 4.4.4 双任务联合学习

最终损失为分类损失与对比损失的加权和：

$$\mathcal{L}_{total} = (1-\alpha)\mathcal{L}_{CE} + \alpha\mathcal{L}_{con}$$

其中 $\alpha$ 为对比损失权重，推荐范围0.1-0.3。

**模型架构扩展**

```
输入文本
    ↓
BERT编码器 → [CLS]表示 (768维)
    ↓
    ├──→ 投影头 → 归一化特征 (128维) → 对比损失
    │    [Linear→ReLU→Dropout→Linear→L2Norm]
    │
    └──→ 分类头 → 分类logits (2维) → 分类损失
         [Linear→Tanh→Dropout→Linear]
```

**图4-2 对比学习增强模型架构**

---

## 4.5 超参数配置总结

**表4-3 完整超参数配置**

| 类别 | 参数 | 默认值 | 说明 |
|-----|------|-------|------|
| **模型** | model_name | bert-base-chinese | 预训练模型 |
| | max_length | 512 | 最大序列长度 |
| | dropout | 0.1 | Dropout比率 |
| **训练** | batch_size | 16 | 批次大小 |
| | learning_rate | 2e-5 | 学习率 |
| | num_epochs | 5 | 训练轮数 |
| | warmup_steps | 500 | 预热步数 |
| | weight_decay | 0.01 | 权重衰减 |
| **损失** | loss_alpha | 0.3 | 长度加权系数 |
| **对比学习** | contrastive_weight | 0.2 | 对比损失权重 |
| | temperature | 0.07 | 温度参数 |
| | projection_dim | 128 | 投影维度 |
| | hard_negative_weight | 0.5 | 硬负样本权重 |

---

## 4.6 本章小结

本章详细介绍了中文AI文本检测模型的设计与实现。主要内容包括：

（1）提出基于BERT的二分类模型架构，利用[CLS]表示进行文本级分类。

（2）设计了包含动态Padding、梯度裁剪、学习率调度等训练策略，提高训练效率和稳定性。

（3）引入对比学习增强模块，通过监督对比损失和硬负样本挖掘学习更鲁棒的特征表示。

（4）给出了完整的超参数配置，便于实验复现。

---

*最后更新: 2026-01-28*
*代码参考: scripts/training/train_bert_improved.py, train_bert_contrastive.py*
