# 第4章 基于 BERT 微调的中文 AI 文本二分类方法

> 本章介绍本文提出的中文 AI 文本检测方法的设计与实现。方法以 BERT 预训练语言模型的微调为核心，围绕"文档级二分类"主任务设计了融合标签平滑、长度感知、双维度加权采样与温度校准的完整训练流水线；在主任务基础上，本文进一步扩展出混合文本检测与边界定位能力，作为主方法的扩展研究。

---

## 4.1 问题定义

中文 AI 文本检测任务可形式化定义为二分类问题。设输入文本 $x = \{x_1, x_2, \ldots, x_n\}$，其中 $x_i$ 表示第 $i$ 个字符，$n$ 为文本长度。记标签空间 $\mathcal{Y} = \{0, 1\}$，其中 $y=0$ 表示文本由人类撰写，$y=1$ 表示文本由 AI 生成。本文目标是学习一个映射函数：

$$f_\theta: \mathcal{X} \to \mathcal{Y}, \quad f_\theta(x) = \arg\max_{c \in \mathcal{Y}} P_\theta(y=c \mid x)$$

其中 $\theta$ 为模型参数。与传统的特征工程方法不同，本文采用端到端的深度学习方法——将文本直接编码为分布式表示，并由参数化的神经网络直接学习后验概率分布 $P_\theta(y \mid x)$。在此基础上，本文在第 4.7 节进一步扩展出 Token 级边界定位任务，用于对混合文本（人类撰写片段与 AI 生成片段拼接而成的文本）进行细粒度分析。

---

## 4.2 模型总体框架

本文方法的整体架构如图 4-1 所示，分为主模型与扩展模型两部分。

```
                              输入文本 x
                                   │
                          ┌────────▼────────┐
                          │   BERT Tokenizer │
                          │ (bert-base-chinese)│
                          └────────┬────────┘
                                   │
                       [CLS] t₁ t₂ … tₙ [SEP]
                                   │
                          ┌────────▼────────┐
                          │   BERT Encoder  │
                          │  (12 层 × 768 维) │
                          └────────┬────────┘
                                   │
                         ┌─────────┴─────────┐
                         │                   │
              h[CLS] ∈ ℝ⁷⁶⁸               h_i ∈ ℝ⁷⁶⁸ (per token)
                         │                   │
               ┌─────────▼──────────┐  ┌────▼──────────────┐
               │  分类头 (Classifier) │  │ Token 分类头      │
               │  Linear→Tanh→       │  │ Linear→Softmax    │
               │  Dropout→Linear     │  │  (扩展模块)        │
               └─────────┬──────────┘  └────┬──────────────┘
                         │                   │
                  logits ∈ ℝ²          logits_i ∈ ℝ²
                         │                   │
             ┌───────────▼───────┐  ┌────────▼──────────┐
             │ Temperature Scaling │  │ 序列标注输出       │
             │   p̂ = softmax(z/T) │  │ {y₁, y₂, …, yₙ}   │
             └───────────┬───────┘  └───────────────────┘
                         │                    │
                  {Human / AI}         边界位置 b ∈ [1, n]
                    (主任务)                (扩展任务)
```

**图4-1 模型总体架构**

本文架构包含两个并行的预测头，二者共享 BERT 编码器作为骨干：

（1）**主模型（BERT 分类器）**：基于 `BertForSequenceClassification`，使用 `[CLS]` token 的最终层隐藏表示进行文档级二分类，输出 Human/AI 两类标签。该模型承担本文的核心任务——对任意输入中文文本判断其是人类撰写还是 AI 生成。

（2）**扩展模型（BERT Token 分类器）**：基于 `BertForTokenClassification`，为每个 token 输出 Human(0)/AI(1) 标签，用于混合文本的边界定位。该模块作为主任务的扩展研究，仅在混合文本分析场景下使用（详见第 4.7 节）。

两个模型独立训练、独立部署，通过级联推理策略组合使用（详见第 4.7.3 节）。本章 4.3–4.6 节主要围绕主模型的设计展开，4.7 节介绍扩展模型。

---

## 4.3 BERT 编码器与输入表示

### 4.3.1 预训练模型选型

本文选择 `bert-base-chinese`（Devlin et al., 2019）作为基础预训练语言模型，主要基于迁移学习（Transfer Learning）的理论依据与中文 AI 文本检测任务的实际需求。

**理论依据**：迁移学习的核心假设是"在大规模通用语料上学到的语言知识（词汇、句法、语义），可以迁移到下游特定任务中"。Howard 与 Ruder（2018）在 ULMFiT 工作中系统性地证明了：在小规模有标注数据集上，"预训练 + 微调"范式的性能可以显著超过从零训练的大模型。BERT 的预训练目标——遮蔽语言模型（Masked Language Modeling, MLM）与下一句预测（Next Sentence Prediction, NSP）——使其获得了丰富的双向语言表示能力，尤其适合判别任务。

**对比其他预训练模型**：本文在方法选型阶段对几种代表性预训练模型进行了比较，结果如表 4-1 所示。

**表4-1 预训练模型对比**

| 维度 | BERT (Encoder) | GPT (Decoder) | RoBERTa-wwm |
|------|---------------|--------------|-------------|
| 架构 | 双向编码器 | 单向解码器 | 双向编码器 |
| 擅长任务 | **分类、序列标注** | 生成、续写 | 分类、序列标注 |
| 参数量 | 110M | 175B+ | 110M |
| 推理成本 | 毫秒级、单卡 | 秒级、多卡 | 毫秒级、单卡 |
| 中文预训练语料 | 中文维基百科 | 中英混合 | 中文维基 + 全词遮蔽 |

综合考虑，BERT 的双向注意力机制使其每个 token 都能同时看到左右上下文，对判别任务天然优于只能看左侧上下文的 GPT 类解码器模型（Devlin et al., 2019）；而与同样双向的 RoBERTa-wwm 相比，`bert-base-chinese` 的开源社区支持更完整，在参数量、推理速度、部署成本上完全相当，因此本文最终选用 `bert-base-chinese`。

### 4.3.2 输入表示

输入文本首先经 BERT 分词器（WordPiece，词表大小为 21,128）分词，并在首部插入特殊 token `[CLS]`、尾部插入 `[SEP]`。对每个 token，BERT 将其表示为三部分嵌入之和：

$$E_{\text{input}}(x_i) = E_{\text{token}}(x_i) + E_{\text{position}}(i) + E_{\text{segment}}(x_i)$$

其中：
- $E_{\text{token}}(x_i) \in \mathbb{R}^{768}$：Token 嵌入，从预训练词向量表中查找；
- $E_{\text{position}}(i) \in \mathbb{R}^{768}$：位置嵌入，编码 token 在序列中的位置信息；
- $E_{\text{segment}}(x_i) \in \mathbb{R}^{768}$：段落嵌入，用于区分不同段落（在单文本二分类任务中全部为 0）。

### 4.3.3 编码器前向传播

BERT 编码器由 12 层 Transformer 块堆叠而成，每层包含多头自注意力（12 头）与前馈神经网络。输入嵌入 $E_{\text{input}} \in \mathbb{R}^{n \times 768}$ 经过 12 层编码后，得到每个 token 的最终层隐藏表示：

$$H = \text{BERT}_\theta(E_{\text{input}}) \in \mathbb{R}^{n \times 768}$$

本文取第一个位置（`[CLS]` token）的输出作为整个文本的语义表示：

$$h_{\text{[CLS]}} = H[0] \in \mathbb{R}^{768}$$

`[CLS]` token 的设计初衷正是用于序列级分类——在 BERT 预训练的 NSP 任务中，模型已经学会将整段文本的语义聚合到 `[CLS]` 位置。本文直接复用这一能力进行 AI 文本检测任务。

BERT 编码器的主要配置参数如表 4-2 所示。

**表4-2 BERT 编码器配置**

| 参数 | 值 | 说明 |
|-----|-----|------|
| 隐藏层维度 | 768 | 每层 Transformer 的输出维度 |
| 注意力头数 | 12 | 多头自注意力的头数 |
| 编码层数 | 12 | Transformer 编码器层数 |
| 前馈层维度 | 3,072 | 前馈网络中间层维度 |
| 最大位置编码 | 512 | 支持的最大序列长度 |
| 词表大小 | 21,128 | 中文词表 token 数 |
| 总参数量 | ~110M | 完整模型参数数量 |

---

## 4.4 分类头设计

在 BERT 编码器输出的 `[CLS]` 表示基础上，本文设计了如下分类头：

```python
classifier = nn.Sequential(
    nn.Linear(768, 768),   # 全连接层
    nn.Tanh(),             # 激活函数
    nn.Dropout(0.1),       # Dropout 防止过拟合
    nn.Linear(768, 2),     # 输出层
)
```

形式化地，给定 `[CLS]` 表示 $h_{\text{[CLS]}} \in \mathbb{R}^{768}$，分类头输出 logits：

$$z = W_2 \cdot \tanh(W_1 \cdot h_{\text{[CLS]}} + b_1) + b_2 \in \mathbb{R}^2$$

其中 $W_1 \in \mathbb{R}^{768 \times 768}$、$W_2 \in \mathbb{R}^{768 \times 2}$、$b_1 \in \mathbb{R}^{768}$、$b_2 \in \mathbb{R}^2$ 为可学习参数。Dropout 比率设置为 0.1。

在训练阶段，logits 直接用于交叉熵损失的计算（见 4.5.1 节）；在推理阶段，logits 需先经温度缩放再通过 Softmax 得到后验概率（见 4.6 节）：

$$P(y \mid x) = \text{Softmax}\left(\frac{z}{T}\right)$$

其中 $T$ 为温度参数，本文取 $T = 0.8165$（详见 4.6 节的优化过程）。

---

## 4.5 微调训练策略

本节介绍 V11c 模型的核心训练策略。与通用文本分类任务相比，中文 AI 文本检测面临两个独特挑战：（1）**长度偏差**——AI 生成文本平均长度显著长于人类文本，模型容易学到"长度 → 标签"的捷径特征；（2）**灰色地带**——存在大量风格模糊的样本（如 AI 轻微润色的人类文章），需要模型保留合理的不确定性。本文的训练策略针对这两个挑战进行了系统性设计，核心包括：长度感知标签平滑损失、双维度加权采样、AdamW + 线性调度、Early Stopping 与梯度裁剪等正则化手段。

### 4.5.1 损失函数：长度感知标签平滑损失

本文提出**长度感知标签平滑损失（Length-Aware Label Smoothing Loss, LA-LSL）**，将标签平滑（Label Smoothing）与长度感知加权融合为单一损失函数。完整公式如下：

**第 1 步**：基础交叉熵损失（带标签平滑）：

$$\mathcal{L}_{\text{CE}}^{(i)} = -\sum_{c=0}^{1} \tilde{y}_c^{(i)} \log \hat{p}_c^{(i)}$$

其中标签 $y^{(i)} \in \{0, 1\}$ 被平滑为：

$$\tilde{y}_c^{(i)} = \begin{cases} 1 - \epsilon & \text{若 } c = y^{(i)} \\ \frac{\epsilon}{K-1} & \text{否则} \end{cases}$$

本文取 $\epsilon = 0.05$，类别数 $K = 2$。

**第 2 步**：长度感知权重：

$$w^{(i)} = \frac{1}{1 + \lambda \cdot \left| \log \frac{L^{(i)}}{\mu} \right|}$$

其中 $L^{(i)}$ 为第 $i$ 个样本的字符长度，$\mu = 500$ 为参考长度，$\lambda = 0.1$ 为长度惩罚系数。该函数的几何意义：当样本长度 $L^{(i)} = \mu$ 时权重 $w^{(i)} = 1$（最高权重）；当长度偏离参考值（过长或过短）时权重降低，且下降幅度随对数距离线性扩大。

**第 3 步**：最终损失为加权平均：

$$\mathcal{L}_{\text{LA-LSL}} = \frac{1}{N} \sum_{i=1}^{N} w^{(i)} \cdot \mathcal{L}_{\text{CE}}^{(i)}$$

**理论依据**：

**（a）Label Smoothing**。Müller 等（2019）在 NeurIPS 的系统性研究表明，硬标签 $[0, 1]$ 会鼓励模型把 logit 推向正无穷（追求 100% 确信），导致过拟合与校准失效；而平滑后的标签让模型的隐藏表示形成更紧凑的类内聚类，同时保留合理的不确定性表达能力。这一特性对 AI 检测任务尤为重要，因为人机文本边界存在大量灰色地带（如 AI 轻微润色的人类文章、高度模仿人类风格的 AI 文本），模型需要能够说"我不太确定"。

**（b）长度感知加权与捷径学习理论**。Geirhos 等（2020）在 _Nature Machine Intelligence_ 提出**捷径学习（Shortcut Learning）**理论：深度神经网络倾向于学习数据中最容易利用的"捷径"特征，而非真正有意义的语义特征。在本文数据集中，AI 文本的平均长度约 650 字，人类文本平均长度约 280 字——若不加干预，模型很容易学到"长 → AI, 短 → Human"的虚假关联，在面对分布外数据（如短的 AI 评论、长的人类报告）时会系统性崩溃。本文的长度感知权重通过降低极端长度样本的梯度贡献，迫使模型无法依赖长度这一捷径特征，转而学习真实的语义差异。

LA-LSL 的 PyTorch 实现如下：

```python
class LengthAwareLabelSmoothingLoss(nn.Module):
    def __init__(self, label_smoothing=0.05, length_penalty_weight=0.1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(
            reduction='none', label_smoothing=label_smoothing
        )
        self.length_penalty_weight = length_penalty_weight

    def forward(self, logits, labels, lengths, mean_length=500):
        ce = self.ce(logits, labels)                      # [N]
        length_ratio = lengths / mean_length              # [N]
        weight = 1.0 / (1.0 + self.length_penalty_weight *
                        torch.abs(torch.log(length_ratio + 1e-6)))
        return (ce * weight).mean()                       # scalar
```

### 4.5.2 双维度加权采样

为进一步缓解训练数据中各类别、各长度分布的不均衡问题，本文在数据采样层引入 **`WeightedRandomSampler` 双维度加权采样**。具体地，将训练样本按字符长度划分为 6 个桶 $B = \{[0,100), [100,200), [200,500), [500,1000), [1000,2000), [2000, +\infty)\}$，并在此基础上与标签维度（Human/AI）做交叉加权：

$$w^{(i)} = \frac{1}{n_{b(i)}} \cdot \frac{1}{n_{y(i)}}$$

其中 $b(i)$ 为样本 $i$ 的长度桶编号，$y(i)$ 为其标签，$n_{b(i)}$ 为该长度桶的样本总数，$n_{y(i)}$ 为该标签类别的样本总数。每个 batch 的样本按 $w^{(i)}$ 有放回地抽取，使得模型在每个**（长度桶, 标签）**组合上都能得到均等的学习机会。

**理论依据**：He 与 Garcia（2009）在 IEEE TKDE 的综述指出，当训练数据各类别样本数量差异显著时，模型会偏向多数类——多数类获得更多梯度更新，少数类被"淹没"。相比仅做标签维度平衡的传统方法，本文的双维度加权采样在标签与长度两个维度上同时去除不均衡性，避免模型在某个（长度桶 × 标签）组合上出现系统性欠拟合。这一采样策略与 4.5.1 节的长度感知损失从梯度层面形成**三重防护体系**：损失层、采样层、评估层（详见第 5 章）。

### 4.5.3 优化器与学习率调度

模型训练使用 AdamW 优化器（Loshchilov & Hutter, 2019），该优化器在 Adam 基础上正确实现了权重衰减（Weight Decay），避免了 Adam 中权重衰减被自适应学习率缩放的问题。优化器配置如表 4-3 所示。

**表4-3 优化器与学习率调度配置**

| 参数 | 值 | 说明 |
|-----|-----|------|
| 优化器 | AdamW | 带解耦权重衰减的 Adam |
| 基础学习率 | 1e-5 | BERT 微调的标准推荐值 |
| 权重衰减 | 0.01 | L2 正则化系数 |
| 预热步数 | 500 | 学习率线性预热步数 |
| 调度器 | 线性衰减 | 预热后线性衰减至 0 |
| 有效 batch size | 32 | batch=8 × 梯度累积 4 步 |

学习率调度采用 `get_linear_schedule_with_warmup`，具体函数为：

$$\eta(t) = \begin{cases}
\eta_0 \cdot \dfrac{t}{t_{\text{warmup}}} & t < t_{\text{warmup}} \\[6pt]
\eta_0 \cdot \dfrac{T - t}{T - t_{\text{warmup}}} & t \geq t_{\text{warmup}}
\end{cases}$$

其中 $\eta_0 = 10^{-5}$ 为基础学习率，$t_{\text{warmup}} = 500$ 为预热步数，$T$ 为总训练步数。预热阶段的缓慢增长有助于在训练初期避免梯度震荡破坏预训练权重；衰减阶段的线性下降则在训练后期降低学习率，使模型收敛至稳定点。

### 4.5.4 正则化与早停

本文在训练过程中使用多层正则化手段，以防止过拟合并提升模型泛化能力：

（1）**Dropout**：在分类头的 Tanh 激活与输出层之间设置 Dropout（比率 0.1），随机丢弃部分神经元，缓解共适应现象。

（2）**梯度裁剪**：训练中对梯度 L2 范数进行裁剪，防止梯度爆炸破坏预训练权重：

$$g \leftarrow \begin{cases} g & \|g\|_2 \leq 1.0 \\ g \cdot \dfrac{1.0}{\|g\|_2} & \|g\|_2 > 1.0 \end{cases}$$

（3）**Early Stopping**：在每个 Epoch 结束后计算验证集 F1 分数，若连续 `patience=2` 个 Epoch 无提升则提前终止训练，并保存最佳权重。本文最终模型训练 4 个 Epoch 后提前终止，最佳权重出现在第 2 个 Epoch（验证准确率 98.75%，F1=0.9883）。

（4）**梯度累积**：由于 GPU 显存限制（RTX 5060 Laptop 8GB），实际 batch size 仅为 8，通过梯度累积 4 步实现有效 batch size 32。梯度累积仅影响优化器的更新频率，不影响参数更新的数学等价性。

### 4.5.5 动态与静态 Padding 的权衡

本文训练阶段采用**静态 padding**（`padding='max_length'`，max_length=256）以保证 batch 内张量形状一致，便于 GPU 并行计算；而推理阶段采用**动态 padding**（`padding=True`，填充至 batch 内最长序列）以最小化无效计算量。这一差异的原因在于：训练阶段的静态 padding 保证了所有 batch 形状相同，CUDA kernel 调用开销最小；而推理阶段面对的是实时请求，动态 padding 能够根据实际请求长度压缩计算量，将长文本推理时间从 24 秒降至 8 秒（详见第 5 章效率分析）。

---

## 4.6 推理与概率校准

训练完成的分类器输出 logits 仅反映类别的相对置信度，不具备真实概率含义。为使模型输出的置信度可直接作为决策依据，本文在推理阶段采用 **Temperature Scaling**（温度缩放，Guo et al., 2017）进行后置校准。

**理论依据**：Guo 等在 ICML 2017 的工作系统性地发现，现代深度神经网络几乎普遍存在**过度自信**（Overconfidence）现象——模型说"我 99% 确定"时，实际正确率可能只有 85%。在 AI 文本检测这一高风险应用场景中（如学术不端检测），校准良好的置信度是避免误判的关键。Temperature Scaling 是该文献中推荐的最简单且最有效的后置校准方法，具有以下优势：

（1）**只引入单个参数 $T$**，易于优化；
（2）**不改变预测结果**（$\arg\max$ 不变，准确率不变），只改善置信度的可靠性；
（3）**通用性强**，可与任何基于 Softmax 输出的分类器结合。

**温度缩放公式**：给定原始 logits $z \in \mathbb{R}^K$，温度缩放后的概率为：

$$P(y = c \mid x) = \frac{\exp(z_c / T)}{\sum_{k=0}^{K-1} \exp(z_k / T)}$$

其中 $T > 0$ 为温度参数。$T = 1$ 时退化为标准 Softmax；$T < 1$ 时分布更尖锐（增强自信度）；$T > 1$ 时分布更平滑（降低自信度）。

**温度参数优化**：本文在独立评估集（independent_data，910 条）上通过最小化负对数似然（NLL）来求解最优温度：

$$T^* = \arg\min_T \left\{ -\frac{1}{N} \sum_{i=1}^{N} \log P_T(y^{(i)} \mid x^{(i)}) \right\}$$

使用 L-BFGS 优化器求解得 $T^* = 0.7872$（独立评估集单独优化）与 $T^* = 0.8165$（三集联合优化，本文最终使用值）。校准前后的期望校准误差（Expected Calibration Error, ECE）变化如表 4-4 所示。

**表4-4 Temperature Scaling 校准效果**

| 指标 | 校准前 ($T=1$) | 校准后 ($T=0.8165$) | 改善 |
|-----|---------------|-------------------|------|
| ECE（期望校准误差） | 0.0168 | **0.0034** | −79.8% |
| 高置信错误数 | 11 | 12 | +1 |
| Accuracy | 98.69% | 98.69% | ±0（不变）|

校准后 ECE 从 0.0168 下降至 0.0034，意味着模型输出的置信度与真实正确率之间的偏差不超过 0.34%——在实际部署中，当模型输出"95% 是 AI"时，可以高置信度地相信其接近真实概率。

---

## 4.7 扩展能力：混合文本检测与边界定位

本文在二分类主任务基础上，进一步扩展出**混合文本检测与边界定位能力**。该扩展研究聚焦于"人类开头 + AI 续写"这一日益常见的混合写作场景，由三个相互配合的机制组成：（1）`[SEP]` 边界标记机制，（2）Token 级边界检测器，（3）双层级联推理架构。需要说明的是，这些机制属于对主二分类任务的**扩展研究**，不改变本文以"文档级二分类"为核心的任务设定。

### 4.7.1 [SEP] 边界标记机制

**动机**：混合文本中人类撰写部分与 AI 生成部分在语言风格上存在明显断裂（Stamatatos, 2009, JASIST），但若将二者拼接为连续文本直接输入模型，模型难以定位到这一风格转换点。本文借助 BERT 预训练中 NSP 任务赋予模型的"段落分界识别能力"，在混合文本的人类/AI 分界处显式插入 `[SEP]` 特殊 token 作为分界信号：

```
人类写的部分 [SEP] AI 续写的部分
```

**理论依据**：这一设计站在两个理论基础之上：

**理论 A：NSP 预训练中 `[SEP]` 的分界语义**。BERT 在 NSP 预训练任务中学习判断"两个句子是否相邻"，这使模型天然将 `[SEP]` token 视为"两段文本的分界信号"。模型在预训练阶段已经学会对 `[SEP]` 两侧的主题、风格、逻辑关系差异产生高灵敏度。

**理论 B：文体学作者归因理论**。Stamatatos（2009）在作者归因方法综述中指出，不同作者具有可量化的写作"指纹"——包括词汇选择、句式结构、语篇衔接方式等维度。在本文场景下，人类与 AI 是两个"不同的作者"，其写作指纹在混合文本的边界处发生突变。

**两理论的结合**：`[SEP]` 标记相当于给模型提供了一个**显式的归纳偏置（Inductive Bias）**——"这里有一个风格转换点，请重点关注"。模型因此能将注意力集中在边界附近的特征差异上，而无需在无先验的情况下自行定位风格断裂点。

**实验效果**：引入 `[SEP]` 边界标记后，混合文本中"人类开头 + AI 续写"（C2 类型）的检测准确率从 79.82% 提升至 93.84%，绝对提升 **14.02 个百分点**。这一结果从经验层面验证了 `[SEP]` 机制的有效性（详细实验见第 5.2.3 节）。

### 4.7.2 Token 级边界检测器

在主分类器判定文本为混合类型后，需进一步定位人类/AI 的具体边界位置。本文将这一任务建模为**序列标注问题（Sequence Labeling）**——即命名实体识别（NER）的经典范式——并使用 `BertForTokenClassification` 实现。

**任务定义**：给定输入 token 序列 $\{t_1, t_2, \ldots, t_n\}$，模型为每个 token 输出标签 $y_i \in \{0, 1\}$，其中 0 表示该 token 属于人类撰写片段，1 表示属于 AI 生成片段。

**模型架构**：在 BERT 编码器基础上，对每个 token 的最终层隐藏表示 $h_i \in \mathbb{R}^{768}$ 附加一个线性分类层：

$$z_i = W_{\text{tok}} \cdot h_i + b_{\text{tok}}, \quad W_{\text{tok}} \in \mathbb{R}^{768 \times 2}, \ b_{\text{tok}} \in \mathbb{R}^2$$

损失函数为逐 token 的交叉熵损失（忽略 padding 位置，通过标记为 −100 实现）：

$$\mathcal{L}_{\text{span}} = -\frac{1}{\sum_i \mathbb{1}[y_i \neq -100]} \sum_i \mathbb{1}[y_i \neq -100] \cdot \log P(y_i \mid t_i, \text{context})$$

**理论依据**：序列标注范式在 NLP 领域是一个被广泛验证的成熟方法，经典 NER 任务的 BIO 标注体系、分词任务的 BMES 体系均属此类。将边界检测建模为序列标注，天然继承了以下优势：（1）每个 token 的标签由全局上下文共同决定（通过 Transformer 的自注意力机制）；（2）可以处理任意长度和任意数量的边界；（3）直接复用成熟的 `BertForTokenClassification` 实现。

**边界后处理**：模型输出的逐 token 预测经过简单后处理（寻找 0→1 的标签转换点）即可得到边界位置。在本文测试集上，边界定位的中位误差为 3 字符，最大误差不超过 8 字符（详见第 5.4 节）。

### 4.7.3 双层级联推理架构

在线上推理阶段，本文组合主分类器与边界检测器构成**双层级联（Cascade）架构**：

```
Step 1: 主分类器（文档级）
        ├── 输出 Human → 直接返回
        ├── 输出 AI   → 直接返回
        └── 输出 Mixed → 进入 Step 2
                            ↓
Step 2: 边界检测器（Token 级）
        └── 输出边界位置 → 返回 {label=Mixed, boundary=b}
```

**理论依据**：级联分类器（Cascade Classifier）思想源自 Viola 与 Jones（2001）在 CVPR 的人脸检测经典工作——先用快速、粗粒度的分类器筛选大部分简单样本，再用精细、耗时的分类器处理困难样本。本文架构完美契合这一思想：

（1）**计算效率**：生产环境中 90% 以上的输入文本为纯人类或纯 AI 文本，仅需主分类器判断即可；Token 级边界检测器的单样本计算开销约为主分类器的 2 倍，级联架构能将这部分开销限制在 10% 以下的"混合"样本上。

（2）**精度优化**：Token 级模型在纯文本上可能产生"伪边界"假阳性；级联架构保证边界检测器只处理已被主分类器判定为"混合"的样本，从源头避免这类错误。

（3）**数据效率**：主分类器使用 63,113 条训练数据，边界检测器仅使用 2,034 条标注了 token 级标签的训练数据——级联架构使得标注稀缺的扩展模块不必承担通用分类职责。

---

## 4.8 超参数配置总结

本文方法的完整超参数配置如表 4-5 所示，所有数值均取自 V11c 训练脚本（`scripts/training/train_v11c.py`）的最终配置。

**表4-5 完整超参数配置**

| 类别 | 参数 | 值 | 说明 |
|-----|------|-----|------|
| **模型** | base_model | bert-base-chinese | 预训练模型 |
| | max_length | 256 | 最大序列长度 |
| | hidden_dim | 768 | BERT 隐藏层维度 |
| | num_layers | 12 | Transformer 层数 |
| | dropout | 0.1 | Dropout 比率 |
| **数据** | min_text_length | 10 | 最小有效字符数 |
| | max_text_length | 5,000 | 最大有效字符数 |
| | length_bins | [0,100,200,500,1000,2000,∞) | 长度分桶边界 |
| **训练** | batch_size | 8 | 微批次大小 |
| | accum_steps | 4 | 梯度累积步数（有效 batch=32） |
| | learning_rate | 1e-5 | 基础学习率 |
| | epochs | 最多 5（Early Stopping 最佳 Epoch 2） | 训练轮数 |
| | warmup_steps | 500 | 学习率预热步数 |
| | weight_decay | 0.01 | 权重衰减 |
| | optimizer | AdamW | 优化器 |
| | scheduler | linear_with_warmup | 学习率调度 |
| | grad_clip | 1.0 | 梯度裁剪阈值 |
| | patience | 2 | Early Stopping 容忍 epoch 数 |
| **损失** | label_smoothing | 0.05 | 标签平滑系数 |
| | length_penalty_weight | 0.1 | 长度惩罚权重 $\lambda$ |
| | mean_length | 500 | 长度权重参考值 $\mu$ |
| **采样** | sampler | WeightedRandomSampler | 双维度加权 |
| | replacement | True | 有放回抽样 |
| **推理** | temperature | 0.8165 | 温度缩放参数 $T^*$ |
| | decision_threshold | 0.8 | 判定阈值（可配置） |
| | padding_strategy | dynamic | 推理动态 padding |

---

## 4.9 本章小结

本章系统介绍了本文提出的基于 BERT 微调的中文 AI 文本二分类方法。主要内容包括：

（1）**问题定义与总体框架**。将任务形式化为二分类问题 $f_\theta: \mathcal{X} \to \{0, 1\}$，整体框架包括主模型（BERT 分类器）与扩展模型（BERT Token 分类器）两部分，主模型承担文档级二分类任务，扩展模型用于混合文本的边界定位。

（2）**基于 bert-base-chinese 的编码器设计**。依据迁移学习理论（Devlin et al., 2019；Howard & Ruder, 2018）选择 `bert-base-chinese` 作为预训练模型，通过三元嵌入（Token/Position/Segment）表示输入，取 `[CLS]` token 的最终层隐藏表示作为整段文本的语义向量。

（3）**微调训练策略**。提出**长度感知标签平滑损失**（融合 Müller et al., 2019 的标签平滑与 Geirhos et al., 2020 的捷径学习防护）、**双维度加权采样**（He & Garcia, 2009 的类别不平衡学习），配合 AdamW + 线性预热线性衰减调度、梯度裁剪、Early Stopping 等正则化手段，构成了完整的训练流水线。

（4）**推理与概率校准**。采用 Guo et al.（2017）的 Temperature Scaling 对输出概率进行后置校准，将期望校准误差从 0.0168 降低至 0.0034，保证模型输出的置信度可直接作为决策依据。

（5）**扩展能力**。在主二分类任务基础上，扩展出以 `[SEP]` 边界标记（NSP + 文体学理论）、Token 级边界检测器（序列标注范式）、双层级联架构（Viola & Jones, 2001）为核心的混合文本分析能力，使本文方法不仅能判断"是否 AI 生成"，还能定位"AI 生成的具体位置"。

本章介绍的方法在第 5 章的实验中得到了充分的验证：主分类任务的加权平均准确率达到 98.69%、F1 达 97.75%，召回率 99.28% 在 5 类对比方法中居首；扩展任务的 C2 混合文本检测率从 79.82% 提升至 93.84%；Temperature Scaling 的校准效果使 ECE 降至 0.0034。这些结果共同证明了本文方法的有效性与工程实用性。

---

## 参考文献（第 4 章引用）

1. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL-HLT*, 4171–4186.
2. Howard, J., & Ruder, S. (2018). Universal Language Model Fine-tuning for Text Classification. *ACL*, 328–339.
3. Müller, R., Kornblith, S., & Hinton, G. (2019). When Does Label Smoothing Help? *NeurIPS*.
4. Geirhos, R., Jacobsen, J. H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., & Wichmann, F. A. (2020). Shortcut Learning in Deep Neural Networks. *Nature Machine Intelligence*, 2(11), 665–673.
5. He, H., & Garcia, E. A. (2009). Learning from Imbalanced Data. *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263–1284.
6. Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. *ICLR*.
7. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *ICML*, 1321–1330.
8. Stamatatos, E. (2009). A Survey of Modern Authorship Attribution Methods. *JASIST*, 60(3), 538–556.
9. Viola, P., & Jones, M. (2001). Rapid Object Detection using a Boosted Cascade of Simple Features. *CVPR*, 511–518.

---

*最后更新: 2026-04-19*
*方法原型: scripts/training/train_v11c.py（主分类器）、scripts/training/train_span_detector.py（边界检测器）*
*理论基础: docs/thesis/theoretical_foundations.md（10 节技术决策的完整理论依据与答辩口径）*
