# AI 生成文本检测领域前沿技术调研报告 (2024-2026)

> 调研日期: 2026-02-10
> 调研范围: 学术论文 (NeurIPS, ICLR, ACL, EMNLP), 竞赛 (PAN CLEF, SemEval, GenAI Detection), 商业工具, 开源项目

---

## 目录

1. [最新模型架构](#1-最新模型架构)
2. [训练技巧](#2-训练技巧)
3. [中文文本检测](#3-中文文本检测)
4. [知名工具与系统](#4-知名工具与系统)
5. [数据集与评测](#5-数据集与评测)
6. [与本项目的关联分析](#6-与本项目的关联分析)

---

## 1. 最新模型架构

### 1.1 基于 BERT/RoBERTa 的改进方法

#### (a) DistilBERT + NLP 特征融合

- **论文**: "Identifying AI-generated content using the DistilBERT transformer and NLP techniques"
- **发表**: Scientific Reports, 2025年7月
- **作者**: Hikmat Ullah Khan 等
- **关键思路**: 在 DistilBERT 之上结合传统 NLP 特征（词汇多样性、句法复杂度等），利用轻量级模型实现高效检测
- **意义**: 证明了轻量级 Transformer 在充分特征工程下也能达到较好的检测效果

#### (b) RoBERTa 双流特征融合 (中文)

- **论文**: "Research on AI-generated Chinese text detection method based on deep learning"
- **发表**: Big Data and Information Analytics, 2025年12月
- **作者**: Chang Su, Yaqi Jiang, Jianlin Wang, Junfang Zhao (中国地质大学)
- **关键思路**: 提出**双流特征融合模型**，将 RoBERTa 语义编码与手工设计的文本统计特征结合
- **数据**: 构建了包含 HC3 数据集、ChatGPT 检测数据集以及自建学术摘要和文学作品数据集的跨领域混合多源语料库
- **性能**: 对 Phi4 生成文本的 Recall 达到 100%；对 DeepSeek R1 和 Qwen 2.5 生成文本也有优异表现

#### (c) PAN CLEF 2025 中的 BERT/RoBERTa 应用

- **任务**: Voight-Kampff Generative AI Detection (PAN @ CLEF 2025)
- **最佳方法**:
  - Subtask 1 (二分类): 微调 `bert-base-uncased`，最佳系统 mean score 达到 **0.99**
  - Subtask 2 (多分类协作文本): 微调 `roberta-large`，结合数据增强处理类不平衡
- **数据增强**: 回译(backtranslation)、同义词/反义词替换、随机删除

### 1.2 LLM 时代的新检测方法

#### (a) HART: 层次化 AI 风险检测框架 + 2D 检测方法

- **论文**: "Decoupling Content and Expression: Two-Dimensional Detection of AI-Generated Text"
- **发表**: arXiv 2503.00258, 2025年3月
- **作者**: Guangsheng Bao 等 (西湖大学/浙江大学)
- **核心创新**:
  1. **HART 框架**: 提出层次化 AI 风险等级(Hierarchical AI Risk in Text Creation)，将检测任务系统化分级
  2. **2D 检测方法**: 将文本解耦为**内容(Content)**和**语言表达(Expression)**两个维度
  3. **关键发现**: 内容维度对表面级别修改具有抗干扰性，可作为检测的关键特征
- **性能**: AUROC 从 0.705 提升至 **0.849** (Level-2 检测)，RAID 基准上从 0.807 提升至 **0.886**
- **开源**: https://github.com/baoguangsheng/truth-mirror

#### (b) Lastde/Lastde++: 无训练 LLM 文本检测 (ICLR 2025)

- **论文**: "Training-free LLM-generated Text Detection by Mining Token Probability Sequences"
- **发表**: ICLR 2025 (Poster)
- **作者**: Yihuai Xu 等 (浙江大学)
- **核心创新**:
  1. **首次引入时间序列分析**到 LLM 文本检测，捕捉 token 概率序列的时域动态
  2. 融合**局部统计特征**与**全局统计特征**
  3. **Lastde++**: 高效变体，支持实时检测
- **优势**: 无需训练数据，在跨域、跨模型、跨语言场景下均达到 SOTA
- **对抗鲁棒性**: 对释义攻击(paraphrasing attacks)表现出更强的鲁棒性
- **开源**: https://github.com/TrustMedia-zju/Lastde_Detector

#### (c) Binoculars: 零样本 LLM 检测

- **论文**: "Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text"
- **发表**: ICML 2024, 持续更新至 2025
- **作者**: Abhimanyu Hans 等
- **核心创新**: 利用两个紧密相关的语言模型计算**交叉困惑度比值**作为检测分数
- **性能**: 在 FPR=0.01% 的条件下检测 ChatGPT 生成文本的 TPR 超过 **90%**
- **关键优势**: 零样本、无需任何训练数据、适用于多种 LLM
- **开源**: https://github.com/ahans30/Binoculars
- **影响**: 被 PAN CLEF 2025 用作官方 Baseline

#### (d) 基于指令微调 LLM 的检测

- **论文**: "AI Generated Text Detection Using Instruction Fine-tuned Large Language and Transformer-Based Models"
- **发表**: arXiv 2507.05157, 2025年7月
- **作者**: Chinnappa Guggilla 等 (Deloitte)
- **核心思路**: 利用指令微调的大语言模型（如 LLaMA 系列）进行文本检测，将检测任务转化为 LLM 的指令遵循任务

### 1.3 对比学习 (Contrastive Learning) 在文本检测中的应用

#### (a) DeTeCtive: 多级对比学习框架 (NeurIPS 2024)

- **论文**: "DeTeCtive: Detecting AI-generated Text via Multi-Level Contrastive Learning"
- **发表**: NeurIPS 2024 (Poster)
- **作者**: Xun Guo, Yongxin He, Shan Zhang 等 (字节跳动/中科院大学)
- **核心创新**:
  1. **核心论点**: 检测 AI 文本的关键在于**区分不同作者的写作风格**，而非简单二分类
  2. **多任务辅助 + 多级对比学习**框架，学习不同写作风格的区分表示
  3. **密集信息检索管道**(Dense Information Retrieval Pipeline)用于推理阶段
  4. **TFIA (Training-Free Incremental Adaptation)**: 无训练增量适应能力，面对 OOD 数据可动态扩展
- **性能**: 在 OOD 零样本评估中**大幅超越**现有方法
- **兼容性**: 与多种文本编码器兼容 (BERT, RoBERTa, DeBERTa 等)
- **开源**: https://github.com/heyongxin233/DeTeCtive

#### (b) DETree: 树结构层次表示学习 (NeurIPS 2025)

- **论文**: "DETree: DEtecting Human-AI Collaborative Texts via Tree-Structured Hierarchical Representation Learning"
- **发表**: NeurIPS 2025 (Poster)
- **作者**: Yongxin He, Shan Zhang, Yixuan Cao 等 (中科院计算所)
- **核心创新**:
  1. **问题建模**: 识别到 AI 文本生成包含多种协作过程（AI写人编辑、人写AI编辑、AI生成AI精炼），不同过程的文本表示具有**内在聚类关系**
  2. **树结构层次表示学习**: 将不同协作过程建模为层次化树结构，而非传统的扁平多分类
  3. 利用树结构进行**层次化对比学习**
- **优势**: 对未见过的新 LLM 具有更强的泛化能力
- **开源**: https://github.com/heyongxin233/DETree

#### (c) Sci-SpanDet: Span 级对比学习 + 结构校准

- **论文**: "Span-level detection of AI-generated scientific text via contrastive learning and structural calibration"
- **发表**: Knowledge-Based Systems, Volume 334, 2026年2月
- **作者**: Zhen Yin, Shenghua Wang
- **核心创新**:
  1. **Span 级别定位**: 非文档级，而是精确到文本片段(Span)的 AI 生成检测
  2. **Section-conditioned 风格建模**: 结合论文章节结构进行风格化建模
  3. **多级对比学习**: 捕捉人类与 AI 之间的细微差异，减轻主题依赖
  4. **BIO-CRF 序列标注 + 指针式边界解码**: 联合边界预测
- **意义**: **与本项目的 `[SEP]` 边界标记机制高度相关**，提供了更先进的 Span 级检测范式

### 1.4 多粒度检测（句子级、段落级、Token 级）

#### (a) 句子级: SenDetEX (EMNLP 2025)

- **论文**: "SenDetEX: Sentence-Level AI-Generated Text Detection for Human-AI Hybrid Content via Style and Context Fusion"
- **发表**: EMNLP 2025, 苏州
- **作者**: Lei Jiang, Desheng Wu, Xiaolong Zheng (中科院自动化所/国科大)
- **核心创新**:
  1. **AutoFill-Refine**: 高质量人机混合文本合成策略
  2. **Style + Context 双融合**: 同时利用风格特征和上下文信息进行句子级判断
  3. 构建了专用的 S-AGTD（句子级AI文本检测）Benchmark
- **性能**: 显著超越所有 baseline，具有良好的迁移性和鲁棒性
- **开源**: https://github.com/TristoneJiang/SenDetEX

#### (b) 句子级: Transformer + NN + CRF

- **论文**: "Fine-Grained Detection of AI-Generated Text Using Sentence-Level Segmentation"
- **发表**: IJCNLP-AACL 2025
- **作者**: L.D.M.S. Sai Teja 等 (NIT Silchar)
- **关键思路**:
  1. 将文档级检测**转化为句子级序列标注任务**
  2. **Transformer + Neural Network + CRF** 三层架构
  3. Transformer 提取语义和句法模式 → NN 增强序列表示 → CRF 优化边界预测
- **性能**: 在两个公开 benchmark 上达到良好效果

#### (c) Token 级: 大规模跨语言 Token 分类

- **论文**: "Robust and Fine-Grained Detection of AI Generated Texts"
- **发表**: arXiv 2504.11952, 2025年4月 (ACL submission)
- **作者**: Ram Mohan Rao Kadiyala 等
- **核心创新**:
  1. **Token 分类**模型，训练在大规模人机共同撰写文本集合上
  2. 新数据集: **240万+** 混合文本，覆盖**23种语言**，涉及多个主流 LLM
  3. 对未见领域、未见生成器、非母语者文本和对抗输入均表现良好
- **评估维度**: 按领域、生成器、对抗方法、输入长度分别评估

#### (d) HACo-Det: 人机共著 Word 级检测

- **论文**: "HACo-Det: A Study Towards Fine-Grained Machine-Generated Text Detection under Human-AI Coauthoring"
- **发表**: arXiv 2506.02959, 2025年6月
- **作者**: Zhixiong Su 等 (西安交通大学)
- **核心创新**:
  1. **Word-level 归因标签**: 通过自动化管道生成人机共著文本并附带词级别标注
  2. 将 7 种文档级检测器**改造为词级检测器**
  3. **AI Ratio**: 提出用数值化 AI 比例来量化共著文本中的 AI 参与度
- **发现**: Metric-based 方法的细粒度检测 F1 仅 0.462；微调模型性能显著更好

---

## 2. 训练技巧

### 2.1 数据增强策略

| 方法 | 来源 | 关键技术 |
|------|------|---------|
| **回译 (Backtranslation)** | PAN CLEF 2025 参赛系统 | 翻译到其他语言再翻回来，增加文本多样性 |
| **同义词/反义词替换** | PAN CLEF 2025 | 针对少数类别的词汇替换扩增 |
| **随机删除** | PAN CLEF 2025 | 随机删除部分词语，模拟不完整文本 |
| **AutoFill-Refine** | SenDetEX (EMNLP 2025) | 高质量人机混合文本合成策略 |
| **自动化共著管道** | HACo-Det (2025) | 自动生成带词级标注的人机共写文本 |

### 2.2 对抗训练

#### (a) GREATER: 贪心对抗促进防御器 (ACL 2025)

- **论文**: "Iron Sharpens Iron: Defending Against Attacks in Machine-Generated Text Detection with Adversarial Training"
- **发表**: ACL 2025 (Long Paper)
- **作者**: Yuanfan Li 等 (西安交通大学/Queen Mary University of London)
- **核心创新**:
  1. **GREATER-A (Adversary)**: 在 Embedding 空间识别并扰动关键 token，结合贪心搜索和剪枝生成隐蔽且具破坏性的对抗样本
  2. **GREATER-D (Detector)**: 从 GREATER-A 的攻击中学习防御，并将防御能力泛化到其他攻击
  3. 从**威胁建模(Threat Modeling)**角度审视检测问题

#### (b) PIFE: 扰动不变特征工程

- **论文**: "Modeling the Attack: Detecting AI-Generated Text by Quantifying Adversarial Perturbations"
- **发表**: arXiv, 2025年10月
- **关键思路**:
  1. 多阶段标准化管道将输入文本转换为标准形式
  2. 利用 Levenshtein 距离和语义相似度量化转换幅度
  3. 将这些信号直接输入分类器
- **发现**: 传统对抗训练在语义攻击面前失效 (TPR@1%FPR 暴跌至 48.8%)，而 PIFE 模型显著更强

#### (c) 主动 vs 被动检测方法综述

- **论文**: "AI-Generated Text Detection: A Comprehensive Review of Active and Passive Approaches"
- **发表**: Computers, Materials and Continua, Vol. 86 No. 3, 2026年1月
- **分类体系**:
  - **被动方法 (Passive)**: 统计特征、神经网络分类器、零样本方法
  - **主动方法 (Active)**: 水印嵌入、可控生成
  - **对抗学习增强**: 专门章节讨论如何利用对抗学习提升检测鲁棒性

### 2.3 水印技术 (Watermarking)

#### (a) SynthID-Text (Google) 理论分析

- **论文**: "On Google's LLM Watermarking System: Theoretical Analysis and Empirical Validation"
- **发表**: ICLR 2026 投稿
- **核心内容**:
  - 首个对 Google SynthID-Text 的理论分析
  - SynthID-Text 的三大组件: Tournament 采样算法、基于 Bayesian/Mean score 的检测策略、统一的 distortionary/non-distortionary 水印框架
  - **发现**: Mean score 对 tournament 层数增加天然脆弱；设计了 Layer Inflation Attack

#### (b) 分布自适应水印框架 (NeurIPS 2025)

- **论文**: "Theoretically Grounded Framework for LLM Watermarking: A Distribution-Adaptive Approach"
- **发表**: NeurIPS 2025
- **核心创新**: 联合优化水印方案和检测器，推导闭合形式最优解，提出 distortion-free 分布自适应水印算法

#### (c) PRO: 开源 LLM 水印

- **论文**: "PRO: Enabling Precise and Robust Text Watermark for Open-Source LLMs"
- **发表**: arXiv, 2025年10月
- **核心挑战**: 开源 LLM 无法在解码阶段嵌入水印，需将水印蒸馏到模型权重中

### 2.4 半监督/自监督方法

- **DeTeCtive (NeurIPS 2024)** 的 TFIA 能力: 无需额外训练即可适应 OOD 数据，本质上是自监督式的增量适应
- **Binoculars / Lastde**: 完全零样本/无训练方法，依赖语言模型的内在统计特性

### 2.5 跨模型泛化

| 方法 | 泛化策略 | 效果 |
|------|---------|------|
| **DeTeCtive** | 多级对比学习 + TFIA | OOD 零样本大幅领先 |
| **Lastde/Lastde++** | Token 概率序列的时间序列分析 | 跨域、跨模型、跨语言均 SOTA |
| **HART 2D Method** | 内容与表达解耦 | RAID benchmark AUROC 0.886 |
| **GREATER** | 对抗训练增强泛化 | 对多种攻击方法具有防御能力 |
| **Token 级 240万数据集** | 大规模多模型多语言训练 | 23种语言、多种 LLM |

---

## 3. 中文文本检测

### 3.1 中文 AI 文本检测的特殊挑战

根据多篇论文和 EmergentMind 的主题综述，中文 AI 文本检测面临以下特殊挑战:

1. **分词 (Tokenization)**: 中文缺乏天然的词边界分隔符，分词质量直接影响检测效果
2. **多义性 (Polysemy)**: 中文单字/词的多义性更强，增加了检测复杂度
3. **风格差异 (Stylistic Variance)**: 中文具有更多的修辞手法和文体变化
4. **跨域挑战**: 学术文本、文学作品、新闻报道等领域的语言特征差异较大
5. **新型 LLM**: DeepSeek R1、Qwen 2.5、GLM-4 等中文 LLM 的快速迭代

### 3.2 代表性中文检测方法

#### (a) RoBERTa-Text 双流模型

- **来源**: 中国地质大学 (2025)
- **方法**: RoBERTa 语义编码 + 文本统计特征双流融合
- **数据**: HC3 + ChatGPT 检测数据集 + CNKI 学术摘要 + 文学作品
- **LLM 覆盖**: DeepSeek R1, Phi4, Qwen 2.5
- **性能**: Phi4 文本 Recall 100%

#### (b) 中科院/国科大的系列工作

- **SenDetEX** (EMNLP 2025): 中科院自动化所出品，句子级检测，对中文场景有很好的适用性
- **DETree** (NeurIPS 2025): 中科院计算所出品，树结构层次化表示学习

### 3.3 混合文本（人类+AI混写）检测

这是 2025 年最热门的研究方向之一:

| 方法 | 粒度 | 关键特色 | 与本项目关联度 |
|------|------|---------|-------------|
| **SenDetEX** | 句子级 | Style + Context 融合 | 高 - 可参考其混合文本合成策略 |
| **DETree** | 文档级(多类别) | 树结构层次分类 | 中 - 多种协作模式分类 |
| **HACo-Det** | 词级 | Word-level 归因 + AI Ratio | 高 - Word-level 边界检测 |
| **Sci-SpanDet** | Span 级 | 对比学习 + BIO-CRF | **极高** - 与本项目 [SEP] 机制最接近 |
| **PAN CLEF 2025 Subtask 2** | 文档级(6类) | 多类协作文本分类 | 中 - 评测框架参考 |

### 3.4 边界检测技术

**与本项目的 `[SEP]` 边界标记和 `bert_span_detector` 直接相关的最新进展:**

1. **BIO-CRF 序列标注**: Sci-SpanDet 和句子级检测工作均采用 BIO 标注 + CRF 的经典方案，但结合了 Transformer 的强语义表示
2. **指针式边界解码 (Pointer-based Boundary Decoding)**: Sci-SpanDet 提出的创新方法，直接预测边界位置
3. **Transformer + NN + CRF**: IJCNLP-AACL 2025 的方法，三层架构优化边界预测
4. **Token 分类 + 滑动窗口**: 大规模 Token 分类方法 (arXiv 2504.11952) 在 240万数据上训练

---

## 4. 知名工具与系统

### 4.1 商业系统

#### GPTZero

- **最新状态** (2026年1-2月): 自称"最准确的商业 AI 检测器"
- **Chicago Booth 2026 Benchmark**: 准确率约 **99%**，领先 Pangram 和 Originality.ai
- **新功能**:
  - Google Docs 集成 (Writing Replay 视频证据)
  - Advanced Insights (超越基本 AI 检查)
  - 剽窃检测器
  - 语法检查和写作反馈
- **支持检测的 LLM**: ChatGPT, GPT-5, Claude, Gemini
- **混合文本检测**: 支持 "AI + Human" 和 "Polished by AI" 场景

#### Originality.ai

- **自我评估**: 在 12 项第三方研究的 Meta 分析中被评为"最准确"
- **优势**: 在 6 项已发表的第三方研究中均被评为最有效工具
- **功能**: AI 检测 + 剽窃检测

#### 准确率对比 (独立测试)

| 工具 | 基本准确率 | 编辑/释义后准确率 | 来源 |
|------|----------|----------------|------|
| GPTZero | ~85% → **99%** (2026优化后) | <80% (编辑后下降) | GPTZero blog, AmpiFire |
| Originality.ai | ~76% (全样本) | 未公开 | AmpiFire 独立测试 |
| GPTZero (Chicago Booth) | **~99%** | - | Chicago Booth 2026 |

> **注意**: 各家的测试方法和数据集不同，准确率数字需谨慎对比。

### 4.2 学术界开源检测工具

| 工具 | 方法 | 特色 | GitHub |
|------|------|------|--------|
| **Binoculars** | 零样本交叉困惑度 | 无需训练，高精度 | github.com/ahans30/Binoculars |
| **DeTeCtive** | 多级对比学习 | OOD 泛化强 | github.com/heyongxin233/DeTeCtive |
| **DETree** | 树结构层次学习 | 混合文本检测 | github.com/heyongxin233/DETree |
| **Lastde** | Token 概率时序分析 | 无训练，ICLR 2025 | github.com/TrustMedia-zju/Lastde_Detector |
| **RAID benchmark** | 综合评测平台 | 600万+生成文本 | github.com/liamdugan/raid |
| **SenDetEX** | 句子级风格+上下文 | 混合文本专用 | github.com/TristoneJiang/SenDetEX |
| **HART/Truth-Mirror** | 2D 内容表达解耦 | 层次化风险检测 | github.com/baoguangsheng/truth-mirror |
| **T-Detect** | 统计检测 | MIT 开源 | github.com/ResearAI/T-Detect |
| **AI-Text-Detection-Tool** | RoBERTa + LIME | 全栈系统含 Chrome 扩展 | github.com/MichaelShpyl/AI-Text-Detection-Tool |

---

## 5. 数据集与评测

### 5.1 重要 Benchmark 数据集

#### (a) RAID (ACL 2024, 持续更新)

- **规模**: 超过 **1000万** 文档 (原始 600万+，持续扩展)
- **覆盖**: 11 个 LLM, 11 个领域, 4 种解码策略, **12 种对抗攻击**
- **特色**: 目前最大最全面的 AI 文本检测评测集
- **Leaderboard**: https://raid-bench.xyz
- **共享任务 (GenAI Content Detection Task 3)**: 9个团队23个检测器提交，多个参赛者在 5% FPR 下达到 **99%+ 准确率**

#### (b) GenAI Content Detection Task 1 (COLING 2025)

- **任务**: 英语和多语言机器生成文本检测 (AI vs. Human)
- **组织者**: Yuxia Wang, Artem Shelmanov, Preslav Nakov 等 (MBZUAI)
- **特色**: 多语言覆盖，系统性评估

#### (c) PAN CLEF 2025: Voight-Kampff

- **Subtask 1**: 二分类 (人类 vs AI)，含风格模仿和混淆攻击
  - 24 个检测器提交，最佳系统 mean score **0.99**
  - 挑战: LLM 被指令模仿特定人类作者风格、未知模型/攻击
- **Subtask 2**: 6 类人机协作文本分类
  - 评估指标: F1, C@1, AUC-ROC, FPR, FNR
  - Baseline: SVM, Compression, Binoculars (Subtask 1); RoBERTa (Subtask 2)

#### (d) SemEval-2026 Task 13: GenAI Code Detection

- **任务**: LLM 生成代码检测与归因 (非纯文本，但方法可借鉴)
- **三个子任务**:
  - Subtask A: 检测 (人类 vs AI 代码)
  - Subtask B: 归因 (确定生成该代码的 LLM)
  - Subtask C: 混合源分析

#### (e) 大规模综合数据集 (arXiv 2510.22874)

- **论文**: "A Comprehensive Dataset for Human vs. AI Generated Text Detection" (2025年10月)
- **特色**: 系统性构建的大规模检测数据集

#### (f) HACo-Det 数据集

- **粒度**: Word-level 归因标签
- **特色**: 自动化管道生成的人机共著文本

### 5.2 评测标准

| 指标 | 用途 | 关键阈值 |
|------|------|---------|
| **AUROC** | 整体检测能力 | >0.85 良好, >0.95 优秀 |
| **TPR@1%FPR** | 低误报条件下的真阳性率 | 关键工业指标 |
| **TPR@5%FPR** | RAID benchmark 标准 | >99% (多个系统达到) |
| **F1 Score** | 精确率和召回率的调和 | PAN CLEF 主要指标 |
| **C@1** | 考虑"不确定"判断的指标 | PAN 特色指标 |

### 5.3 综合性 Survey 论文

| 论文 | 发表 | 重点 |
|------|------|------|
| "A Survey on LLM-Generated Text Detection" | Computational Linguistics, 2025年3月 | 水印、统计、神经网络、人工辅助方法全面综述 |
| "AI-generated text detection: A comprehensive review" | Computer Science Review, Vol.58, 2025年11月 | 技术基础、方法论、评估框架、实际应用 |
| "Active and Passive Approaches Review" | CMC, Vol.86 No.3, 2026年1月 | 主动 vs 被动方法分类体系 |
| "Factors influencing detectability" | NRC Canada / JAIR, 2025年5月 | 影响检测能力的因素分析 |

---

## 6. 与本项目的关联分析

### 6.1 当前项目技术定位

本项目 (`datacollection`) 的核心技术:
- **分类器**: `bert_v2_with_sep` (98.71% 准确率)，使用 `[SEP]` 边界标记
- **边界检测器**: `bert_span_detector` (Token 准确率 96.69%)
- **核心创新**: `[SEP]` 边界标记机制

### 6.2 可借鉴的前沿技术

#### 优先级 P0 (强烈建议引入)

1. **对比学习增强**
   - 参考: DeTeCtive (NeurIPS 2024) 的多级对比学习
   - 方案: 在 BERT 微调时加入对比学习损失，学习区分不同写作风格
   - 预期收益: 提升 OOD 泛化能力和跨模型检测效果
   - **项目中已有**: `docs/plans/contrastive_learning_guide.md`

2. **BIO-CRF 边界检测升级**
   - 参考: Sci-SpanDet 的 BIO-CRF + 指针式边界解码
   - 方案: 将现有 `bert_span_detector` 升级为 Transformer + CRF 架构
   - 预期收益: 更精确的边界定位

3. **数据增强**: 回译 + 同义词替换 + 随机删除 (PAN CLEF 2025 验证有效)

#### 优先级 P1 (推荐尝试)

4. **上下文融合检测**
   - 参考: SenDetEX (EMNLP 2025) 的 Style + Context 融合
   - 方案: 在句子级检测时融入上下文窗口信息
   - 预期收益: 提升混合文本中的边界检测准确率

5. **树结构层次分类**
   - 参考: DETree (NeurIPS 2025) 的层次化协作分类
   - 方案: 将续写/改写/润色等类型建模为层次结构
   - 预期收益: 更精细的多分类效果

6. **2D 内容-表达解耦**
   - 参考: HART (西湖大学, 2025)
   - 方案: 将文本解耦为内容和表达两个维度进行独立检测
   - 预期收益: 提升对改写/润色文本的检测鲁棒性

#### 优先级 P2 (长期探索)

7. **对抗训练**: 参考 GREATER (ACL 2025) 的对抗框架
8. **水印检测**: 作为补充检测手段
9. **零样本检测集成**: 将 Binoculars/Lastde 作为 ensemble 成员

### 6.3 评测对标建议

1. 将模型在 **RAID benchmark** 上评测，获得可对比的标准化分数
2. 参考 **PAN CLEF 2025** 的评测指标体系 (F1, C@1, AUC-ROC)
3. 构建中文版的 **混合文本评测集**，参考 SenDetEX 的 AutoFill-Refine 策略

---

## 参考文献

### 顶会论文

1. Guo et al., "DeTeCtive: Detecting AI-generated Text via Multi-Level Contrastive Learning", NeurIPS 2024. [arXiv:2410.20964]
2. He et al., "DETree: DEtecting Human-AI Collaborative Texts via Tree-Structured Hierarchical Representation Learning", NeurIPS 2025. [arXiv:2510.17489]
3. Xu et al., "Training-free LLM-generated Text Detection by Mining Token Probability Sequences (Lastde)", ICLR 2025. [arXiv:2410.06072]
4. Hans et al., "Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text", ICML 2024. [arXiv:2401.12070]
5. Li et al., "Iron Sharpens Iron (GREATER): Defending Against Attacks in MGT Detection with Adversarial Training", ACL 2025.
6. Jiang et al., "SenDetEX: Sentence-Level AI-Generated Text Detection for Human-AI Hybrid Content via Style and Context Fusion", EMNLP 2025.
7. Bao et al., "Decoupling Content and Expression: Two-Dimensional Detection of AI-Generated Text (HART)", arXiv:2503.00258, 2025.
8. Kadiyala et al., "Robust and Fine-Grained Detection of AI Generated Texts", arXiv:2504.11952, 2025.
9. Su et al., "HACo-Det: Fine-Grained MGT Detection under Human-AI Coauthoring", arXiv:2506.02959, 2025.
10. Yin & Wang, "Span-level detection of AI-generated scientific text via contrastive learning and structural calibration (Sci-SpanDet)", Knowledge-Based Systems 334, 2026.

### Survey 论文

11. Wu et al., "A Survey on LLM-Generated Text Detection: Necessity, Methods, and Future Directions", Computational Linguistics, 2025.
12. Kehkashan et al., "AI-generated text detection: A comprehensive review", Computer Science Review 58, 2025.
13. Xiang et al., "AI-Generated Text Detection: A Comprehensive Review of Active and Passive Approaches", CMC 86(3), 2026.
14. Fraser et al., "Detecting AI-generated text: factors influencing detectability with current methods", NRC Canada / JAIR, 2025.

### 竞赛与 Benchmark

15. Dugan et al., "RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors", ACL 2024.
16. Bevendorff et al., "Overview of PAN 2025: Voight-Kampff Generative AI Detection", CLEF 2025.
17. Wang et al., "GenAI Content Detection Task 1: English and Multilingual Machine-Generated Text Detection", COLING 2025.
18. SemEval-2026 Task 13: GenAI Code Detection & Attribution.

### 中文方向

19. Su et al., "Research on AI-generated Chinese text detection method based on deep learning", BDIA 9:328-349, 2025.
20. Maktabdar Oghaz, "Detection and classification of ChatGPT-generated content using deep transformer models", Frontiers in AI 8:1458707, 2025.

### 水印技术

21. "Theoretically Grounded Framework for LLM Watermarking: A Distribution-Adaptive Approach", NeurIPS 2025.
22. "On Google's SynthID-Text: Theoretical Analysis and Empirical Validation", ICLR 2026 submission.
23. Xue et al., "PRO: Enabling Precise and Robust Text Watermark for Open-Source LLMs", arXiv:2510.23891, 2025.

---

*本报告基于 2026年2月10日的搜索结果整理，涵盖 2024-2026 年间的核心进展。*
