# 论文摘要（中英文版本）

> 最后更新：2026-04-19，对齐 V11c 训练集（63,113 条）与 2,599 条无泄露评估集加权平均口径。

---

## 中文摘要

**题目**：基于 BERT 微调的中文 AI 生成文本检测方法研究

**摘要**：

随着 ChatGPT、GPT-4、Claude、DeepSeek、Gemini、LLaMA 等大语言模型（LLM）的快速发展，AI 生成的中文文本在质量上已接近甚至达到人类水平，这给学术诚信、内容审核和知识产权保护带来了严峻挑战。现有中文 AI 文本检测研究存在数据开放度不足、模型多样性不够、混合文本场景覆盖少、工程可复现性弱等问题。本文围绕中文 AI 生成文本的二分类检测任务，从数据、方法、实验、工程四个维度开展了系统性研究工作。

首先，在数据层面，本文以 HC3-Chinese 为核心公开来源，结合 THUCNews、Wikipedia_CN、M4、VCSUM 等多源 Human 文本，并系统性生成覆盖 GPT、Claude、DeepSeek、Gemini、LLaMA、Qwen、Kimi、GLM 共 8 大 LLM 家族、46 个具体模型的 AI 文本，构建了总规模 63,113 条的训练集，并设计了 core_v1_test_clean（545 条）、independent_data（910 条）、merged_v2_val_clean（1,144 条）三个独立无泄露评估子集共 2,599 条样本。在此基础上，本文进一步设计并实施了 V10→V11a→V11b→V11c 四阶段 Data-Centric AI 数据治理流水线，在模型结构与超参完全固定的前提下，通过标签噪声清洗、弱域增补、长文边界修复三步治理，将独立评估集准确率从 97.69% 提升至 98.57%，总错误数降低 38%。

其次，在方法层面，本文提出了基于 `bert-base-chinese` 微调的文档级二分类方法。为应对中文 AI 文本检测中的长度偏差与灰色地带问题，本文提出长度感知标签平滑损失（Length-Aware Label Smoothing Loss），融合 Müller 等（2019）的标签平滑与 Geirhos 等（2020）的捷径学习防护理论；配合双维度加权采样（标签 × 长度桶）、AdamW 优化器、线性预热线性衰减学习率调度、Early Stopping、梯度裁剪等正则化手段，构成了完整的训练流水线。针对模型概率校准问题，采用 Guo 等（2017）的 Temperature Scaling 进行后置校准，在独立评估集上优化得到 T*=0.8165，期望校准误差 ECE 从 0.0168 降至 0.0034。在主二分类任务基础上，本文进一步扩展了以 [SEP] 边界标记机制（NSP + Stamatatos 2009 文体学理论）、Token 级边界检测器（序列标注范式）、双层级联架构（Viola & Jones, 2001）为核心的混合文本分析能力。

实验结果表明，本文方法在 2,599 条无泄露评估集上取得 98.69% 的加权平均准确率、97.75% 的 F1 分数、99.28% 的召回率，在 FastText / TextCNN / DPCNN / BERT-BiGRU / 本文方法 5 类对比方法中召回率居首且为唯一突破 99% 的方案。长度鲁棒性测试中，模型在 300–3,000 字各区间均达到 100% 准确率，性能方差为 0.0；格式对抗测试最大下降仅 0.05%，证明模型学习的是真实语义特征而非长度或格式偏差。混合文本检测中，引入 [SEP] 边界标记后 C2 类续写检测率从 79.82% 提升至 93.84%（+14.02 个百分点），边界定位误差稳定在 0–8 字符。部署效率方面，模型在单卡 RTX 5060 Laptop GPU 上达到 127.4 样本/秒的推理吞吐，GPU 峰值显存 672 MB，已在阿里云 ECS 生产环境稳定运行。

本文的研究成果为中文 AI 文本检测提供了一套完整的"数据治理 × 理论驱动 × 工程落地"方法论参考，具有重要的理论价值与实际应用意义。

**关键词**：AI 生成文本检测；大语言模型；BERT 微调；数据中心 AI；Temperature Scaling；混合文本边界定位；中文自然语言处理

---

## English Abstract

**Title**: Research on Chinese AI-Generated Text Detection via BERT Fine-tuning

**Abstract**:

With the rapid advancement of large language models (LLMs) such as ChatGPT, GPT-4, Claude, DeepSeek, Gemini, and LLaMA, AI-generated Chinese text has approached or even matched human-level quality, posing serious challenges to academic integrity, content moderation, and intellectual property protection. Existing research on Chinese AI text detection suffers from limited data availability, insufficient model diversity, inadequate coverage of mixed-text scenarios, and weak engineering reproducibility. This thesis conducts systematic research on the binary classification of Chinese AI-generated text from four dimensions: data, methodology, experimentation, and engineering deployment.

First, at the data level, we construct a 63,113-sample training set using HC3-Chinese as the core public source, supplemented by multi-source Human texts from THUCNews, Wikipedia_CN, M4, and VCSUM, and systematically generated AI texts covering 46 specific models across 8 LLM families (GPT / Claude / DeepSeek / Gemini / LLaMA / Qwen / Kimi / GLM). We also design three independent leakage-free evaluation subsets: core_v1_test_clean (545 samples), independent_data (910 samples), and merged_v2_val_clean (1,144 samples), totaling 2,599 samples. Building on this, we design and implement a V10→V11a→V11b→V11c four-stage Data-Centric AI governance pipeline, which—while keeping the model architecture and hyperparameters completely fixed—improves the independent-evaluation accuracy from 97.69% to 98.57% through label-noise removal, weak-domain supplementation, and long-text boundary repair, reducing total errors by 38%.

Second, at the methodology level, we propose a document-level binary classifier based on `bert-base-chinese` fine-tuning. To address the length bias and "gray zone" problems in Chinese AI-text detection, we propose a Length-Aware Label Smoothing Loss that integrates Müller et al.'s (2019) label smoothing and Geirhos et al.'s (2020) shortcut-learning defense theory, combined with two-dimensional weighted sampling (label × length bucket), AdamW optimizer, linear warm-up with linear decay scheduling, Early Stopping, and gradient clipping, forming a complete training pipeline. For probability calibration, we adopt Guo et al.'s (2017) Temperature Scaling with T*=0.8165, reducing the Expected Calibration Error (ECE) from 0.0168 to 0.0034 on the independent evaluation set. On top of the primary binary-classification task, we extend the method with a [SEP] boundary-marking mechanism (based on NSP pre-training and Stamatatos's 2009 stylometric theory), a Token-level boundary detector (sequence-labeling paradigm), and a two-layer cascade architecture (Viola & Jones, 2001) for mixed-text analysis.

Experimental results show that our method achieves 98.69% weighted-average accuracy, 97.75% F1, and 99.28% recall on the 2,599-sample leakage-free evaluation set, with recall ranking first and being the only solution exceeding 99% among the five compared methods (FastText / TextCNN / DPCNN / BERT-BiGRU / ours). In length-robustness tests, the model achieves 100% accuracy across all length intervals from 300 to 3,000 characters with zero performance variance. In format-adversarial tests, the maximum performance drop is only 0.05%, demonstrating that the model learns genuine semantic features rather than length or format biases. In mixed-text detection, introducing the [SEP] boundary marker increases the C2-type continuation detection rate from 79.82% to 93.84% (+14.02 percentage points), with boundary localization errors stably within 0–8 characters. In deployment efficiency, the model achieves 127.4 samples per second inference throughput on a single RTX 5060 Laptop GPU with 672 MB peak GPU memory, and has been stably deployed on Aliyun ECS in a production environment.

The research results of this thesis provide a complete "Data Governance × Theory-Driven × Engineering Deployment" methodology reference for Chinese AI text detection, with significant theoretical value and practical application significance.

**Keywords**: AI-generated text detection; Large language models; BERT fine-tuning; Data-Centric AI; Temperature Scaling; Mixed-text boundary localization; Chinese natural language processing

---

## 摘要写作要点（供后续润色参考）

### 1. 结构与要素

本摘要按以下结构组织，总字数约 800 字（中文）/ 800 词（英文）：

```
研究背景与问题 (1-2 段)
  ↓
数据工作 (1 段，含 V11c 治理流水线)
  ↓
方法工作 (1-2 段，含 LA-LSL / Temperature Scaling / [SEP] 扩展)
  ↓
实验结果 (1 段，关键数字)
  ↓
研究意义总结 (1-2 句)
```

### 2. 关键数据清单（确保与正文一致）

| 数据项 | 数值 | 用于 |
|-------|------|------|
| 训练集规模 | **63,113** 条 | 数据集描述 |
| 评估集规模 | **2,599** 条（三子集）| 数据集描述 |
| LLM 家族数 | **8** 大家族 / 46 模型 | 数据集描述 |
| 加权平均准确率 | **98.69%** | 实验结果 |
| F1 分数 | **97.75%** | 实验结果 |
| 召回率 | **99.28%** | 实验结果 |
| Temperature T* | **0.8165** | 方法 |
| ECE 校准前 | **0.0168** | 方法 |
| ECE 校准后 | **0.0034** | 方法（重点）|
| V10 独立评估准确率 | 97.69% | 数据治理 |
| V11c 独立评估准确率 | **98.57%** | 数据治理（重点）|
| 总错误降幅 | **38%** | 数据治理 |
| C2 检测率提升 | 79.82% → **93.84%** (+14.02%) | 混合文本（创新点）|
| 推理吞吐 | **127.4** 样本/秒 | 工程 |
| GPU 峰值显存 | **672 MB** | 工程 |
| 长度方差 | **0.0** | 鲁棒性 |
| 格式对抗最大下降 | **0.05%** | 鲁棒性 |

### 3. 关键词选择

**中文关键词（7 个）**：
- AI 生成文本检测（主题）
- 大语言模型（背景）
- BERT 微调（方法）
- 数据中心 AI（创新点）
- Temperature Scaling（方法）
- 混合文本边界定位（扩展）
- 中文自然语言处理（领域）

**English Keywords（7 项）**：
- AI-generated text detection
- Large language models
- BERT fine-tuning
- Data-Centric AI
- Temperature Scaling
- Mixed-text boundary localization
- Chinese natural language processing

### 4. 口径对齐说明

本摘要使用的全部数据均与以下章节一致：

- 第 3 章 §3.5.1 数据集基本统计
- 第 4 章 §4.6 Temperature Scaling 校准效果
- 第 4 章 §4.7.1 [SEP] 边界标记机制
- 第 5 章 §5.2.1 表 5-5 主要实验结果
- 第 5 章 §5.5 效率分析

---

*最后更新: 2026-04-19*
*口径来源: 第 3 章（数据集）、第 4 章（方法）、第 5 章（实验）*
*文件位置: docs/thesis/abstract_template.md*
