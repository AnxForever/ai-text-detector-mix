# 第2章 相关工作

> 本章综述与中文 AI 生成文本检测相关的研究工作。在检测方法的分类上，本文按"统计特征与零样本方法"、"预训练模型微调监督方法"、"细粒度与混合文本检测"、"对抗鲁棒性与水印方法"四条主线进行梳理，并专门总结中文场景下的研究现状与数据资源。落脚点上，本章通过对多条研究路线的优劣对比，阐明在中文工程落地场景下，**基于 BERT 微调的监督分类是最稳妥、最可复现、最适合实际部署的技术路线**，为本文方法的选型提供充分的文献依据。

---

## 2.1 研究综述与分类框架

AI 生成文本检测的目标是判断一段给定文本是由人类撰写还是由大语言模型（Large Language Model, LLM）生成。自 2022 年 ChatGPT 问世以来，该方向在短短两年内已发展出多条并行的研究路线。本章按"是否需要训练检测器"、"监督信号来源"、"检测粒度"三个维度，将现有方法归纳为四大类（如表 2-1 所示）。

**表2-1 AI 文本检测方法分类框架**

| 分类维度 | 主要路线 | 代表工作 | 本文立场 |
|---------|---------|---------|---------|
| 统计特征与零样本 | 无需训练，依赖参考模型 | DetectGPT, Binoculars, Token Prob Seq | 外部强基线 |
| 预训练模型微调 | 监督分类（BERT/RoBERTa）| RoBERTa 微调, DeTeCtive | **本文主线** |
| 细粒度与混合文本 | 句级、词级、Token/Span 级 | SenDetEX, HACo-Det, DETree | 本文扩展能力 |
| 对抗鲁棒性与水印 | 攻防视角、主动检测 | Iron Sharpens Iron, DDR | 鲁棒性评测参考 |

本章 2.2–2.5 节依次介绍四条路线的代表工作，2.6 节专门总结中文场景下的研究现状与可用数据资源，2.7 节在前文基础上给出本文的技术路线选择依据。

---

## 2.2 基于统计特征与零样本的检测方法

这一路线的核心假设是：LLM 生成的文本与人类撰写的文本在**可观测统计特征**上存在差异，无需训练专门的检测器，仅通过统计量比较即可完成判别。该路线的最大优势是**无需训练、跨模型泛化能力强**，但通常对短文本与轻度改写较为敏感。

### 2.2.1 早期统计特征：困惑度与突发度

早期方法主要基于语言模型的困惑度（Perplexity, PPL）：

$$\text{PPL}(x) = \exp\left(-\frac{1}{n}\sum_{i=1}^{n} \log P(x_i \mid x_{<i})\right)$$

其中 $P(x_i \mid x_{<i})$ 由参考语言模型给出。由于 LLM 生成文本更符合语言模型自身的概率分布，其困惑度通常显著低于人类文本。与之配对的指标还有**词汇多样性**（Type-Token Ratio, TTR）与**突发度**（Burstiness）——人类写作往往表现出更强的词汇使用不均匀性（某些词集中出现）。

这类方法的局限在于：（1）短文本统计不稳定；（2）阈值在不同领域上难以统一；（3）容易被人工轻微改写绕过。因此近年来的研究逐渐转向更稳健的零样本方法。

### 2.2.2 DetectGPT：基于概率曲率的零样本检测

Mitchell 等（2023）在 ICML 提出 **DetectGPT**（L? 参考 Mitchell et al., 2023），核心思想是利用 LLM 输出文本的"概率局部极值"性质：

> **关键观察**：LLM 生成的文本位于其自身概率分布的**局部最大值**附近——对该文本进行微小扰动（synonym replacement、段落重写），生成概率会显著下降；相比之下，人类文本不具有这种对称的"概率峰"。

DetectGPT 通过对输入文本 $x$ 生成 $M$ 个扰动版本 $\{\tilde{x}^{(m)}\}_{m=1}^{M}$，计算扰动前后的对数概率差：

$$\Delta(x) = \log P(x) - \frac{1}{M}\sum_{m=1}^{M} \log P(\tilde{x}^{(m)})$$

当 $\Delta(x)$ 显著大于 0 时判定为 AI 生成文本。该方法在英文基准上取得了较强的零样本泛化能力，但对参考模型（扰动生成器）的选择较为敏感。

### 2.2.3 Binoculars：对照模型困惑度比

Hans 等（2024）在 ICML 提出 **Binoculars**（Hans et al., 2024），这是一种基于"双模型困惑度比"的零样本检测方法。Binoculars 的核心公式是：

$$B(x) = \frac{\text{PPL}_{\text{observer}}(x)}{\text{Cross-PPL}_{\text{observer}, \text{performer}}(x)}$$

其中 `observer` 与 `performer` 是两个不同的参考 LLM。该方法在英文多模型基准上能够在极低的误报率（FPR < 1%）下达到较高的真阳率（TPR），对不同 LLM 之间的泛化表现稳定。Binoculars 的局限在于需要两个参考模型，计算成本较高；在中文场景下需要寻找合适的观察者 / 执行者组合。

### 2.2.4 Token 概率序列挖掘

Xu 等（2025）在 ICLR Poster 提出 **Training-free LLM-generated Text Detection by Mining Token Probability Sequences**（Xu et al., 2025），进一步将检测依据从单一统计量扩展到 Token 级概率序列的时序模式。该方法的优势在于对跨域跨模型场景有较强鲁棒性，但需要能够获取 Token 概率序列，对商业 API 场景（仅返回文本）不适用。

### 2.2.5 零样本方法的共性局限

统计与零样本方法的共性缺陷是：（1）**依赖参考模型质量**——参考模型与待检模型越相近，检测性能越高；（2）**对短文本与结构化文本失效**——短于 100 字的文本统计量不稳定；（3）**对对抗改写敏感**——简单的同义词替换即可显著降低检测置信度。因此这类方法通常作为对照基线使用，在生产部署中较少作为主方案。

---

## 2.3 基于预训练模型微调的监督检测方法

这一路线将 AI 文本检测建模为标准的二分类问题，通过在标注数据上微调预训练语言模型（如 BERT、RoBERTa）得到专用分类器。相比零样本方法，监督方法的主要优势是**在训练分布内的准确率更高、推理速度更快、工程落地更成熟**；主要挑战是跨模型泛化能力与数据质量依赖性。

### 2.3.1 BERT / RoBERTa 监督分类

OpenAI 早期发布的 AI Text Classifier 即基于 RoBERTa 微调，在 2023 年因准确率不足主动下线。在学术界，许多工作证明通过高质量数据治理，BERT/RoBERTa 微调仍可达到 98% 以上的准确率。BERT 相对 RoBERTa 的优势在于：（1）公开的中文预训练模型（`bert-base-chinese`）生态更成熟；（2）在判别任务上二者性能差距不显著；（3）部署成本更可控。这也是本文选择 BERT 作为基础模型的主要依据（详见第 4.3.1 节）。

### 2.3.2 DeTeCtive：多级对比学习

Guo 等（2024）在 NeurIPS Poster 提出 **DeTeCtive**（Guo et al., 2024），将对比学习引入 AI 文本检测。DeTeCtive 在 BERT 编码器之上同时进行 Token 级、句子级、文档级三个层次的对比学习，使模型学习到人类与 AI 文本的多粒度风格差异。该方法在跨域零样本评估上优于普通微调。其局限是训练与检索管道较复杂，工程落地难度较高。

### 2.3.3 DistilBERT 轻量化方法

Khan 等（2025）在 *Scientific Reports* 发表 **Identifying AI-generated content using the DistilBERT transformer**（Khan et al., 2025），展示了使用蒸馏后的轻量模型（DistilBERT）结合统计特征也能取得可观的检测性能。这一路线对资源受限的部署场景有参考价值，但在中文场景下的泛化性能仍需验证。

### 2.3.4 指令微调检测器

2024 年以来，一些工作尝试用指令微调框架（Instruction Tuning）构建检测器，将"判断文本是否 AI 生成"视为一个指令任务（Li et al., 2024）。这类方法在 OOD 数据上通常优于传统分类器，但推理成本显著高于 BERT 分类器，难以满足在线检测的实时性要求。

### 2.3.5 预训练微调路线的共性优势

监督微调路线的核心优势包括：（1）**训练分布内准确率高**——可达 98%+；（2）**推理成本低**——BERT-base 在单卡 GPU 上可达 100+ 样本/秒；（3）**数据治理空间大**——通过数据清洗、弱域补充等 Data-Centric 手段可以系统性提升性能；（4）**中文生态成熟**——`bert-base-chinese` 等模型工程落地验证充分。这些优势使其成为本文方法的技术主线（详见第 4 章）。

---

## 2.4 细粒度检测与混合文本分析

随着"人类撰写开头 + AI 续写尾部"等混合写作形式日益普及，研究焦点从文档级二分类逐渐扩展到**细粒度检测**——句级、词级、Token/Span 级，以及层次化的人机协作建模。

### 2.4.1 SenDetEX：句子级检测

Jiang 等（2025）在 EMNLP 发表 **SenDetEX: Sentence-Level AI-Generated Text Detection for Human-AI Hybrid Content via Style and Context Fusion**（Jiang et al., 2025）。该方法对混合文本中的每个句子独立判定 Human/AI 标签，通过风格与上下文融合建模句子级特征，在 S-AGTD Benchmark 上取得了当时最佳的句级检测性能。SenDetEX 的关键贡献是证明了"句子级判别"的可行性——每个句子具有足够的语义信息支撑独立决策。

### 2.4.2 HACo-Det：词级细粒度归因

Su 等（2025）在 arXiv 发布 **HACo-Det: A Study Towards Fine-Grained Machine-Generated Text Detection under Human-AI Coauthoring**（Su et al., 2025），进一步将粒度推至词级。该工作提出了 "AI Ratio" 指标，量化一段文本中 AI 生成片段所占比例。HACo-Det 的意义在于明确了"人机共著"场景的正式研究问题定义，为后续细粒度检测研究提供了基础。

### 2.4.3 Token / Span 级检测与 Sci-SpanDet

Kadiyala 等（2025）在 arXiv 发表 **Robust and Fine-Grained Detection of AI-Generated Texts**（Kadiyala et al., 2025），将边界检测建模为多语 Token 分类任务，支持 23 种语言。Yin 与 Wang（2026）在 *Knowledge-Based Systems* 发表 **Sci-SpanDet: Span-level detection of AI-generated scientific text via contrastive learning and structural calibration**（Yin & Wang, 2026），专门针对学术文本的 Span 级检测。二者共同验证了**序列标注范式（NER-like）在 AI 文本边界检测中的有效性**——这也是本文第 4.7.2 节 Token 级边界检测器的直接理论依据。

### 2.4.4 DETree：协作文本的层次表示学习

He 等（2025）在 NeurIPS Poster 发表 **DETree: Detecting Human-AI Collaborative Texts via Tree-Structured Hierarchical Representation Learning**（He et al., 2025），提出树状层次结构建模人机协作文本。该方法在"未见过的生成器"场景泛化能力优于扁平分类器，但工程复杂度较高。

### 2.4.5 细粒度检测的研究趋势

综合 2.4.1–2.4.4 节的进展可以观察到：**研究焦点已从"文档级判别"向"边界定位与混合文本分析"迁移**。这一趋势与实际使用场景高度吻合——单纯的"全篇 AI"或"全篇 Human"场景正在减少，混合书写反而成为主流。本文在文档级分类主线基础上扩展边界检测能力（第 4.7 节），正是顺应这一趋势。

---

## 2.5 对抗鲁棒性与水印检测

除"检测方法本身"之外，近年来对**检测器的鲁棒性与对抗稳定性**研究也迅速升温。这一方向的核心问题是：检测器在面对改写攻击、格式扰动、对抗样本时是否仍保持性能？

### 2.5.1 对抗训练：Iron Sharpens Iron

Li 等（2025）在 ACL 发表 **Iron Sharpens Iron: Defending Against Attacks in Machine-Generated Text Detection with Adversarial Training**（Li et al., 2025），首次在 AI 文本检测场景系统性引入对抗训练。该工作提出的 GREATER 评测协议对同义词替换、Prompt 注入、格式扰动等多类攻击进行统一度量，证明对抗训练能显著提升检测器在攻击场景下的稳定性。

### 2.5.2 Decoupling：二维解耦建模

Bao 等（2025）在 arXiv 发布 **Decoupling Content and Expression: Two-Dimensional Detection of AI-Generated Text**（Bao et al., 2025），将文本的"内容"与"表达"进行解耦建模。该方法在 RAID 基准上的 AUROC 显著提升，提供了另一个思路——通过分离检测信号的来源，提升对多样化攻击的鲁棒性。

### 2.5.3 水印方法概述

水印方法（Watermarking）从"模型生成时主动标记"角度提供另一种检测路径。分为**软水印**（调整词汇分布的统计特征）与**硬水印**（在特定位置插入可识别的标记）。这一路线的根本局限是**需要模型提供方的配合**——对于已经生成的历史文本、开源模型输出、以及后处理改写场景，水印方法通常失效。因此本文不采用水印路线，但将其作为相关工作进行介绍。

### 2.5.4 本文方法对鲁棒性的回应

本文通过第 5 章的格式对抗测试（最大性能下降 0.05%）与独立评估集泛化测试（independent_data 准确率 98.57%）对检测器的鲁棒性进行量化评估，保证本文方法的"高准确率"不是格式偏差造成的虚高，而是具备跨域稳定性的真实性能。

---

## 2.6 中文 AI 文本检测研究现状

上述 2.2–2.5 节介绍的方法多数以英文为主。本节专门聚焦中文场景下的研究现状与数据资源。

### 2.6.1 中文数据资源

中文 AI 文本检测领域目前可用的公开资源如表 2-2 所示。

**表2-2 中文 AI 文本检测公开数据资源**

| 资源 | 来源 | 规模 | 特点 | 本文使用 |
|-----|------|-----|------|--------|
| HC3-Chinese | Guo et al., 2023 | 24K+ | ChatGPT 问答对照 | ✓ 核心来源 |
| THUCNews | 清华 NLP, 2006 | 740K+ | 新闻语料（仅 Human）| ✓ Human 补充 |
| NLPCC AI 检测任务 | NLPCC 2025 | 未公开 | 共享任务官方数据 | 待申请 |
| GenAIDetect | COLING Workshop | 未公开 | 英文为主 | 未使用 |
| PAN/CLEF 2025 | CLEF 2025 | 多任务 | 英文为主，强基线 | 未使用 |
| MGTBench | GitHub | 50K+ | 多模型多语言 | 未使用 |

本文在 HC3-Chinese 与 THUCNews 基础上，自行扩展了 46 个 LLM 的中文生成数据（详见第 3.2.3 节），形成覆盖 8 大模型家族的多源中文数据集。

### 2.6.2 国内代表性工作

Su 等（2025）在 *Big Data and Information Analytics* 发表 **Research on AI-generated Chinese text detection method based on deep learning**（Su et al., 2025），提出基于 RoBERTa 的双流融合框架，在 HC3-Chinese 与自建多域数据上取得了较高的多模型召回率。该工作是国内中文 AI 文本检测的代表性研究，但代码未公开，可复现性受限。

### 2.6.3 中文研究的共性问题

当前中文场景研究存在以下共性问题：

（1）**数据开放度不足**：NLPCC 官方数据需申请、国内硕博学位论文研究成果的数据与代码开放程度较低；

（2）**评测协议不统一**：不同工作采用的评估集、指标、基线差异较大，论文间直接对比困难；

（3）**跨模型泛化证据薄弱**：多数工作仅报告单一 LLM（通常为 ChatGPT）下的指标，对新型 LLM 的适配性缺乏证据；

（4）**混合文本场景覆盖少**：公开的中文混合文本检测数据集几近缺失。

本文工作针对上述问题进行了系统性回应：（1）开源完整的 V11c 训练数据治理流程；（2）统一 2,599 条无泄露评估集口径；（3）在 independent_data 子集中专门纳入 GPT-4/GPT-5/Gemini-3/LLaMA-405B 等前沿模型输出；（4）在方法章节设计了 [SEP] 边界机制并在第 5.2.3 节展示混合文本检测性能。

### 2.6.4 综述类工作

Computers Materials and Continua 期刊 2026 年刊发了一篇综述 **AI-Generated Text Detection: A Comprehensive Review of Active and Passive Approaches**，系统梳理了主动（水印）与被动（检测）两大路线。该综述为本文研究提供了分类框架参考，但其内容仍以英文场景为主，中文场景覆盖有限。

---

## 2.7 本章小结

本章从四条主线系统梳理了 AI 生成文本检测领域的研究进展：

（1）**基于统计特征与零样本的方法**（DetectGPT、Binoculars、Token Probability Sequence）在跨模型泛化上具有优势，但对参考模型依赖度高、对短文本与改写攻击敏感，通常作为外部基线；

（2）**基于预训练模型微调的监督方法**（BERT / RoBERTa / DeTeCtive）在训练分布内性能最高、推理最快、工程最成熟，是当前工业落地的主流路线；

（3）**细粒度检测与混合文本分析**（SenDetEX、HACo-Det、Token/Span 级方法、DETree）代表了研究前沿，与本文的"分类 + 边界定位"扩展架构高度契合；

（4）**对抗鲁棒性与水印方法**提供了攻防视角的检测思路，本文通过第 5 章的格式对抗测试与独立评估集泛化测试对鲁棒性进行了量化回应。

**本文的技术路线选择依据**：综合考虑检测准确率、推理成本、中文生态成熟度、数据治理可行性与工程落地可复现性，本文选择**基于 `bert-base-chinese` 微调的监督分类**作为主线技术路线，在此基础上扩展 [SEP] 边界机制与 Token 级边界检测器以支持混合文本分析。这一选择的关键理由是：在中文 AI 文本检测的工程落地场景下，BERT 微调路线能够在"准确率 / 泛化能力 / 推理成本 / 数据治理空间 / 可复现性"五个维度之间取得最佳平衡。

本章所综述的文献为本文后续章节提供了理论依据与方法参照：第 3 章（数据集构建）参考了 HC3-Chinese（Guo et al., 2023）作为核心公开来源；第 4 章（方法设计）沿用了 Devlin 等（2019）的 BERT 架构、Müller 等（2019）的标签平滑、Geirhos 等（2020）的捷径学习防护、Guo 等（2017）的 Temperature Scaling 等理论基础；第 5 章（实验与分析）参考了 Iron Sharpens Iron（Li et al., 2025）的对抗鲁棒性评测思路设计了格式对抗测试。

---

## 参考文献（第 2 章引用）

1. Mitchell, E., Lee, Y., Khazatsky, A., Manning, C. D., & Finn, C. (2023). DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature. *ICML*, 24950–24962.
2. Hans, A., et al. (2024). Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text. *ICML*.
3. Xu, Y., et al. (2025). Training-free LLM-generated Text Detection by Mining Token Probability Sequences. *ICLR Poster*.
4. Guo, X., et al. (2024). DeTeCtive: Detecting AI-generated Text via Multi-Level Contrastive Learning. *NeurIPS Poster*.
5. Khan, H. U., et al. (2025). Identifying AI-generated content using the DistilBERT transformer and NLP techniques. *Scientific Reports*.
6. Jiang, L., et al. (2025). SenDetEX: Sentence-Level AI-Generated Text Detection for Human-AI Hybrid Content via Style and Context Fusion. *EMNLP*.
7. Su, Z., et al. (2025). HACo-Det: A Study Towards Fine-Grained Machine-Generated Text Detection under Human-AI Coauthoring. *arXiv:2506.02959*.
8. Kadiyala, R. M. R., et al. (2025). Robust and Fine-Grained Detection of AI-Generated Texts. *arXiv:2504.11952*.
9. Yin, Z., & Wang, S. (2026). Sci-SpanDet: Span-level detection of AI-generated scientific text via contrastive learning and structural calibration. *Knowledge-Based Systems*.
10. He, Y., et al. (2025). DETree: Detecting Human-AI Collaborative Texts via Tree-Structured Hierarchical Representation Learning. *NeurIPS Poster*.
11. Li, Y., et al. (2025). Iron Sharpens Iron: Defending Against Attacks in Machine-Generated Text Detection with Adversarial Training. *ACL Long Paper*.
12. Bao, G., et al. (2025). Decoupling Content and Expression: Two-Dimensional Detection of AI-Generated Text. *arXiv:2503.00258*.
13. Su, C., et al. (2025). Research on AI-generated Chinese text detection method based on deep learning. *Big Data and Information Analytics*.
14. Guo, B., Zhang, X., Wang, Z., Jiang, M., Nie, J., Ding, Y., Yue, J., & Wu, Y. (2023). How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection. *arXiv:2301.07597*.

> 📋 注：第 2 章仅列出本章首次引用的文献；Devlin et al. (2019)、Müller et al. (2019)、Geirhos et al. (2020)、Guo et al. (2017)、Stamatatos (2009)、Viola & Jones (2001)、He & Garcia (2009)、Ng (2021)、Northcutt et al. (2021) 等已在第 3、4 章参考文献中列出，此处不重复。

---

*最后更新: 2026-04-19*
*文献元数据: docs/thesis/literature_matrix_v1.csv（L001–L020 已确认，L021–L024 学位论文待下载）*
*文献综述: docs/thesis/literature_review_v1.md（V1 初稿，已提交导师审阅）*
