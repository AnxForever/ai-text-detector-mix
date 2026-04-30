# Defense KB v2

> 答辩知识库 v2 — 单一权威来源
> 整合自 `DEFENSE_KB_CURATED.md`（13 章）+ `ADVISOR_ACADEMIC_QA.md`（60 章）
> 口径优先级：本文件 > `DEFENSE_CURRENT_STATUS.md` > 模型目录下评估与训练日志
> 格式：每章 = 问题 → 一句话结论 → 2-4 条证据 → 来源路径

---

## A. 项目定位

### A-1. 研究问题

```yaml
---
chapter_id: A-1
title: "研究问题"
tags: [研究目标, 系统设计, 闭环]
evidence_paths: [docs/project/DEFENSE_CURRENT_STATUS.md, docs/project/ADVISOR_ACADEMIC_QA.md, docs/thesis/project_technical_deep_dive.md]
priority: high
aliases: ["核心问题是什么", "你到底要解决什么"]
---
```

**Problem**: 本项目的核心研究问题是什么？

**Conclusion**: 构建面向中文场景的 AI 生成文本检测系统，围绕数据构建、方法设计、评估协议、边界检测与部署演示形成完整研究闭环，而非单一分类器训练。

**Evidence**:
1. 数据层：构建覆盖多模型、多题材、多长度区间的中文训练集与独立评估集
2. 方法层：在 `bert-base-chinese` 微调框架上提升准确率、召回率、概率校准与长度鲁棒性
3. 扩展层：从"整篇是否为 AI"扩展到"哪一段是 AI 生成"的边界定位
4. 工程层：部署为 FastAPI + Next.js 可演示系统，支持在线检测与项目问答

**Source**: `docs/project/DEFENSE_CURRENT_STATUS.md`, `docs/project/ADVISOR_ACADEMIC_QA.md`, `docs/thesis/project_technical_deep_dive.md`

---

### A-2. 目标方法

```yaml
---
chapter_id: A-2
title: "目标方法"
tags: [方法设计, BERT, 双层架构]
evidence_paths: [docs/thesis/theoretical_foundations.md, docs/thesis/project_technical_deep_dive.md, docs/thesis/chapter5_experiments_filled.md]
priority: high
aliases: ["你用了什么方法", "怎么做的"]
---
```

**Problem**: 本项目采用什么方法实现 AI 文本检测？

**Conclusion**: 以 `bert-base-chinese` 微调为主线，结合数据中心治理、Temperature Scaling 校准、`[SEP]` 边界标记与 Token 级边界检测器，形成文档级分类 + 细粒度边界定位的双层检测架构。

**Evidence**:
1. 文档级分类器：基于 `BertForSequenceClassification`，判断整段文本属于 Human / AI / mixed
2. 边界检测器：基于 `BertForTokenClassification`，在混合文本中定位风格切换位置
3. 两者级联互补：先由分类器完成整体判别，再由边界检测器完成细粒度分析

**Source**: `docs/thesis/theoretical_foundations.md`, `docs/thesis/project_technical_deep_dive.md`, `docs/thesis/chapter5_experiments_filled.md`

---

### A-3. 应用场景

```yaml
---
chapter_id: A-3
title: "应用场景"
tags: [应用场景, 工程价值, 部署]
evidence_paths: [docs/project/ADVISOR_ACADEMIC_QA.md, docs/project/DEFENSE_CURRENT_STATUS.md, api/api.py]
priority: medium
 aliases: ["能用在哪里", "有什么用"]
---
```

**Problem**: 该系统的实际应用场景和工程价值是什么？

**Conclusion**: 系统可用于学术诚信辅助审核、内容平台风控、教育场景检测演示，以及中文 AI 文本识别研究的可复现实验基线，已具备 FastAPI 后端 + Next.js 前端的完整可运行形态。

**Evidence**:
1. 教育场景：辅助识别课程作业、报告、摘要中的 AI 生成内容
2. 内容治理场景：用于平台审核、投稿筛查或辅助风控分析
3. 研究与演示场景：作为中文 AI 文本检测任务的工程化原型，展示"检测 + 边界定位 + 可解释证据"链路
4. 工程价值：后端 FastAPI 服务、前端 Next.js 演示界面，支持在线检测与项目问答

**Source**: `docs/project/ADVISOR_ACADEMIC_QA.md`, `docs/project/DEFENSE_CURRENT_STATUS.md`, `api/api.py`

---

### A-4. 一句话讲清

```yaml
---
chapter_id: A-4
title: "一句话讲清"
tags: [非技术评委, 快速说明]
evidence_paths: [docs/project/ADVISOR_ACADEMIC_QA.md]
priority: medium
aliases: ["一句话介绍", "怎么跟不懂技术的人说"]
---
```

**Problem**: 如果评委不懂技术，怎么一句话讲清楚这个项目？

**Conclusion**: 我做的是一个中文 AI 文本检测系统，它不仅能判断一段文字更像人写还是 AI 写，还能在一些人机混写场景下大致指出风格切换的位置。

**Evidence**:
1. 覆盖主任务：文档级人类/AI 二分类
2. 覆盖扩展能力：混合文本边界定位
3. 覆盖应用价值：可部署的工程系统

**Source**: `docs/project/ADVISOR_ACADEMIC_QA.md`（Q60）

---

## B. 方法与创新

### B-1. 为什么选择 BERT

```yaml
---
chapter_id: B-1
title: "为什么选择 BERT"
tags: [BERT, 模型选择, 编码器]
evidence_paths: [docs/thesis/theoretical_foundations.md, docs/project/ADVISOR_ACADEMIC_QA.md]
priority: high
aliases: ["为什么不用 GPT", "为什么不选生成模型"]
---
```

**Problem**: 为什么选择 BERT 而不是 GPT / LLaMA 一类生成模型？

**Conclusion**: 本项目是监督判别任务，BERT 双向编码器更适合理解整体语义并执行分类，且微调和部署成本低、与 `[SEP]` 边界标记等扩展机制更契合。

**Evidence**:
1. 任务匹配：编码器更适合理解整段文本并输出分类标签，GPT/LLaMA 更擅长自回归生成
2. 工程成本：`bert-base-chinese` 微调和部署成本显著低于超大生成模型，更适合本科毕设场景
3. 扩展兼容：`[SEP]` 边界标记、Token 级边界检测和级联结构与编码器式结构更契合

**Source**: `docs/thesis/theoretical_foundations.md`, `docs/project/ADVISOR_ACADEMIC_QA.md`（Q2）

---

### B-2. 双层检测架构

```yaml
---
chapter_id: B-2
title: "双层检测架构"
tags: [双层架构, 级联, 分类器, 边界检测器]
evidence_paths: [docs/project/ADVISOR_ACADEMIC_QA.md, docs/thesis/project_technical_deep_dive.md]
priority: high
aliases: ["两个模型怎么配合", "分类器和边界检测器什么关系"]
---
```

**Problem**: 双层检测架构中，分类器与边界检测器分别承担什么作用？

**Conclusion**: 文档级分类器负责判断整篇文本类别，边界检测器负责在混合文本中定位风格切换位置，两者级联互补而非替代。

**Evidence**:
1. 分类器回答"这段文本整体更像 Human 还是 AI？"
2. 边界检测器回答"如果是混合文本，风格切换大概发生在哪里？"
3. 只做分类无法分析人机混写场景；只做边界检测会在纯文本场景引入不必要开销和伪边界风险

**Source**: `docs/project/ADVISOR_ACADEMIC_QA.md`（Q3, Q23）

---

### B-3. [SEP] 边界标记

```yaml
---
chapter_id: B-3
title: "[SEP] 边界标记"
tags: [SEP, 边界标记, 混合文本, C2]
evidence_paths: [docs/project/ADVISOR_ACADEMIC_QA.md, docs/thesis/chapter5_experiments_filled.md, docs/thesis/theoretical_foundations.md]
priority: high
aliases: ["SEP 怎么用", "为什么加 SEP 有效"]
---
```

**Problem**: 为什么 `[SEP]` 边界标记能够提升混合文本检测效果？

**Conclusion**: `[SEP]` 显式提供段落切换信号，使模型更容易学习"前后片段作者风格不同"的边界特征；引入后 C2 续写检测率从 79.82% 提升到 93.84%。

**Evidence**:
1. 预训练适配：BERT 在预训练阶段已学习过与句段分隔相关的表示
2. 任务建模：混合文本本质是风格切换检测，显式边界标记降低模型猜测切换点难度
3. 实验结果：引入 `[SEP]` 后 C2 类续写混合文本检测率提升 14.02 个百分点

**Source**: `docs/project/ADVISOR_ACADEMIC_QA.md`（Q4）, `docs/thesis/chapter5_experiments_filled.md`

---

### B-4. 三大创新点

```yaml
---
chapter_id: B-4
title: "三大创新点"
tags: [创新点, 答辩核心]
evidence_paths: [docs/thesis/theoretical_foundations.md, docs/thesis/project_technical_deep_dive.md, docs/thesis/chapter5_experiments_filled.md]
priority: high
aliases: ["创新点是什么", "你有什么贡献"]
---
```

**Problem**: 本文的核心创新点是什么？

**Conclusion**: 三大创新：(1) 中文 BERT 微调二分类方法+标签平滑/长度感知/加权采样/温度校准；(2) `[SEP]` 边界标记机制；(3) 双层检测架构实现从整篇判别到细粒度边界定位。

**Evidence**:
1. 创新 1：在中文 AI 文本检测任务上构建完整的 BERT 微调二分类方法，通过多项技术提升准确率与可信度
2. 创新 2：引入 `[SEP]` 边界标记机制，使模型更敏感地感知人类段落与 AI 段落的风格切换
3. 创新 3：构建"双层检测架构"，将文档级分类器与 Token 级边界检测器级联
4. 附加价值：完成数据治理、无泄露评估、API 与前端演示系统的工程闭环

**Source**: `docs/thesis/theoretical_foundations.md`, `docs/thesis/project_technical_deep_dive.md`, `docs/thesis/chapter5_experiments_filled.md`

---

### B-5. 不选 RoBERTa/GPT/零样本/水印

```yaml
---
chapter_id: B-5
title: "不选 RoBERTa/GPT/零样本/水印"
tags: [模型选择, RoBERTa, 零样本, 水印]
evidence_paths: [docs/project/ADVISOR_ACADEMIC_QA.md, docs/plans/ai_text_detection_research_survey_2025.md]
priority: medium
aliases: ["为什么不选 RoBERTa", "为什么不用零样本", "为什么不选水印"]
---
```

**Problem**: 为什么不选择 RoBERTa、GPT 自判别、零样本方法或水印方法作为主线？

**Conclusion**: 在中文判别任务中 BERT 与 RoBERTa 效果差距不显著且 BERT 生态更成熟；零样本方法依赖强、对短文本敏感、推理成本高；水印方法前提条件不满足——因此 BERT 微调是当前场景下更稳妥的主线选择。

**Evidence**:
1. RoBERTa：中文判别任务效果差距不显著，`bert-base-chinese` 生态更成熟，本文重点在数据治理而非换骨干
2. 零样本（DetectGPT/Binoculars）：对参考模型依赖强、对短文本与轻度改写敏感、推理成本高
3. 水印：历史文本无水印、开源模型未必支持、后处理后水印可能失效
4. GPT 自判别：推理成本更高、输出受提示词影响、难以保证可复现性

**Source**: `docs/project/ADVISOR_ACADEMIC_QA.md`（Q20, Q21, Q27, Q57）

---

## C. 数据与训练

### C-1. 训练集规模与来源

```yaml
---
chapter_id: C-1
title: "训练集规模与来源"
tags: [数据集, 训练集, 63113, 多模型覆盖]
evidence_paths: [docs/project/ADVISOR_ACADEMIC_QA.md, models/bert_v11c_boundary_fix/training_log.json, docs/thesis/thesis_data_reference.md]
priority: high
aliases: ["训练数据有多少", "训练集怎么来的"]
---
```

**Problem**: 当前训练集规模和来源覆盖如何？

**Conclusion**: 清洗后训练集 63,113 条，覆盖 8 大 LLM 家族、46 个具体模型、92 类人类文本来源；AI 侧覆盖 GPT/DeepSeek/Qwen/Claude/Gemini/Kimi/LLaMA/GLM 等，Human 侧覆盖 HC3 问答/THUCNews/Wikipedia_CN/M4/VCSUM/formal_collected 等。

**Evidence**:
1. 训练样本数：63,113（AI: 32,744, Human: 30,369）
2. AI 侧覆盖 8 大 LLM 家族、46 个具体模型
3. Human 侧覆盖 92 类人类文本来源
4. 独立评估集 `independent_data` 专门纳入训练未见的 GPT-4/GPT-5/Gemini-3/LLaMA-405B 等

**Source**: `docs/project/ADVISOR_ACADEMIC_QA.md`（Q17）, `models/bert_v11c_boundary_fix/training_log.json`, `docs/thesis/thesis_data_reference.md`

---

### C-2. V11c vs V10 提升来源

```yaml
---
chapter_id: C-2
title: "V11c vs V10 提升来源"
tags: [数据治理, V11c, V10, 模板移除, 弱域增补]
evidence_paths: [docs/project/DEFENSE_CURRENT_STATUS.md, models/bert_v11c_boundary_fix/training_log.json, docs/project/RISK_IMPLEMENTATION_2026-02-12.md]
priority: high
aliases: ["V11c 比 V10 好在哪", "提升怎么来的"]
---
```

**Problem**: V11c 相比 V10 的性能提升主要来自哪些因素？

**Conclusion**: 提升来自数据中心治理而非更换骨干模型：移除 750 条硬编码模板 + 1,767 条 unknown + 7 条长度违规，补充 300 条 formal_collected + 300 条 LLaMA-405B + 2,131 条长文 AI 边界修复样本；独立评估集准确率 97.69%→98.57%（+0.88%），总错误数下降 38%。

**Evidence**:
1. 移除 750 条硬编码模板样本、1,767 条 unknown 样本、7 条长度违规样本
2. 补充 300 条 formal_collected 弱域 + 300 条 LLaMA-405B 弱域 + 2,131 条长文 AI 边界修复
3. 独立评估集准确率：97.69%→98.57%（+0.88%），总错误 21→13（-38%）
4. LLaMA-405B 检出率：88.9%→100%

**Source**: `docs/project/DEFENSE_CURRENT_STATUS.md`, `models/bert_v11c_boundary_fix/training_log.json`, `docs/project/RISK_IMPLEMENTATION_2026-02-12.md`

---

### C-3. 数据治理逻辑

```yaml
---
chapter_id: C-3
title: "数据治理逻辑"
tags: [数据清洗, 模板移除, unknown, 弱域增补]
evidence_paths: [docs/project/RISK_IMPLEMENTATION_2026-02-12.md, models/bert_v11c_boundary_fix/training_log.json]
priority: high
aliases: ["数据怎么清洗的", "治理做了什么"]
---
```

**Problem**: 数据治理具体做了什么？

**Conclusion**: V11c 经四阶段治理：A1 风险审计移除模板样本、A1 unknown 分流移除无法追溯来源样本、B2 弱域增补 formal_collected 和 LLaMA-405B、B2 长文 AI 边界修复补充 256+ 样本。

**Evidence**:
1. A1 风险审计：移除 750 条硬编码模板匹配样本
2. A1 unknown 分流：移除 1,767 条无法追溯来源的样本
3. B2 弱域增补：补充 300 条 formal_collected + 300 条 LLaMA-405B 样本
4. B2 长文 AI 边界修复：补充 2,131 条 256+ 字符 AI 样本

**Source**: `docs/project/RISK_IMPLEMENTATION_2026-02-12.md`, `models/bert_v11c_boundary_fix/training_log.json`

---

### C-4. Data-Centric AI 理论锚点

```yaml
---
chapter_id: C-4
title: "Data-Centric AI 理论锚点"
tags: [Data-Centric AI, 控制变量, 数据治理]
evidence_paths: [docs/project/ADVISOR_ACADEMIC_QA.md, docs/project/DEFENSE_KB_CURATED.md]
priority: high
aliases: ["为什么说体现了 Data-Centric", "改数据比换模型好在哪"]
---
```

**Problem**: 为什么说这个项目体现了 Data-Centric AI 思路？

**Conclusion**: V10→V11c 的核心改进在固定模型骨干、超参与评估协议的前提下，通过数据治理持续提升性能——独立评估集准确率 97.69%→98.57%、总错误数下降 38%，是 Data-Centric AI 的典型实证路径。

**Evidence**:
1. 固定变量：模型骨干（bert-base-chinese）、超参数、评估协议均不变
2. 唯一变量：训练数据的治理（清洗 + 增补 + 修复）
3. 效果：独立评估集准确率 +0.88%，总错误 -38%
4. 证明：改数据在此任务上比单纯换结构更有效、更可解释

**Source**: `docs/project/ADVISOR_ACADEMIC_QA.md`（Q19）, `docs/project/DEFENSE_KB_CURATED.md`（§7）

---

## D. 指标与口径

### D-1. 核心指标

```yaml
---
chapter_id: D-1
title: "核心指标"
tags: [指标, Accuracy, Recall, ECE, 口径]
evidence_paths: [docs/project/DEFENSE_CURRENT_STATUS.md, models/bert_v11c_boundary_fix/eval_comparison.json, models/bert_v11c_boundary_fix/training_log.json]
priority: high
aliases: ["指标有哪些", "核心数字是多少"]
---
```

**Problem**: 当前推荐模型的核心指标有哪些？

**Conclusion**: 论文主表 Accuracy 98.69%（2599 条）、验证集 98.75%、独立评估集 98.57%、三集平均 98.56%、Token 级边界检测 96.69%、ECE 0.0034。

**Evidence**:
1. 论文主表 / 基线对比整体 Accuracy：98.69%（2,599 条无泄露评估集）
2. 验证集准确率：98.75%；独立评估集准确率：98.57%；三集平均：98.56%
3. Token 级边界检测准确率：96.69%
4. 最优温度 T=0.8165，ECE 从 0.0168 降至 0.0034

**Source**: `docs/project/DEFENSE_CURRENT_STATUS.md`, `models/bert_v11c_boundary_fix/eval_comparison.json`, `models/bert_v11c_boundary_fix/training_log.json`

---

### D-2. 多个 98% 数字怎么报

```yaml
---
chapter_id: D-2
title: "多个 98% 数字怎么报"
tags: [口径, 答辩技巧, 98.69, 98.56, 98.75]
evidence_paths: [docs/project/ADVISOR_ACADEMIC_QA.md, docs/project/DEFENSE_KB_CURATED.md]
priority: high
aliases: ["到底报哪个数字", "几个 98 什么关系"]
---
```

**Problem**: 98.69%、98.56%、98.75%、98.57% 这些数字分别是什么口径？

**Conclusion**: 先说口径再报数字：98.75% 是验证集、98.57% 是 independent_data、98.56% 是三集平均、98.69% 是论文主表 2599 条整体 Accuracy。

**Evidence**:
1. 98.75%：验证集准确率（`val_acc` on `merged_v2/val.csv`）
2. 98.57%：独立评估集准确率（`independent_data` 910 条）
3. 98.56%：三集 Accuracy 直接平均（快速汇报口径）
4. 98.69%：2,599 条无泄露评估集按统一主表口径汇总的整体 Accuracy

**Source**: `docs/project/ADVISOR_ACADEMIC_QA.md`（Q29, Q46, Q54）, `docs/project/DEFENSE_KB_CURATED.md`（§3）

---

### D-3. 混淆矩阵

```yaml
---
chapter_id: D-3
title: "混淆矩阵"
tags: [混淆矩阵, FP, FN, 误报, 漏报]
evidence_paths: [models/bert_v11c_boundary_fix/eval_perclass.json, docs/thesis/chapter5_experiments_filled.md]
priority: high
aliases: ["误报漏报多少", "混淆矩阵是什么"]
---
```

**Problem**: 混淆矩阵说明了什么？误报与漏报分别是多少？

**Conclusion**: 三集聚合（2,599 条）TN=1586、FP=28、FN=6、TP=979；人类正确识别率 98.27%、AI 正确识别率 99.39%、漏报率 0.61%、误报率 1.73%，整体风险形态"低漏报、可控误报"。

**Evidence**:
1. TN=1,586, FP=28, FN=6, TP=979
2. 人类文本正确识别率：98.27%；AI 文本正确识别率：99.39%
3. 漏报率：0.61%；误报率：1.73%
4. 风险形态：低漏报、可控误报，符合学术检测对 AI 文本召回率的偏好

**Source**: `models/bert_v11c_boundary_fix/eval_perclass.json`, `docs/thesis/chapter5_experiments_filled.md`

---

### D-4. 校准 ECE

```yaml
---
chapter_id: D-4
title: "校准 ECE"
tags: [Temperature Scaling, ECE, 概率校准, 可信度]
evidence_paths: [docs/thesis/theoretical_foundations.md, docs/thesis/chapter5_experiments_filled.md, models/bert_v11c_boundary_fix/README.md]
priority: high
aliases: ["ECE 是什么", "温度校准怎么做的"]
---
```

**Problem**: Temperature Scaling 与 ECE 在本文中分别说明什么？

**Conclusion**: Temperature Scaling 是后置概率校准方法；ECE 衡量"模型说自己有多确定"与"实际有多准确"之间的偏差；本文 T*=0.8165，ECE 从 0.0168 降到 0.0034，说明模型概率输出可信。

**Evidence**:
1. Temperature Scaling：不改变分类排序，调整 softmax 输出概率使其更接近真实置信度
2. ECE（Expected Calibration Error）：衡量校准前后概率偏差
3. T*=0.8165，ECE 从 0.0168 降到 0.0034（降幅 80%）
4. 模型不仅"更准"而且"更稳"，适合答辩/API 场景对置信度说明

**Source**: `docs/thesis/theoretical_foundations.md`, `docs/thesis/chapter5_experiments_filled.md`, `models/bert_v11c_boundary_fix/README.md`

---

### D-5. 召回率含义

```yaml
---
chapter_id: D-5
title: "召回率含义"
tags: [召回率, Recall, 99.28%, 漏检]
evidence_paths: [docs/project/ADVISOR_ACADEMIC_QA.md, models/bert_v11c_boundary_fix/eval_perclass.json]
priority: high
aliases: ["99.28% 召回率是什么意思", "召回率代表什么"]
---
```

**Problem**: 99.28% 的召回率意味着什么？

**Conclusion**: 召回率 99.28% 意味着 AI 文本样本中模型能识别出绝大多数，仅漏检 6 条（per-class 口径约 99.39%）；本文方案更偏向减少漏检，符合学术诚信和内容审核场景需求。

**Evidence**:
1. AI 样本共 985 条，仅漏检 6 条
2. per-class 口径下 AI 文本正确识别率约 99.39%
3. 在学术诚信与内容审核场景下，漏检 AI 文本通常比适度误报更难接受
4. 召回率是所有对比方法中唯一超过 99% 的方案

**Source**: `docs/project/ADVISOR_ACADEMIC_QA.md`（Q30）, `models/bert_v11c_boundary_fix/eval_perclass.json`

---

## E. 评估协议

### E-1. 2599 主口径

```yaml
---
chapter_id: E-1
title: "2599 主口径"
tags: [评估集, 2599, 无泄露, 三集]
evidence_paths: [docs/project/ADVISOR_ACADEMIC_QA.md, docs/thesis/chapter5_experiments_filled.md]
priority: high
aliases: ["为什么用 2599 条", "评估集怎么设计的"]
---
```

**Problem**: 为什么以 2,599 条无泄露评估集作为主口径？

**Conclusion**: 2,599 条由 `core_v1_test_clean`(545) + `independent_data`(910) + `merged_v2_val_clean`(1144) 三个独立子集组成，规模足够、无泄露、覆盖更广，比单一测试集更稳妥。

**Evidence**:
1. 规模足够：三个独立子集组成，样本量明显高于单一测试集
2. 无泄露：三个子集都经过与训练集的去重校验
3. 覆盖更广：既包含常规测试样本，也包含训练未充分覆盖的新型 LLM 输出
4. 适合作为答辩主表和基线比较的统一依据

**Source**: `docs/project/ADVISOR_ACADEMIC_QA.md`（Q16）

---

### E-2. independent_data 重要性

```yaml
---
chapter_id: E-2
title: "independent_data 重要性"
tags: [independent_data, 泛化, 独立评估]
evidence_paths: [docs/project/ADVISOR_ACADEMIC_QA.md, models/bert_v11c_boundary_fix/eval_comparison.json]
priority: high
aliases: ["independent_data 是什么", "为什么要有独立评估集"]
---
```

**Problem**: independent_data 对本文为什么重要？

**Conclusion**: independent_data 专门用于检验模型面对训练未充分覆盖分布时是否仍稳定，包含 GPT-4/GPT-5/Gemini-3/LLaMA-405B 等前沿模型输出，与训练集做了去泄露校验，是"不是只在训练分布内表现好"的关键证据。

**Evidence**:
1. 包含训练未充分覆盖的新型 LLM 输出（GPT-4/GPT-5/Gemini-3/LLaMA-405B 等）
2. 包含不同来源的人类文本，验证 Human 侧稳定性
3. 与训练集做了去泄露校验
4. 是本文泛化能力的重要证据（但不是无限外推的充分条件）

**Source**: `docs/project/ADVISOR_ACADEMIC_QA.md`（Q28, Q47）, `models/bert_v11c_boundary_fix/eval_comparison.json`

---

### E-3. 与基线对比

```yaml
---
chapter_id: E-3
title: "与基线对比"
tags: [基线对比, FastText, TextCNN, DPCNN, BERT-BiGRU]
evidence_paths: [docs/thesis/chapter5_experiments_filled.md, docs/project/DEFENSE_KB_CURATED.md]
priority: high
aliases: ["和基线比怎么样", "BERT-BiGRU 比你高怎么办"]
---
```

**Problem**: 与基线方法对比时应如何表述？

**Conclusion**: 在 2,599 条评估集上 FastText 97.65%、TextCNN 97.08%、DPCNN 97.04%、BERT-BiGRU 98.81%、本文 V11c 98.69%；关键不在 Accuracy 最高，而在 V11c 召回率 99.28% 唯一突破 99%、且具备 `[SEP]` 边界机制和完整工程部署能力。

**Evidence**:
1. FastText 97.65%、TextCNN 97.08%、DPCNN 97.04%、BERT-BiGRU 98.81%、本文 98.69%
2. V11c 召回率达 99.28%，是所有方法中唯一突破 99% 的方案
3. 降低漏检在 AI 文本检测场景比单纯追求更高 Accuracy 更重要
4. V11c 还具备 `[SEP]` 边界机制、Token 边界检测与完整工程部署能力

**Source**: `docs/thesis/chapter5_experiments_filled.md`, `docs/project/DEFENSE_KB_CURATED.md`（§8）

---

## F. 局限与质疑

### F-1. 局限性

```yaml
---
chapter_id: F-1
title: "局限性"
tags: [局限性, 外部有效性, 边界]
evidence_paths: [docs/project/ADVISOR_ACADEMIC_QA.md, docs/project/DEFENSE_KB_CURATED.md, models/bert_v11c_boundary_fix/README.md]
priority: high
aliases: ["有什么局限", "短板是什么"]
---
```

**Problem**: 本项目目前的主要局限性是什么？

**Conclusion**: 主要针对中文场景、对弱覆盖文体可能欠拟合、对新模型和重度改写文本性能可能波动、边界检测器对弱边界场景仍有困难；最准确结论是"在当前中文工程场景下可复现的检测方案"而非"彻底解决 AI 文本检测"。

**Evidence**:
1. 主要针对中文文本，对英文或多语场景不做保证
2. 对诗歌、古文、社交媒体极短文本等弱覆盖文体可能存在欠拟合风险
3. 对训练与评估都未覆盖的新模型、经过重度人工改写的 AI 文本，性能仍可能波动
4. 边界检测器对明显拼接式混合文本效果较好，但对弱边界、强润色场景仍有困难

**Source**: `docs/project/ADVISOR_ACADEMIC_QA.md`（Q9, Q55）, `docs/project/DEFENSE_KB_CURATED.md`（§13）

---

### F-2. 跨域泛化

```yaml
---
chapter_id: F-2
title: "跨域泛化"
tags: [跨域, 英文, 专业领域, 泛化边界]
evidence_paths: [docs/project/ADVISOR_ACADEMIC_QA.md]
priority: medium
aliases: ["换英文行不行", "法律医学能用吗"]
---
```

**Problem**: 如果换成英文文本或专业领域，当前方法还能直接用吗？

**Conclusion**: 不能直接外推——模型和数据集围绕中文构建、`bert-base-chinese` 预训练为中文语料；对法律、医学等专业领域也未充分覆盖，属于明确承认的外部有效性边界。

**Evidence**:
1. 当前模型和数据集都是围绕中文构建的
2. `bert-base-chinese` 预训练本身就是中文语料
3. 法律文书、医学报告、古代汉语等专业领域没有被充分覆盖
4. 如需英文版本，需重新选择预训练模型、英文数据和评估协议

**Source**: `docs/project/ADVISOR_ACADEMIC_QA.md`（Q43, Q44）

---

### F-3. 过拟合风险

```yaml
---
chapter_id: F-3
title: "过拟合风险"
tags: [过拟合, independent_data, 格式对抗]
evidence_paths: [docs/project/ADVISOR_ACADEMIC_QA.md, models/bert_v11c_boundary_fix/eval_comparison.json]
priority: high
aliases: ["是不是过拟合", "怎么证明没过拟合"]
---
```

**Problem**: 怎么证明不是过拟合？

**Conclusion**: 最稳妥回答不是"绝对没有过拟合"而是"当前证据表明模型没有明显陷入训练分布内自嗨"——independent_data 仍达 98.57%、格式对抗测试最大下降仅 0.05%、多子集一致性说明模型并非靠单一捷径特征取胜。

**Evidence**:
1. 使用 910 条 `independent_data` 验证训练未充分覆盖的新模型和新来源文本
2. V11c 在 `independent_data` 上仍达到 98.57% 准确率
3. 格式对抗测试最大性能下降仅 0.05%
4. 混淆矩阵分析和多子集一致性也支持"非过拟合"判断

**Source**: `docs/project/ADVISOR_ACADEMIC_QA.md`（Q41）, `models/bert_v11c_boundary_fix/eval_comparison.json`

---

### F-4. 商用工具对比

```yaml
---
chapter_id: F-4
title: "商用工具对比"
tags: [GPTZero, Turnitin, 商用工具, 可复现]
evidence_paths: [docs/project/ADVISOR_ACADEMIC_QA.md, docs/project/DEFENSE_KB_CURATED.md, docs/plans/ai_text_detection_research_survey_2025.md]
priority: medium
aliases: ["和 GPTZero 比怎么样", "和商用工具比有什么意义"]
---
```

**Problem**: 和商用检测工具相比到底有什么意义？

**Conclusion**: 本文不主张全面超越所有商业产品，而是提供一套可复现、可解释、可控的中文检测研究与工程方案；商业工具通常黑箱、对中文和混合文本支持有限，本文完整给出数据/方法/评估/工程闭环。

**Evidence**:
1. 可比项：本文使用统一 2,599 条评估集公开口径（Accuracy 98.69%、Recall 99.28%、ECE 0.0034）；商用工具大多不公开评测协议
2. 不可比项：本文聚焦中文场景并支持文档级 + Token 级双层检测；多数商用工具仅给整段判别
3. 可解释性：本文方法、数据治理、温度校准、错误案例都有完整记录可复现和审查

**Source**: `docs/project/ADVISOR_ACADEMIC_QA.md`（Q56）, `docs/project/DEFENSE_KB_CURATED.md`（§15）, `docs/plans/ai_text_detection_research_survey_2025.md`

---

## G. 工程与训练细节

### G-1. 训练超参依据

```yaml
---
chapter_id: G-1
title: "训练超参依据"
tags: [超参数, learning_rate, label_smoothing, length_penalty, accum_steps, max_length]
evidence_paths: [models/bert_v11c_boundary_fix/training_log.json, docs/project/DEFENSE_KB_CURATED.md]
priority: high
aliases: ["超参数怎么选的", "为什么用这个学习率"]
---
```

**Problem**: 训练超参为什么这样选？

**Conclusion**: lr=1e-05 是 BERT-base 微调常规起点且该规模数据下的稳健点；label_smoothing=0.05 抑制过度自信；length_penalty_weight=0.1 缓解长度偏置；accum_steps=4 在 8GB 显存下等效 batch=32；max_length=256 覆盖 95%+ 样本且推理比 512 快约 4 倍；patience=2 因多代实验均观察到 Epoch 2 最优。

**Evidence**:
1. `learning_rate=1e-05`：2e-05 在小数据上过拟合、5e-06 收敛偏慢，1e-05 是 63,113 条规模下的稳健点
2. `label_smoothing=0.05`：配合 Temperature Scaling 使概率输出更稳，0 时 ECE 抬升、0.1 牺牲 Accuracy
3. `length_penalty_weight=0.1`：缓解长度偏置，0.2 时短文本召回反而下降
4. `accum_steps=4`：batch_size=8 + 4 步累积等效 batch=32，兼顾稳定性与显存占用
5. `max_length=256`：覆盖训练集 95%+ 样本，推理速度比 512 快约 4 倍
6. Early Stopping patience=2：V10/V11a/V11b/V11c 全部观察到 Epoch 2 最优

**Source**: `models/bert_v11c_boundary_fix/training_log.json`, `docs/project/DEFENSE_KB_CURATED.md`（§16）

---

### G-2. 部署效率

```yaml
---
chapter_id: G-2
title: "部署效率"
tags: [推理速度, 吞吐, 显存, 部署]
evidence_paths: [docs/project/DEFENSE_CURRENT_STATUS.md, models/CLAUDE.md]
priority: medium
aliases: ["推理快不快", "显存多少", "能部署吗"]
---
```

**Problem**: 当前模型的部署效率和资源占用如何？

**Conclusion**: 1,144 条样本总耗时 8.98 秒、吞吐 127.4 样本/秒、GPU 峰值显存 672 MB、Batch size 32，具备明确的工程部署可行性。

**Evidence**:
1. 推理样本数：1,144 条总耗时 8.98 秒
2. 吞吐：127.4 样本/秒
3. GPU 峰值显存：672 MB
4. Batch size：32（8 × 4 accum_steps）

**Source**: `docs/project/DEFENSE_CURRENT_STATUS.md`, `models/CLAUDE.md`

---

### G-3. 混合文本检测能力

```yaml
---
chapter_id: G-3
title: "混合文本检测能力"
tags: [混合文本, C2, C3, C4, 边界检测]
evidence_paths: [docs/thesis/chapter5_experiments_filled.md, docs/project/ADVISOR_ACADEMIC_QA.md]
priority: high
aliases: ["混合文本怎么看", "人机混写能检测吗"]
---
```

**Problem**: 混合文本检测与边界定位能力体现在哪里？

**Conclusion**: 系统支持 C2（AI 续写 93.84%）、C3（AI 改写 100%）、C4（AI 润色 92.89%）、Human 纯文本 99.58%；引入 `[SEP]` 后 C2 检测率从 79.82% 提升到 93.84%，提升约 14 个百分点。

**Evidence**:
1. C2（AI 续写）：93.84%——引入 `[SEP]` 后从 79.82% 提升 14.02 个百分点
2. C3（AI 改写）：100.00%；C4（AI 润色）：92.89%
3. Human 纯文本：99.58%
4. 不只给标签，具备对人机混合写作进行结构化分析的能力

**Source**: `docs/thesis/chapter5_experiments_filled.md`, `docs/project/ADVISOR_ACADEMIC_QA.md`（Q14, Q33）

---

### G-4. 误报漏报取舍

```yaml
---
chapter_id: G-4
title: "误报漏报取舍"
tags: [误报, 漏报, 取舍, 阈值]
evidence_paths: [docs/project/ADVISOR_ACADEMIC_QA.md, models/bert_v11c_boundary_fix/eval_perclass.json]
priority: high
aliases: ["误报漏报怎么平衡", "为什么宁可多报不要漏"]
---
```

**Problem**: 在误报和漏报之间，本文是如何取舍的？

**Conclusion**: 本文更倾向降低漏报（FN=6），在可控误报范围内（FP=28，误报率 1.73%）换取更高 AI 文本召回能力；这符合学术检测和内容审核场景"宁可进一步人工复核，也不要大量漏掉 AI 文本"的需求。

**Evidence**:
1. 当前 FN 只有 6 条（漏报率 0.61%），FP 为 28 条（误报率 1.73%）
2. 在学术诚信与内容审核场景下，漏检 AI 文本通常比适度误报更难接受
3. 如果应用方更担心误判，可在后端把阈值上调到 0.85 或更高
4. 取舍不是"忽略误报"，而是在实际需求下优先保证 AI 文本识别能力

**Source**: `docs/project/ADVISOR_ACADEMIC_QA.md`（Q25, Q53）, `models/bert_v11c_boundary_fix/eval_perclass.json`
