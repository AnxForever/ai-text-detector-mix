# 模型训练改进计划 — 下一轮迭代

> 生成时间: 2026-02-10
> 基于: 数据集盘点 + 前沿方法调研

---

## 一、当前状态摘要

### 1.1 模型现状

| 模型 | 验证准确率 | 训练样本 | 已知问题 |
|------|-----------|---------|---------|
| bert_v2_with_sep | 98.71% | ~66,000 | 基准线，整体表现最平衡 |
| bert_v3_core_v2 | 99.57% | 57,435 | 技术文档 ✅ |
| bert_v4_defense_focused | 99.75% | 6,533 | 小数据、技术文档 ✅、泛化待验证 |
| bert_v5_paired | 100%* | 7,800 | **技术文档漏检**，小验证集 |
| bert_v6_merged | 97.94% | 54,649 | **技术文档漏检**，准确率反降 |
| bert_v7_improved | 待确认 | 待确认 | 最新，效果待验证 |
| bert_span_detector | 96.69% (Token) | - | 边界检测器 |

**核心问题**: v5/v6 模型在 AI 生成的技术文档上严重漏检（误判为 Human），根因是训练数据中技术类内容 90%+ 为 AI 生成，缺少人类技术文档样本。

### 1.2 数据集盘点

#### 核心训练数据

| 数据集 | 路径 | 样本数 | 格式 | 说明 |
|-------|------|-------|------|------|
| core_v1 | `datasets/active/core_v1/` | train: 600,952 / val: 72,169 / test: 78,380 | CSV (text, label, source, length, category) | 原始主训练集，推荐 |
| core_v2 | `datasets/active/core_v2/` | train: 790,664 / val: 96,294 / test: 97,411 | CSV | 扩展版 |
| core_v3 | `datasets/active/core_v3/` | train: 807,445 / val: 87,871 | CSV | 最新版，无单独测试集 |
| 主训练集.jsonl | `datasets/active/主训练集.jsonl` | 31,684 | JSONL (text_id, text, label, scenario, model...) | AI 生成数据，字段丰富 |

#### 补充数据

| 数据集 | 路径 | 样本数 | 说明 |
|-------|------|-------|------|
| defense_focused | `datasets/defense_focused/` | train: 18,566 / val: 2,404 / test: 2,494 | 防御型数据，含技术文档 |
| paired_v1 | `datasets/paired/paired_v1_*` | train: 2,600 / test: 325 | 配对数据 v1 |
| paired_v2 | `datasets/paired/paired_v2_*` | train: 7,800 / test: 975 | 配对数据 v2 |
| paired_v3 | `datasets/paired/paired_v3_all_*` | train: 9,298 / test: 1,163 | 配对数据 v3 (最全) |
| merged_v1 | `datasets/merged_v1/` | train: 608,752 / val: 73,144 | 合并数据 v1 |
| merged_v2 | `datasets/merged_v2/` | train: 618,769 / val: 74,154 | 合并数据 v2 |
| human_consolidated | `datasets/human_consolidated/` | all: 5,512 / supplement: 7,512 / toutiao: 2,000 | 人类数据汇总 |
| human_supplement | `datasets/human_supplement/` | 10,662 | 多样化人类样本 |
| defense_patch | `datasets/defense_patch/` | ~1,000+ | 防御补丁数据 |
| generated_tutorials | `datasets/generated_tutorials/` | CSV: 18KB / JSONL: 空 | 技术教程生成数据 |

#### 外部数据集

| 数据集 | 路径 | 大小 | 说明 |
|-------|------|------|------|
| HC3-Chinese | `datasets/external/HC3-Chinese/` | 42MB, 12,853 条 | 中文人类+ChatGPT 问答对 |
| M4 | `datasets/external/M4/` | 1.2GB | 多语言多模型生成文本 (含 qazh_chatgpt/davinci 中文) |
| DuReader | `datasets/external/DuReader/` | 24MB | 中文阅读理解数据集 |
| THUCNews | `datasets/external/THUCNews/` | 4MB | 新闻文本 |
| LCSTS | `datasets/external/LCSTS/` | 4.9MB | 中文短文本摘要 |
| VCSum | `datasets/external/VCSum/` | 54MB | 视频字幕摘要 |

#### 评估数据

| 数据集 | 路径 | 说明 |
|-------|------|------|
| fair_test | `datasets/eval/fair_test/` | 公平测试集 (core_v1_test_clean, independent_data, merged_v2_val_clean) |
| eval_splits_v1 | `datasets/eval/splits/v1/` | ID/OOD/Mixed 评估拆分 |

---

## 二、前沿方法调研 (2024-2026)

### 2.1 对比学习 (Contrastive Learning)

#### DeTeCtive (NeurIPS 2024)
- **论文**: "DeTeCtive: Detecting AI-generated Text via Multi-Level Contrastive Learning"
- **GitHub**: https://github.com/heyongxin233/DeTeCtive
- **核心思路**: 多级对比学习框架 + 密集信息检索管道
  - 句子级对比: 学习人类 vs AI 写作风格的细粒度差异
  - 文档级对比: 捕获整体文本结构特征
  - 多任务辅助: 同时进行分类和风格识别
- **关键优势**: 解决 OOD 泛化问题，兼容多种文本编码器 (BERT/RoBERTa)
- **与本项目关联**: 直接适用，可在现有 BERT 基础上加入对比学习头

#### Span-level Contrastive Detection (Knowledge-Based Systems, 2026)
- **论文**: "Span-level detection of AI-generated scientific text via contrastive learning and structural calibration"
- **核心思路**:
  - Span 级别检测（非文档级）
  - 对比学习 + 结构校准
  - 针对科学文本中 AI 生成片段的定位
- **与本项目关联**: 与我们的 `[SEP]` 边界标记 + bert_span_detector 思路高度吻合

#### PAN 2025 竞赛方案
- **任务**: Voight-Kampff Generative AI Detection
- **最佳方案之一**: Genre embedding + Contrastive Learning
  - 将文体类型编码融入对比学习框架
  - 支持混合文本和模糊场景检测
- **与本项目关联**: 我们的 `scenario` 字段可作为 genre embedding 输入

### 2.2 多粒度检测

#### Fine-Grained Detection (IJCNLP-AACL 2025)
- **论文**: "Fine-Grained Detection of AI-Generated Text Using Sentence-Level Segmentation"
- **核心思路**: 从文档级到句子级的序列标注
- **方法**: 使用句子级序列标注模型代替文档级二分类
- **关键发现**: 对混合/编辑后文本的检测效果远优于文档级方法

#### FAID (arXiv 2025)
- **论文**: "FAID: Fine-Grained AI-Generated Text Detection Using Multi-Task Learning"
- **核心思路**:
  - 三分类: Human / LLM-generated / Human-LLM collaborative
  - 多语言、多领域、多生成器数据集 FAIDSet
  - 多任务学习框架
- **与本项目关联**: 我们的 C2/C3/C4 混合类型正是 Human-LLM collaborative

#### Robust and Fine-Grained Detection (arXiv 2025)
- **论文**: "Robust and Fine-Grained Detection of AI Generated Texts"
- **核心思路**: 结合鲁棒性和细粒度的统一框架

### 2.3 新型架构

#### SENTRA (arXiv 2025)
- **论文**: "SENTRA: Selected-Next-Token Transformer for LLM Text Detection"
- **核心思路**: 利用 next-token prediction 概率分布作为检测特征
  - 不同于传统 BERT 分类，关注 token 级预测概率的统计特征
  - 更好地捕获 LLM 特有的生成模式

#### DivEye (ICLR 2026)
- **论文**: "Diversity Boosts AI-Generated Text Detection"
- **核心思路**: 基于 surprisal 多样性的零样本检测
  - 观察: 人类文本的词汇/结构不可预测性变化更丰富
  - 方法: 提取 surprisal 统计特征（均值、方差、峰度等）
  - 性能: 比现有零样本方法提升达 33.2%
  - 高可解释性

### 2.4 对抗鲁棒性

#### DAMAGE (ACL GenAIDetect Workshop 2025)
- **论文**: "DAMAGE: Detecting Adversarially Modified AI Generated Text"
- **关注**: 对抗性修改后的 AI 文本检测

#### DACTYL (arXiv 2025)
- **论文**: "DACTYL: Diverse Adversarial Corpus of Texts Yielded from LLMs"
- **贡献**: 提供多样化对抗 AI 文本语料库用于评测

#### 鲁棒性综述 (MDPI Mathematics 2025)
- **论文**: "Enhancing the Robustness of AI-Generated Text Detectors: A Survey"
- **关键策略**:
  - 对抗训练: 在训练中加入 paraphrase/rewrite 样本
  - 数据增强: 使用多种 LLM 生成、不同 decoding 策略
  - 集成方法: 多模型投票

### 2.5 泛化性研究

#### On the Generalizability (UWaterloo 硕士论文 2026)
- **关键发现**:
  1. 零样本方法 (如 Binoculars) 在新模型上失效
  2. 监督学习 (RoBERTa/DeBERTa) 在跨模型场景仍维持高 TPR
  3. 多模型训练数据对泛化至关重要

#### 综合评测 (Computer Science Review 2025)
- **论文**: "AI-generated text detection: A comprehensive review of methods"
- **覆盖**: 技术基础、方法论、评估框架、实际应用

### 2.6 商业工具现状 (2025-2026)

| 工具 | 特点 | 准确率 |
|------|------|-------|
| Originality.ai | 最严格，paraphrase 检测强 | 业界领先 |
| GPTZero | 教育场景主流，批量 API | 高 |
| Winston AI | 文档级检测 | 高 |
| Turnitin | 学术诚信集成 | 中高 |
| Copyleaks | 多语言支持 | 中高 |

---

## 三、核心问题诊断

### 3.1 技术文档漏检 (最高优先级)

**现象**: bert_v5/v6 将 AI 生成的技术文档高置信度判为 Human
**根因**: 训练数据偏斜 — 技术类内容几乎全是 AI 生成，模型学到了"技术文档 = AI"的逆向偏见，当配对/合并数据引入后被稀释

**解决方向**:
1. 补充人类技术文档样本 (来源: THUCNews 科技类、CSDN 博文、知乎专栏)
2. 使用 defense_focused 数据中的成功案例
3. 对比学习约束：让模型学到"风格差异"而非"话题差异"

### 3.2 泛化能力不足

**现象**: 模型在特定验证集上高准确率，但对新场景/新模型生成文本泛化差
**根因**: 训练数据模型来源单一 (主要 deepseek-v3.2)

### 3.3 混合文本边界检测精度

**现象**: bert_span_detector 96.69% Token 准确率，仍有提升空间
**根因**: 边界检测依赖 `[SEP]` 标记，对渐进式风格变化不够敏感

---

## 四、改进方案

### 方案 A: 数据增强 + 平衡训练 (低风险，快速见效)

**目标**: 修复技术文档漏检，提升数据多样性

#### A1. 人类技术文档补充
- 从 THUCNews 科技栏目提取 2,000-5,000 条 → label=0 (Human)
- 从 DuReader 技术类问答提取人类回答 → label=0
- 从 HC3-Chinese 百科/开放问答中筛选技术类 → label=0
- 手动收集知乎技术专栏、CSDN 博文样本 200-500 条

#### A2. AI 生成多模型扩展
- 使用当前主训练集 JSONL 中的 `model` 字段分析：目前有 deepseek-v3.2、meta_llama-3.1-405b、gpt-oss-120b
- 补充: Qwen、GLM、Mixtral 等模型生成的技术文档
- 使用不同 decoding 参数 (temperature 0.3-1.0)

#### A3. 对抗样本增强
- 对 AI 生成文本做 paraphrase (词汇替换、句式调整)
- 混合人类和 AI 片段创建更多 C2/C3/C4 类型数据
- 加入"简单改写"的 AI 文本（模拟规避检测）

#### 预估效果: 技术文档检测修复 + 整体准确率回升到 99%+

### 方案 B: 对比学习增强 (中风险，显著提升泛化)

**目标**: 引入多级对比学习，提升跨模型泛化能力

#### B1. DeTeCtive 框架适配
- 在 BERT-base-chinese 上添加对比学习头
- 句子级对比: 同一话题的 Human vs AI 文本作为正负样本对
- 利用 `paired_v3` 数据集 (11,623 条配对数据) 作为天然对比学习样本
- 文档级对比: 不同 scenario/domain 的文本作为辅助信号

#### B2. Genre-aware Contrastive Learning
- 参考 PAN 2025 方案，将 `scenario` (education/business/...) 编码为 genre embedding
- 在对比损失中加入 genre 条件：同 genre 不同来源的文本对应更强的对比信号

#### B3. 实现步骤
```python
# 伪代码
class ContrastiveBERT(nn.Module):
    def __init__(self, bert_model):
        self.bert = bert_model
        self.projection = nn.Linear(768, 256)  # 对比投影头
        self.classifier = nn.Linear(768, 2)     # 分类头

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]

        # 对比学习
        projected = F.normalize(self.projection(cls_output), dim=-1)

        # 分类
        logits = self.classifier(cls_output)

        return logits, projected

# 损失函数
loss = classification_loss + alpha * contrastive_loss
```

#### 预估效果: 跨模型泛化提升 5-10%，OOD 场景显著改善

### 方案 C: 多粒度统一模型 (高投入，长期方向)

**目标**: 统一文档级分类和 Span 级边界检测

#### C1. 多任务学习架构
- 共享 BERT encoder
- 任务1: 文档级分类 (Human/AI/Mixed)
- 任务2: Token 级序列标注 (边界检测)
- 任务3: 句子级序列标注 (哪些句子是 AI 写的)

#### C2. 参考 FAID 框架
- 三分类: Human / AI-generated / Human-AI collaborative
- 对应我们的: label=0 / label=1 / C2+C3+C4

#### C3. Span-level Contrastive Calibration
- 参考 Knowledge-Based Systems 2026 论文
- 在 Span 级加入结构校准
- 利用 `[SEP]` 标记作为监督信号

---

## 五、推荐执行路线

### 阶段 1: 紧急修复 (1-2 天)

**重点: 方案 A — 数据增强 + 平衡训练**

1. **补充人类技术文档** (A1)
   - 从 THUCNews + HC3-Chinese 提取技术类人类文本
   - 目标: 3,000-5,000 条高质量人类技术文档

2. **重新平衡训练集**
   - 基于 core_v3 或 merged_v2 + 新增人类技术样本
   - 确保技术领域人类/AI 比例接近 1:1

3. **训练 bert_v8**
   - 配置: epochs=5, batch_size=16, lr=2e-5
   - 验证指标: 整体准确率 + 技术文档召回率

4. **公平评估**
   - 使用 `datasets/eval/fair_test/` 进行多组测试
   - 对比 bert_v3, v4, v6, v8 在技术文档上的表现

### 阶段 2: 对比学习升级 (3-5 天)

**重点: 方案 B — 对比学习**

1. **数据准备**
   - 利用 paired_v3 (11,623 条) 构建对比样本对
   - 生成同话题的 Human-AI 配对数据

2. **实现 ContrastiveBERT**
   - 在 bert_v8 基础上添加对比学习头
   - 多级对比损失: 句子级 + 文档级

3. **训练 bert_v9_contrastive**
   - 两阶段训练: 先分类预训练 → 再对比微调
   - 或联合训练: 分类损失 + α × 对比损失

4. **泛化评估**
   - 使用 M4 数据集中不同模型生成的文本测试 OOD 性能
   - 使用 DACTYL 风格的对抗样本测试鲁棒性

### 阶段 3: 多粒度统一 (1-2 周)

**重点: 方案 C — 统一多粒度**

1. **架构设计**
   - 共享 BERT encoder + 分类头 + Span 头 + 句子头

2. **数据标注**
   - 扩展现有 `[SEP]` 标注数据
   - 添加句子级标注

3. **训练 bert_v10_unified**
   - 多任务学习调参

---

## 六、未充分利用的数据资源

以下数据尚未在最新训练中使用或使用不足：

| 资源 | 样本数 | 潜在用途 |
|------|-------|---------|
| HC3-Chinese 全量 | 12,853 | 人类+ChatGPT 问答对，对比学习天然素材 |
| M4 中文子集 (qazh) | ~6,000 | 多模型(ChatGPT/Davinci)中文文本 |
| DuReader 技术问答 | - | 人类技术文档补充 |
| THUCNews 科技类 | - | 人类技术文档补充 |
| human_supplement | 10,662 | 多样化人类样本 |
| VCSum | 54MB | 视频字幕摘要（口语风格人类数据） |
| defense_focused | 18,566 train | 已证明对技术文档有效的数据 |
| 主训练集.jsonl 元数据 | 31,684 | 含丰富字段(model, scenario, decoding_profile)，可用于条件对比学习 |

---

## 七、评估计划

### 标准评估

| 测试集 | 说明 | 指标 |
|-------|------|------|
| fair_test/core_v1_test_clean | 核心测试集 | Acc, F1, P, R |
| fair_test/independent_data | 独立测试数据 | Acc, F1, P, R |
| fair_test/merged_v2_val_clean | 合并验证集 | Acc, F1, P, R |

### 细分评估

| 维度 | 说明 |
|------|------|
| 按 scenario 分 | education / business / tech / casual 各自准确率 |
| 按文本长度分 | 短文本(<100字) / 中文本(100-500字) / 长文本(>500字) |
| 按生成模型分 | deepseek / llama / gpt 各自检测率 |
| 技术文档专项 | 人类技术文档 + AI 技术文档 的 TP/FP |
| 对抗测试 | paraphrase / 混合改写 后的检测率 |

---

## 八、参考文献

1. DeTeCtive (NeurIPS 2024): https://arxiv.org/abs/2410.20964
2. Span-level Contrastive Detection (KBS 2026): https://doi.org/10.1016/j.knosys.2025.115123
3. FAID (arXiv 2025): https://arxiv.org/abs/2505.14271
4. SENTRA (arXiv 2025): https://arxiv.org/abs/2509.12385
5. DivEye (ICLR 2026): https://arxiv.org/abs/2509.18880
6. Robust Fine-Grained Detection (arXiv 2025): https://arxiv.org/abs/2504.11952
7. DAMAGE (ACL 2025): https://aclanthology.org/2025.genaidetect-1.9/
8. DACTYL (arXiv 2025): https://arxiv.org/abs/2508.00619
9. Comprehensive Review (CMC 2026): https://doi.org/10.32604/cmc.2025.073347
10. PAN 2025 Voight-Kampff: https://link.springer.com/chapter/10.1007/978-3-032-04354-2_21
11. Generalizability Study (UWaterloo 2026): https://uwspace.uwaterloo.ca/items/0a45a910-c13a-4323-9acf-806f9378591e
12. Robustness Survey (MDPI 2025): https://www.mdpi.com/2227-7390/13/13/2145
13. AI Detection Tools 2026: https://www.eweek.com/news/best-ai-detectors-2026/
14. Sentence-Level Segmentation (IJCNLP-AACL 2025): https://aclanthology.org/2025.findings-ijcnlp.48.pdf

---

*此文档由数据集盘点 + 前沿方法搜索自动生成，建议在执行前由人工审核确认。*
