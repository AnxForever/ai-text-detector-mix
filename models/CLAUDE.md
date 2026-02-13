[根目录](../CLAUDE.md) > **models**

# 模型模块

## 模块职责

存储训练好的 BERT 模型文件，包括分类器和边界检测器。

**重要**: 此目录为只读，不要删除模型文件!

## 模型列表

### 当前推荐模型

| 模型目录 | 用途 | 验证准确率 | 训练样本 | 状态 |
|---------|------|-----------|---------|------|
| `bert_v11c_boundary_fix/` | **推荐分类器** | 98.75% | 63,113 | 生产就绪 |
| `bert_v10_augmented/` | 上一代分类器 | 98.85% | 62,899 | 已被 V11c 取代 |
| `bert_span_detector/` | 边界检测器 | 96.69% (Token) | - | 生产 |

### 模型演进历史

| 版本 | 验证准确率 | 训练样本 | 独立评估集 | 三集平均 | 技术文档检测 |
|-----|-----------|---------|-----------|---------|-------------|
| bert_v2_with_sep | 98.71% | ~66,000 | - | - | 正确 |
| bert_v3_core_v2 | 99.57% | 57,435 | - | - | 正确 |
| bert_v4_defense_focused | 99.75% | 6,533 | - | - | 正确 |
| bert_v5_paired | 100%* | 7,800 | - | - | 漏检 |
| bert_v6_merged | 97.94% | 54,649 | 93.19% | 95.85% | 漏检 |
| bert_v7_improved | 98.44% | 60,440 | 94.07% | 96.81% | 漏检 |
| bert_v8_calibrated | 98.39% | 60,440 | 94.40% | 96.88% | 漏检 |
| bert_v9_p0_supplement | 98.85% | 61,872 | 94.73% | 97.37% | 部分漏检 |
| **bert_v10_augmented** | **98.85%** | **62,899** | **97.69%** | **98.36%** | **已修复** |
| **bert_v11c_boundary_fix** | **98.75%** | **63,113** | **98.57%** | **98.56%** | **已修复** |

*独立评估集 (910 条含真实 LLM 输出)，三集平均含 core_v1_test_clean + independent_data + merged_v2_val_clean

## 已知问题

### 技术文档漏检问题——V10 已修复

V5-V9 模型对 AI 生成的技术/学术内容存在漏检。V10 通过定向数据增强（+500 教育类 AI、+500 人类技术文本、+300 多样化短文本）彻底解决。

**各模型在真实 AI 技术文本上的表现** (independent_data, 910 样本):

| 来源模型 | 样本数 | V8 检出率 | V9 检出率 | V10 检出率 | **V11c 检出率** |
|---------|-------|----------|----------|-----------|---------------|
| GPT-5 | 8 | 100% | 75% | 100% | **100%** |
| DeepSeek-v3.2 | 8 | 100% | 75% | 100% | **100%** |
| Gemini-3-flash | 16 | 68.8% | 93.8% | 100% | **100%** |
| Gemini-3-pro-search | 8 | 100% | 87.5% | 100% | **87.5%** |
| GPT-OSS-120B | 8 | 100% | 87.5% | 100% | **100%** |
| LLaMA-3.1-405B | 9 | 88.9% | 88.9% | 88.9% | **100%** |

### 残余问题

- Gemini-3-pro-search: V11c 有 1/8 新回归（低置信度边界样本 conf=0.61）
- m4_chatgpt 有 1/50 漏检（98%，跨版本稳定）
- FP 误报 13 条（较 V10 的 21 条减少 38%）

## bert_v11c_boundary_fix 详细信息（推荐部署）

**创建时间**: 2026-02-13

### 训练策略

**V11c 风险治理路线**: V10 数据经过四阶段清洗与增补:
1. **A1 风险审计**: 移除 750 条硬编码模板匹配样本
2. **A1 unknown 分流**: 移除 1,767 条无法追溯来源的样本
3. **B2 弱域增补**: 补充 300 formal_collected + 300 LLaMA-405B 样本
4. **B2 长文 AI 边界修复**: 补充 2,131 条 256+ 字符 AI 样本（恢复 V10 长文覆盖）

数据规模: V10 62,980 → V11a 60,456 → V11b 61,056 → V11c 63,187 (清洗后 63,113)

### 训练配置
- 基础模型: bert_v7_improved (fine-tune)
- batch_size: 8, accum_steps: 4 (有效 batch=32)
- max_length: 256
- epochs: 4 (Early Stopping patience=2, best at Epoch 2)
- learning_rate: 1e-05
- label_smoothing: 0.05
- length_aware_loss: 0.1
- min_text_length: 10
- 训练样本: 63,113 (AI: 32,744, Human: 30,369)

### 训练过程
| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val F1 |
|-------|-----------|-----------|----------|---------|--------|
| 1 | 0.1235 | 99.55% | 0.0858 | 97.68% | 0.9785 |
| **2** | **0.1100** | **99.81%** | **0.0621** | **98.75%** | **0.9883** |
| 3 | 0.1080 | 99.91% | 0.0680 | 98.54% | 0.9863 |
| 4 | 0.1078 | 99.91% | 0.0749 | 98.23% | 0.9835 |

**最佳模型**: Epoch 2 (val_acc=98.75%, val_f1=0.9883)

### 评估结果 (V10 → V11c 对比)

| 评估集 | V10 | **V11c** | Delta |
|-------|-----|---------|-------|
| core_v1_test_clean (545) | 98.35% | **97.98%** | -0.37% |
| independent_data (910) | 97.69% | **98.57%** | **+0.88%** |
| merged_v2_val_clean (1144) | 99.04% | **99.13%** | +0.09% |
| **三集平均** | 98.36% | **98.56%** | **+0.20%** |

### 关键改进（V10 → V11c）

| 指标 | V10 | V11c | 变化 |
|------|----|----|------|
| 独立评估集准确率 | 97.69% | 98.57% | **+0.88%** |
| 三集平均 | 98.36% | 98.56% | **+0.20%** |
| independent 总错误 | 21 | 13 | **-38%** |
| LLaMA-405B 检出率 | 88.9% | 100% | **+11.1%** |
| formal_collected 正确率 | 96.0% | 96.5% | **+0.5%** |
| ECE（校准误差） | 0.0058 | 0.0034 | **-41%** |

### 推理配置
```python
MODEL_PATH = "models/bert_v11c_boundary_fix"
MAX_LENGTH = 256
TEMPERATURE = 0.8165  # Temperature Scaling (910 样本校准)
```

## bert_v10_augmented 详细信息（已被 V11c 取代）

**创建时间**: 2026-02-12

### 训练策略

**方案 α 纯数据增强**：在 V9 基础上针对性补充训练数据，不改模型架构或损失函数。

新增数据（共 1,296 条清洗后入库）：
- +500 教育类 AI 文本（技术教程、学术论文风格）
- +500 人类技术文本（真实技术博客、学术笔记）
- +300 多样化短文本（<128 字符人类口语/日常）

### 训练配置
- 基础模型: bert_v7_improved (fine-tune)
- batch_size: 8, accum_steps: 4 (有效 batch=32)
- max_length: 256
- epochs: 4 (Early Stopping patience=2, best at Epoch 2)
- learning_rate: 1e-05
- label_smoothing: 0.05
- length_aware_loss: 0.1
- min_text_length: 10
- 训练样本: 62,899 (AI: 32,815, Human: 30,084)

### 训练过程
| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val F1 |
|-------|-----------|-----------|----------|---------|--------|
| 1 | 0.1215 | 99.65% | 0.0614 | 98.63% | 0.9872 |
| **2** | **0.1094** | **99.84%** | **0.0581** | **98.85%** | **0.9892** |
| 3 | 0.1078 | 99.91% | 0.0811 | 98.00% | 0.9814 |
| 4 | 0.1068 | 99.96% | 0.0876 | 98.04% | 0.9818 |

**最佳模型**: Epoch 2 (val_acc=98.85%, val_f1=0.9892)

### 评估结果 (五代对比)

| 评估集 | V6 | V7 | V8 | V9 | **V10** |
|-------|----|----|----|----|---------|
| core_v1_test (545) | 96.88% | 97.98% | 97.61% | 98.35% | **98.35%** |
| independent (910) | 93.19% | 94.07% | 94.40% | 94.73% | **97.69%** |
| merged_v2_val (1144) | 97.73% | 98.60% | 98.78% | 98.86% | **99.04%** |
| **三集平均** | 95.93% | 96.88% | 96.93% | 97.31% | **98.36%** |

*评估集已去除与训练集重叠的样本 (core_v1: -115条, merged_v2: -41条)，independent_data 无泄露

### 关键改进（V9 → V10）

| 指标 | V9 | V10 | 变化 |
|------|----|----|------|
| 独立评估集准确率 | 94.73% | 97.69% | **+2.96%** |
| 三集平均 | 97.31% | 98.36% | **+1.05%** |
| FN（漏检） | 10 | 3 | **-70%** |
| FP（误报） | 38 | 18 | **-53%** |
| 高置信错误 | 45 | 19 | **-58%** |
| GPT-5 检出率 | 75% | 100% | **+25%** |
| DeepSeek 检出率 | 75% | 100% | **+25%** |
| ECE（校准误差） | 0.0223 | 0.0112 | **-50%** |

### 推理配置
```python
MODEL_PATH = "models/bert_v10_augmented"
MAX_LENGTH = 256
TEMPERATURE = 0.8931  # Temperature Scaling (910 样本校准)
```

## 模型加载

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 加载推荐分类器 (V11c)
tokenizer = BertTokenizer.from_pretrained('models/bert_v11c_boundary_fix')
model = BertForSequenceClassification.from_pretrained('models/bert_v11c_boundary_fix')

# 推理 (带 Temperature Scaling)
inputs = tokenizer(text, max_length=256, padding='max_length', truncation=True, return_tensors='pt')
logits = model(**inputs).logits
scaled_probs = torch.softmax(logits / 0.8165, dim=-1)
```

## 相关文件清单

```
models/
├── bert_v2_with_sep/           # 基准分类器 (98.71%)
├── bert_v3_core_v2/            # 新场景版 (99.57%)
├── bert_v4_defense_focused/    # 防御增强版 (99.75%)
├── bert_v5_paired/             # 配对数据版 (100%*)
├── bert_v6_merged/             # 合并版 (95.85% 三集)
├── bert_v7_improved/           # 改进版 (96.81% 三集)
├── bert_v8_calibrated/         # 校准版 (96.88% 三集)
├── bert_v9_p0_supplement/      # P0增强版 (97.37% 三集)
├── bert_v10_augmented/         # 上一代 (98.36% 三集) 技术文档漏检已修复
├── bert_v11c_boundary_fix/    # ★推荐★ (98.56% 三集) 风险治理+边界修复
├── bert_span_detector/         # 边界检测器 (96.69%)
└── bert_improved/              # 早期改进版模型
```

## 变更记录 (Changelog)

### 2026-02-13
- 添加 bert_v11c_boundary_fix (风险治理路线 V11c)
- 推荐模型更新为 bert_v11c_boundary_fix (三集平均 98.56%)
- LLaMA-405B 漏检问题已修复: 88.9%→100%
- 独立评估集总错误减少 38%: 21→13
- Temperature Scaling: T=0.8165, ECE=0.0034
- 独立评估集提升: 97.69%→98.57% (+0.88%)

### 2026-02-12
- 添加 bert_v10_augmented (方案 α 纯数据增强)
- 推荐模型更新为 bert_v10_augmented (三集平均 98.34%)
- 技术文档漏检问题已修复: GPT-5 75%→100%, DeepSeek 75%→100%
- Temperature Scaling: T=0.8931, ECE=0.0058
- 独立评估集提升: 94.73%→97.69% (+2.96%)

### 2026-02-11
- 添加 bert_v7_improved / bert_v8_calibrated / bert_v9_p0_supplement
- 推荐模型更新为 bert_v9_p0_supplement (三集平均 97.37%)
- Temperature Scaling 重新校准: T=2.1525 → T=1.1363 (基于 910 真实样本)
- 更新技术文档漏检分析: V9 仍有 6/8 错判为技术类内容
- 独立评估集三轮修复完成 (1,139 → 910 条)

### 2026-02-09
- 添加 bert_v6_merged 模型
- 发现技术文档漏检问题

### 2026-01-28
- 初始化模块文档

---

*文档更新时间: 2026-02-13*
