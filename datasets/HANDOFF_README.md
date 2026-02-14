# AI文本检测数据集 - 交接文档

> 作者: [您的名字]
> 更新日期: 2026-01-29
> 项目: 中文AI/Human文本二分类检测

---

## 一、项目概述

本项目构建了用于检测中文文本是否由AI生成的数据集。核心任务是二分类：
- **Human (0)**: 人类撰写的文本
- **AI (1)**: 由大语言模型生成的文本

### 项目成果

| 指标 | 数值 |
|-----|------|
| 最佳模型准确率 | 98.57% |
| 训练数据规模 | 57,619 条 |
| 数据来源数量 | 9 个独立来源 |
| 高难度测试场景 | 11 个 |

---

## 二、数据集总览

### 2.1 推荐使用的数据集

```
datasets/bert_v2_overnight/
├── train_balanced.csv   # 训练集 (57,619 条) ⭐推荐
├── val_balanced.csv     # 验证集 (7,202 条)  ⭐推荐
├── test_balanced.csv    # 测试集 (7,203 条)  ⭐推荐
└── difficult_tests/     # 高难度测试集 (11个场景)
```

### 2.2 数据统计

| 数据集 | 总样本数 | Human | AI | Human平均长度 | AI平均长度 |
|-------|---------|-------|-----|-------------|-----------|
| 训练集 (平衡版) | 57,619 | 28,809 | 28,810 | 415字符 | 411字符 |
| 验证集 (平衡版) | 7,202 | 3,601 | 3,601 | 406字符 | 410字符 |
| 测试集 (平衡版) | 7,203 | 3,602 | 3,601 | 425字符 | 407字符 |

**关键特性**: Human/AI 平均长度差仅 **3-18字符**，有效避免了长度偏差问题。

---

## 三、数据来源详解

### 3.1 来源分布 (训练集)

| 来源 | 样本数 | 类型 | 说明 |
|-----|-------|------|------|
| **core_v1** | 14,658 | Human | 项目核心收集的人类文本 |
| **HC3-Chinese-ChatGPT** | 13,181 | AI | HC3数据集中ChatGPT生成的回答 |
| **generated_v2** | 11,116 | AI | 本项目生成的AI文本 (多模型) |
| **HC3-Chinese** | 8,121 | Human | HC3数据集中人类撰写的回答 |
| **M4-qazh** | 3,141 | Human | M4数据集中文问答部分 |
| **LCSTS** | 2,739 | Human | 大规模中文短文本摘要数据集 |
| **M4-chatgpt** | 2,273 | AI | M4数据集中ChatGPT生成文本 |
| **M4-davinci** | 2,240 | AI | M4数据集中GPT-3 davinci文本 |
| **VCSum** | 150 | Human | 视频评论摘要数据集 |

### 3.2 数据来源说明

#### HC3-Chinese (Human ChatGPT Comparison Corpus)
- **论文**: "How Close is ChatGPT to Human Experts?"
- **链接**: https://github.com/Hello-SimpleAI/chatgpt-comparison-detection
- **包含领域**: 法律、医学、金融、心理学、百科、开放问答

#### M4 (Multi-generator, Multi-domain, Multi-lingual)
- **论文**: "M4: Multi-generator, Multi-domain, and Multi-lingual Black-Box Machine-Generated Text Detection"
- **链接**: https://github.com/mbzuai-nlp/M4
- **包含模型**: ChatGPT, GPT-3 davinci, 多种语言

#### LCSTS (Large-scale Chinese Short Text Summarization)
- **来源**: 新浪微博
- **特点**: 短文本，摘要风格

#### VCSum
- **来源**: 视频评论
- **特点**: 口语化，短句

---

## 四、数据格式

### 4.1 CSV 格式

```csv
text,label,source
"这是一段示例文本...",0,core_v1
"AI生成的内容示例...",1,HC3-Chinese-ChatGPT
```

**字段说明**:
- `text`: 文本内容 (字符串)
- `label`: 标签，0=Human, 1=AI (整数)
- `source`: 数据来源标识 (字符串)

### 4.2 JSONL 格式 (高难度测试集)

```json
{"text": "文本内容...", "label": 0, "source": "core_v1"}
```

---

## 五、高难度测试集

位置: `datasets/bert_v2_overnight/difficult_tests/`

### 5.1 长度相关测试

| 测试集 | 样本数 | 目的 |
|-------|-------|------|
| **length_trap.jsonl** | 600 | 短AI + 长Human (测试长度捷径) |
| **reverse_length.jsonl** | 400 | 长AI + 短Human (反向验证) |
| **very_short.jsonl** | 400 | 极短文本 (<80字符) |
| **boundary_length.jsonl** | 400 | 边界长度 (500-1000字符) |

### 5.2 来源分层测试

| 测试集 | 样本数 | 类型 |
|-------|-------|------|
| source_HC3-Chinese.jsonl | 200 | Human |
| source_VCSum.jsonl | 200 | Human |
| source_LCSTS.jsonl | 200 | Human |
| source_M4-qazh.jsonl | 200 | Human |
| source_HC3-Chinese-ChatGPT.jsonl | 200 | AI |
| source_M4-chatgpt.jsonl | 200 | AI |
| source_M4-davinci.jsonl | 200 | AI |

---

## 六、历史数据集 (参考)

### 6.1 core_v1 (原始数据)

```
datasets/active/core_v1/
├── train.csv    # 46,849 条 (有长度偏差)
├── val.csv      # 验证集
├── test.csv     # 5,858 条
└── README.md
```

**注意**: core_v1 存在严重的**长度偏差问题**:
- Human 平均长度: 434 字符
- AI 平均长度: 794 字符
- 差距: 360 字符

这导致模型可能学习"短文本=Human, 长文本=AI"的捷径特征。

### 6.2 外部数据集 (原始)

```
datasets/external/
├── M4/          # M4 多语言数据集
├── DuReader/    # 阅读理解数据集
└── VCSum/       # 视频评论摘要
```

---

## 七、模型迭代记录

### V1: bert_v2_overnight (2026-01-28)

- **问题**: 长度偏差严重 (Human 438字符 vs AI 910字符)
- **结果**: 同分布 99.92%, 但高难度场景差 (部分 <80%)

### V2: bert_v2_balanced (2026-01-29) ✅ 当前版本

- **改进**: 长度平衡 (Human 415字符 vs AI 412字符)
- **结果**: 所有高难度场景 >97%

| 场景 | V1 → V2 |
|-----|---------|
| 长度陷阱 | 89.83% → **98.83%** |
| 反向长度 | 80.00% → **99.00%** |
| 极短文本 | 87.75% → **98.00%** |
| M4-davinci | 47.70% → **100%** 🔥 |

---

## 八、使用指南

### 8.1 加载数据

```python
import pandas as pd

# 推荐: 使用平衡数据集
train = pd.read_csv('datasets/bert_v2_overnight/train_balanced.csv',
                    encoding='utf-8-sig')
val = pd.read_csv('datasets/bert_v2_overnight/val_balanced.csv',
                  encoding='utf-8-sig')
test = pd.read_csv('datasets/bert_v2_overnight/test_balanced.csv',
                   encoding='utf-8-sig')

print(f"训练集: {len(train)} 条")
print(f"标签分布: {train['label'].value_counts().to_dict()}")
```

### 8.2 加载高难度测试集

```python
import json

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

length_trap = load_jsonl('datasets/bert_v2_overnight/difficult_tests/length_trap.jsonl')
```

### 8.3 模型训练

```bash
# 使用平衡数据集训练
python scripts/training/train_balanced.py

# 完整评估
python scripts/evaluation/eval_complete.py
```

---

## 九、重要提醒

### ⚠️ 必须注意

1. **优先使用 `bert_v2_overnight/` 下的平衡数据集**
2. **不要直接使用 core_v1，存在长度偏差**
3. **评估时必须测试高难度场景，不能只看总准确率**

### 💡 最佳实践

1. 长度平衡是关键 - Human/AI 平均长度差应 <50字符
2. 表面指标会骗人 - 99%准确率可能只是学了捷径
3. 必须做 OOD (Out-of-Distribution) 测试
4. 按特征分组评估 (长度、来源)

---

## 十、文件清单

```
datasets/
├── bert_v2_overnight/           # ⭐ 主要数据集
│   ├── train_balanced.csv       # 训练集
│   ├── val_balanced.csv         # 验证集
│   ├── test_balanced.csv        # 测试集
│   └── difficult_tests/         # 高难度测试
│       ├── length_trap.jsonl
│       ├── reverse_length.jsonl
│       ├── very_short.jsonl
│       ├── boundary_length.jsonl
│       ├── source_*.jsonl
│       └── index.json
├── active/core_v1/              # 原始数据 (有偏差)
├── external/                    # 外部数据集原始文件
├── raw/                         # 原始数据源
│   ├── HC3-Chinese/            # HC3中文数据
│   └── human_texts/            # 人类文本收集
├── mixed/hybrid/                # 混合文本数据 (C2/C3/C4)
├── registry.json                # 数据集注册表
└── HANDOFF_README.md            # 本文档
```

---

## 十一、联系方式

如有问题，请联系:
- [您的邮箱]
- [您的微信/其他联系方式]

---

*文档生成时间: 2026-01-29*
