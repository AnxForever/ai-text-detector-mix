[根目录](../CLAUDE.md) > **datasets**

# 数据集模块

## 模块职责

存储和管理项目所有数据集，包括训练数据、评估数据、混合文本数据和原始数据源。

## 数据集组织

### 活跃数据集 (推荐使用)

| 数据集 | 路径 | 说明 |
|-------|------|------|
| **core_v1** | `datasets/active/core_v1/` | 主训练集 (66,001条) |

### 评估数据集

| 数据集 | 路径 | 说明 |
|-------|------|------|
| eval_splits_v1 | `datasets/eval/splits/v1/` | ID/OOD/Mixed评估拆分 |

### 混合文本数据

| 数据集 | 路径 | 说明 |
|-------|------|------|
| mixed_candidates | `datasets/mixed/candidates/` | 混合测试候选 |
| mixed_sources | `datasets/mixed/hybrid/` | 混合文本源 (C2/C3/C4) |

### 分析数据

| 数据集 | 路径 | 说明 |
|-------|------|------|
| analysis_classified | `datasets/analysis/classified/` | 规则分类结果 |
| analysis_routed | `datasets/analysis/routed/` | 路由池 (core/hard/review/reject) |
| analysis_pred_probs | `datasets/analysis/pred_probs/` | 预测概率输出 |

### 原始数据

| 数据集 | 路径 | 说明 |
|-------|------|------|
| raw_sources | `datasets/raw/` | 原始数据源 (MGTBench, HC3等) |

## 核心数据集详情

### core_v1 (主训练集)

```
datasets/active/core_v1/
├── train.csv        # 训练集 (52,800条)
├── val.csv          # 验证集 (6,600条)
├── test.csv         # 测试集 (6,601条)
├── full_dataset.csv # 完整数据
├── all_human.csv    # 纯人类文本
├── all_ai.csv       # 纯AI文本
├── merge_log.json   # 合并日志
└── README.md
```

### 数据格式

CSV 格式，主要字段:
- `text`: 文本内容
- `label`: 标签 (0=Human, 1=AI)
- `category`: 类别 (可选: C2/C3/C4/Human)
- `source`: 数据来源

### 混合数据 (hybrid)

```
datasets/mixed/hybrid/
├── c2_*.json        # C2续写数据
├── c3_*.json        # C3改写数据
├── c4_*.json        # C4润色数据
├── merged_all.json  # 合并数据
└── multimodel/      # 多模型生成数据
```

## 数据注册表

`datasets/registry.json` 包含所有数据集的元信息:

```json
{
  "entries": [
    {
      "name": "core_v1",
      "category": "active_train_candidates",
      "path": "datasets/active/core_v1",
      "recommended": "true"
    }
  ]
}
```

## 常用操作

### 加载训练数据

```python
import pandas as pd

train_df = pd.read_csv('datasets/active/core_v1/train.csv', encoding='utf-8-sig')
val_df = pd.read_csv('datasets/active/core_v1/val.csv', encoding='utf-8-sig')
test_df = pd.read_csv('datasets/active/core_v1/test.csv', encoding='utf-8-sig')
```

### 加载混合数据

```python
import json

with open('datasets/mixed/hybrid/merged_all.json', 'r', encoding='utf-8') as f:
    hybrid_data = json.load(f)
```

## 数据统计

| 数据集 | 样本数 | Human | AI |
|-------|-------|-------|-----|
| core_v1 训练集 | 52,800 | ~26,400 | ~26,400 |
| core_v1 验证集 | 6,600 | ~3,300 | ~3,300 |
| core_v1 测试集 | 6,601 | ~3,300 | ~3,301 |
| 混合数据 | 7,563 | 1,500 | 6,063 |

## 常见问题 (FAQ)

**Q: 应该使用哪个数据集训练?**
A: 使用 `datasets/active/core_v1/`，这是经过清洗和去重的推荐数据集。

**Q: 混合数据中C2/C3/C4是什么?**
A:
- C2 (续写): 人类开头 + AI续写
- C3 (改写): AI改写人类文本
- C4 (润色): AI润色人类文本

**Q: 如何添加新数据?**
A: 使用 `scripts/data_cleaning/` 中的脚本处理，然后更新 `registry.json`。

## 相关文件清单

```
datasets/
├── README.md              # 数据集索引说明
├── registry.json          # 数据集注册表
├── active/                # 活跃训练数据
│   └── core_v1/
├── eval/                  # 评估数据
│   └── splits/
├── mixed/                 # 混合文本数据
│   ├── candidates/
│   └── hybrid/
├── analysis/              # 分析输出
├── raw/                   # 原始数据源
├── generated/             # 生成数据
├── planning/              # 计划输出
├── logs/                  # 日志
├── samples/               # 样本
└── archive/               # 归档数据
```

## 变更记录 (Changelog)

### 2026-01-28
- 初始化模块文档

---

*文档生成时间: 2026-01-28T12:42:53+0800*
