[根目录](../CLAUDE.md) > **scripts**

# 脚本模块

## 模块职责

包含项目所有 Python 脚本，分为训练、评估、生成、数据清洗、演示和工具六大子模块。

## 子模块概览

| 子模块 | 职责 | 脚本数量 |
|-------|------|---------|
| `training/` | 模型训练 | 12 |
| `evaluation/` | 模型评估 | 19 |
| `generation/` | AI文本生成 | 29 |
| `data_cleaning/` | 数据清洗处理 | 18 |
| `demo/` | 可视化演示 | 1 |
| `utils/` | 工具函数 | 1 |

## 入口与启动

### 训练脚本

```bash
# BERT改进版训练
python scripts/training/train_bert_improved.py --epochs 5 --batch_size 16

# 边界检测器训练
python scripts/training/train_span_detector.py --epochs 10

# BiGRU变体
python scripts/training/train_bert_bigru.py --epochs 5

# DPCNN变体
python scripts/training/train_dpcnn.py --epochs 5
```

### 评估脚本

```bash
# 完整评估
python scripts/evaluation/eval_complete.py

# 交互式测试
python scripts/evaluation/test_single_text.py --interactive

# 综合评估
python scripts/evaluation/comprehensive_eval.py

# 生成报告
python scripts/evaluation/generate_report.py
```

### 数据处理脚本

```bash
# 添加SEP标记
python scripts/data_cleaning/add_sep_markers.py

# 准备Span标签
python scripts/data_cleaning/prepare_span_labels.py

# 重建combined v2
python scripts/data_cleaning/rebuild_combined_v2.py
```

### 演示脚本

```bash
# 可视化检测演示
python scripts/demo/visualize_detection.py
```

## 关键依赖与配置

### 公共依赖

```python
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report
```

### 设备配置

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

## 数据模型

### AIDetectionDataset

训练数据集类，支持:
- 动态padding
- 长度加权采样
- 多种标签格式

### BERTTrainer

训练器类，参数:
- `model_name`: 预训练模型
- `max_length`: 最大序列长度 (默认512)
- `batch_size`: 批次大小 (默认16)
- `learning_rate`: 学习率 (默认2e-5)
- `num_epochs`: 训练轮数 (默认5)

## 测试与质量

### 测试目录

- `scripts/data_cleaning/tests/`
- `scripts/evaluation/tests/`
- `scripts/generation/tests/`
- `scripts/training/tests/`

> 注: 当前测试目录存在但可能为空

## 常见问题 (FAQ)

**Q: 导入错误 "No module named scripts.bert_prep"?**
A: 确保在项目根目录运行脚本，并激活虚拟环境。

**Q: CUDA内存不足?**
A: 减小 `batch_size` 或使用 `--max_length 256`。

**Q: 训练数据路径错误?**
A: 检查 `datasets/active/core_v1/` 目录是否存在训练集文件。

## 相关文件清单

```
scripts/
├── training/           # 训练脚本
│   ├── train_bert_improved.py      # 主训练脚本
│   ├── train_span_detector.py      # 边界检测器
│   ├── train_bert_bigru.py         # BiGRU变体
│   ├── train_dpcnn.py              # DPCNN变体
│   └── length_weighted_loss.py     # 损失函数
├── evaluation/         # 评估脚本
│   ├── eval_complete.py            # 完整评估
│   ├── test_single_text.py         # 单文本测试
│   └── comprehensive_eval.py       # 综合评估
├── generation/         # 生成脚本
│   ├── scenario_fill_generate.py   # 场景填充生成
│   ├── gen_multimodel.py           # 多模型生成
│   └── batch_hybrid_gen.py         # 批量混合生成
├── data_cleaning/      # 数据清洗
│   ├── add_sep_markers.py          # 添加SEP标记
│   ├── prepare_span_labels.py      # Span标签准备
│   └── merge_batch_data.py         # 数据合并
├── demo/               # 演示
│   └── visualize_detection.py      # 可视化演示
└── utils/              # 工具
    └── api_config.py               # API配置加载
```

## 变更记录 (Changelog)

### 2026-01-28
- 初始化模块文档

---

*文档生成时间: 2026-01-28T12:42:53+0800*
