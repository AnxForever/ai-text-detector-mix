# Chinese AI-Generated Text Detection with Boundary Markers

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Accuracy](https://img.shields.io/badge/Accuracy-98.71%25-brightgreen.svg)](FINAL_RESULTS.md)

> 中文AI文本检测系统 - 基于边界标记的混合文本检测

## 🎯 项目简介

针对中文混合文本（人类+AI）的检测系统，实现了从粗粒度分类到细粒度边界定位的完整解决方案。

### 核心成果
- ✅ **整体准确率**: 98.71%
- ✅ **C2混合文本检测**: 93.84% (提升14%)
- ✅ **边界定位准确率**: 96.69% (Token级)
- ✅ **实际边界误差**: <10字符

### 技术创新
- 🔥 **边界标记机制**: 使用`[SEP]`标记显式标注人类/AI边界
- 🔥 **双层检测架构**: 分类器 + 边界检测器
- 🔥 **Token级精确定位**: 实现细粒度边界检测

📖 **完整成果**: [FINAL_RESULTS.md](FINAL_RESULTS.md) | **数据模型**: [DATA_AND_MODELS.md](DATA_AND_MODELS.md)

---

## 🚀 快速演示

```bash
cd /mnt/c/datacollection
source .venv/bin/activate
export HF_HUB_OFFLINE=1

# 运行可视化演示
python scripts/demo/visualize_detection.py

# 查看完整评估
python scripts/evaluation/eval_complete.py

# 生成评估报告
python scripts/evaluation/generate_report.py
```

---

## 📂 项目结构

```
datacollection/
├── 📖 核心文档
│   ├── README.md              # 本文档
│   ├── FINAL_RESULTS.md       # 最终成果报告 ⭐
│   ├── TRAINING_PLAN.md       # 训练计划
│   └── API_KEYS.md            # API配置
│
├── 🤖 模型 (779MB)
│   ├── bert_v2_with_sep/      # 主分类器 (98.71%准确率)
│   └── bert_span_detector/    # 边界检测器 (96.69%准确率)
│
├── 📊 数据集 (575MB)
│   ├── combined_v2/           # 训练数据 (66,001条)
│   ├── hybrid/                # 混合数据 (7,563条)
│   └── final_clean/           # 基础数据 (55,438条)
│
├── 🛠️ 脚本
│   ├── training/              # 训练脚本
│   ├── evaluation/            # 评估脚本
│   ├── demo/                  # 演示脚本 ⭐
│   └── data_cleaning/         # 数据处理
│
└── 📈 结果
    ├── evaluation_results/    # 评估报告
    └── logs/                  # 训练日志
```

---

## 📊 核心成果

### 模型性能

| 指标 | 数值 |
|------|------|
| 整体准确率 | 98.71% |
| C2 (续写) | 93.84% |
| C3 (改写) | 100% |
| C4 (润色) | 92.89% |
| Token分类 | 96.69% |

### 技术创新

1. **边界标记机制**: 在混合文本边界插入`[SEP]`标记，C2检测提升14%
2. **双层检测架构**: 分类器 + 边界检测器
3. **Token级标注**: 精确定位人类/AI边界

---

## 📝 数据集统计

| 数据集 | 样本数 | 说明 |
|--------|--------|------|
| Combined v2 | 66,001 | 训练/验证/测试 |
| 混合数据 | 7,563 | C2/C3/C4/Human |
| Span标注 | 2,034 | Token级标注 |

---

## 🎬 演示效果

运行 `python scripts/demo/visualize_detection.py` 查看：
- 分类结果（Human/AI + 置信度）
- 边界位置检测
- 文本分段展示

**实际效果**:
- 示例1: 边界62字符 → 检测62字符 ✅
- 示例2: 边界62字符 → 检测61字符 ✅
- 示例3: 边界154字符 → 检测162字符 ✅

---

## 📞 更多信息

- 完整成果: [FINAL_RESULTS.md](FINAL_RESULTS.md)
- 训练计划: [TRAINING_PLAN.md](TRAINING_PLAN.md)
- 评估报告: `evaluation_results/final_report.txt`

---

*最后更新: 2026-01-26*
