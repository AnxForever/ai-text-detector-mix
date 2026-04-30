# Chinese AI-Generated Text Detection with Boundary Markers

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Three-Set Avg](https://img.shields.io/badge/Three--Set_Avg-98.36%25-brightgreen.svg)](docs/project/DEFENSE_CURRENT_STATUS.md)

> 中文AI文本检测系统 - 基于边界标记的混合文本检测

## 🎯 项目简介

针对中文混合文本（人类+AI）的检测系统，实现了从粗粒度分类到细粒度边界定位的完整解决方案。

### 核心成果
- ✅ **当前推荐模型**: `bert_v10_augmented`
- ✅ **三集平均准确率**: 98.36% (core_v1_test + independent + merged_v2_val)
- ✅ **独立评估集准确率**: 97.69%
- ✅ **边界定位准确率**: 96.69% (Token级)
- ✅ **实际边界误差**: <10字符

### 技术创新
- 🔥 **边界标记机制**: 使用`[SEP]`标记显式标注人类/AI边界
- 🔥 **双层检测架构**: 分类器 + 边界检测器
- 🔥 **Token级精确定位**: 实现细粒度边界检测

📖 **答辩快照（最新）**: [docs/project/DEFENSE_CURRENT_STATUS.md](docs/project/DEFENSE_CURRENT_STATUS.md)
📖 **基线结果（v2阶段）**: [docs/project/FINAL_RESULTS.md](docs/project/FINAL_RESULTS.md)

## 🤗 Hugging Face模型

- 🔥 [BERT分类器（基线发布）](https://huggingface.co/AnxForever/chinese-ai-detector-bert) - 98.71%准确率（基线 V2，HF 托管）
- 🏆 当前生产模型：`models/bert_v11c_boundary_fix`（验证集 98.75% / 独立评估 98.57% / 三集均值 **98.56%**）
- 🎯 [边界检测器](https://huggingface.co/AnxForever/chinese-ai-detector-span) - Token级定位
- 📊 [训练数据集](https://huggingface.co/datasets/AnxForever/chinese-ai-detection-dataset) - 66K样本（基线）

---

## 🚀 快速演示

**在线使用**: [Hugging Face模型](https://huggingface.co/AnxForever/chinese-ai-detector-bert)

**本地运行**:
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

📘 **详细教程**: [QUICKSTART.md](QUICKSTART.md)

---

## 📂 项目结构

```
datacollection/
├── api/                       # FastAPI 服务
├── scripts/                   # 训练/评估/生成/清洗脚本
│   ├── training/
│   ├── evaluation/
│   ├── generation/
│   ├── data_cleaning/
│   └── demo/
├── models/                    # 多代模型（推荐: bert_v10_augmented）
├── datasets/                  # active/eval/mixed/raw/analysis/feedback_loop
├── docs/                      # 项目文档与计划
│   ├── project/
│   └── plans/
└── frontend/                  # 毕设演示前端（Next.js）
```

---

## 📊 核心成果

### 模型性能

| 指标 | 数值（当前推荐） |
|------|------------------|
| 验证集准确率 (V10) | 98.85% |
| 独立评估集 (910) | 97.69% |
| 三集平均 | 98.36% |
| Token分类 (Span) | 96.69% |

### 技术创新

1. **边界标记机制**: 在混合文本边界插入`[SEP]`标记，C2检测提升14%
2. **双层检测架构**: 分类器 + 边界检测器
3. **Token级标注**: 精确定位人类/AI边界

---

## 📝 数据集统计

| 数据集 | 样本数 | 说明 |
|--------|--------|------|
| core_v1 | 58,563 | 基线训练/验证/测试 |
| merged_v2 | 69,347 | v8/v9/v10主数据池 |
| train_v10 | 62,980 | v10训练集 |
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

- 当前答辩口径: [docs/project/DEFENSE_CURRENT_STATUS.md](docs/project/DEFENSE_CURRENT_STATUS.md)
- 完整成果（基线）: [docs/project/FINAL_RESULTS.md](docs/project/FINAL_RESULTS.md)
- 训练计划: [docs/project/TRAINING_PLAN.md](docs/project/TRAINING_PLAN.md)
- 评估报告: `evaluation_results/final_report.txt`

---

*最后更新: 2026-02-12*
