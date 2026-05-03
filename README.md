# Chinese AI-Generated Text Detection

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Three-Set Avg](https://img.shields.io/badge/Three--Set_Avg-98.36%25-brightgreen.svg)](docs/project/DEFENSE_CURRENT_STATUS.md)

> 中文AI文本检测系统 - 当前线上以 V11c 二分类模型区分 Human / AI

## 🎯 项目简介

面向中文场景的 AI 生成文本检测系统，当前线上主链路使用 `bert_v11c_boundary_fix` 输出 Human / AI 二分类结果。混合文本边界检测做过实验，但因样本规模和真实分布不足，当前不作为线上能力启用，也不返回 `mixed`。

### 核心成果
- ✅ **当前推荐模型**: `bert_v11c_boundary_fix`
- ✅ **三集平均准确率**: 98.56%
- ✅ **独立评估集准确率**: 98.57%
- ✅ **线上输出口径**: Human / AI 二分类
- ✅ **置信度校准**: Temperature Scaling `T=0.8165`，ECE=0.0034

### 技术创新
- 🔥 **风险治理数据清洗**: 移除模板样本与 unknown 样本，降低虚高与过拟合风险
- 🔥 **弱域与长文补强**: 补充 formal、LLaMA-405B 与长文本 AI 样本
- 🔥 **置信度校准**: 使用 Temperature Scaling 改善线上置信度解释

📖 **答辩快照（最新）**: [docs/project/DEFENSE_CURRENT_STATUS.md](docs/project/DEFENSE_CURRENT_STATUS.md)
📖 **基线结果（v2阶段）**: [docs/project/FINAL_RESULTS.md](docs/project/FINAL_RESULTS.md)

## 🤗 Hugging Face模型

- 🔥 [BERT分类器（基线发布）](https://huggingface.co/AnxForever/chinese-ai-detector-bert) - 98.71%准确率（基线 V2，HF 托管）
- 🏆 当前生产模型：`models/bert_v11c_boundary_fix`（验证集 98.75% / 独立评估 98.57% / 三集均值 **98.56%**）
- 🎯 [边界检测器](https://huggingface.co/AnxForever/chinese-ai-detector-span) - 历史实验模型，当前线上默认不启用
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
├── models/                    # 多代模型（推荐: bert_v11c_boundary_fix）
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
| 验证集准确率 (V11c) | 98.75% |
| 独立评估集 (910) | 98.57% |
| 三集平均 | 98.56% |
| 输出类型 | Human / AI |

### 技术创新

1. **数据风险治理**: 移除模板与 unknown 来源样本，提升独立评估稳定性
2. **弱域补强**: 针对 formal 与 LLaMA-405B 等弱项补充样本
3. **校准部署**: 使用 Temperature Scaling 和人工反馈闭环支撑线上解释

---

## 📝 数据集统计

| 数据集 | 样本数 | 说明 |
|--------|--------|------|
| core_v1 | 58,563 | 基线训练/验证/测试 |
| merged_v2 | 69,347 | v8/v9/v10主数据池 |
| train_v10 | 62,980 | v10训练集 |
| train_v11c_candidate | 63,187 | 当前推荐模型训练候选集 |
| feedback_loop | 持续增长 | 人工确认误判样本闭环 |

---

## 🎬 演示效果

运行 `python scripts/demo/visualize_detection.py` 查看：
- 分类结果（Human/AI + 置信度）
- 句级辅助分析
- 人工反馈闭环入口

---

## 📞 更多信息

- 当前答辩口径: [docs/project/DEFENSE_CURRENT_STATUS.md](docs/project/DEFENSE_CURRENT_STATUS.md)
- 完整成果（基线）: [docs/project/FINAL_RESULTS.md](docs/project/FINAL_RESULTS.md)
- 训练计划: [docs/project/TRAINING_PLAN.md](docs/project/TRAINING_PLAN.md)
- 评估报告: `evaluation_results/final_report.txt`

---

*最后更新: 2026-02-12*
