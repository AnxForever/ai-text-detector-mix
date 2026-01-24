# AI文本检测模型 - 使用说明

> 基于BERT的中文AI文本检测模型（去格式偏差版本）

---

## 快速开始

### 一行命令启动（Windows/Linux/macOS通用）

```bash
python start.py
```

就这么简单！脚本会自动处理环境变量和启动测试工具。

### 或者手动启动

**Windows PowerShell:**
```powershell
.venv\Scripts\Activate.ps1
$env:HF_HUB_OFFLINE = "1"
python scripts\evaluation\test_single_text.py --interactive
```

**Linux/macOS:**
```bash
source .venv/bin/activate
export HF_HUB_OFFLINE=1
python3 scripts/evaluation/test_single_text.py --interactive
```

---

## 📚 论文写作文档

### 核心文档（撰写论文必读）

| 文档 | 说明 | 用途 |
|------|------|------|
| 📖 **[PROJECT_COMPLETE_REPORT.md](PROJECT_COMPLETE_REPORT.md)** | **完整项目报告** | 数据集构建、格式去偏、模型训练全流程 |
| 📝 **[PAPER_QUICK_REFERENCE.md](PAPER_QUICK_REFERENCE.md)** | **论文快速参考** | 关键数据、图表、结论速查表 |

### 实验结果文档

| 文档 | 内容 |
|------|------|
| [docs/EXPERIMENT_FINAL_RESULTS.md](docs/EXPERIMENT_FINAL_RESULTS.md) | 完整实验结果汇总 |
| [docs/PAPER_RELATED_WORK_DRAFT.md](docs/PAPER_RELATED_WORK_DRAFT.md) | Related Work章节草稿 |
| [docs/PAPER_RESULTS_TABLES.md](docs/PAPER_RESULTS_TABLES.md) | Results章节表格（10个表+6个图） |
| [docs/PAPER_DISCUSSION_POINTS.md](docs/PAPER_DISCUSSION_POINTS.md) | Discussion章节论点 |
| [docs/SIMPLE_RULE_VS_BERT_ANALYSIS.md](docs/SIMPLE_RULE_VS_BERT_ANALYSIS.md) | 简单规则vs BERT性能对比 |

---

## 🎯 核心成果

### 模型性能

- **测试准确率**: 100.00% (2208/2208)
- **F1分数**: 1.0000
- **格式免疫**: ✅ 优秀（最大下降仅0.05%）

### 格式去偏效果

| 指标 | 去偏前 | 去偏后 | 改善 |
|------|--------|--------|------|
| 格式偏差 | 64.90% | 2.41% | ↓62.49% |
| 简单规则准确率 | 81.61% | 48.87% | ↓32.74% |

### 简单规则 vs BERT

| 阶段 | 简单规则 | BERT | 提升幅度 |
|------|---------|------|---------|
| 去偏前 | 81.61% | 99.95% | +18.34% |
| 去偏后 | 48.87% | 100.00% | **+51.13%** |
| **增长倍数** | - | - | **2.79倍** |

---

## 📁 项目结构

```
datacollection/
├── start.py                              # 一键启动脚本
│
├── 📖 论文写作文档
│   ├── PROJECT_COMPLETE_REPORT.md        # 完整项目报告 ⭐⭐⭐
│   ├── PAPER_QUICK_REFERENCE.md          # 论文快速参考 ⭐⭐⭐
│   └── docs/                             # 详细实验结果
│       ├── EXPERIMENT_FINAL_RESULTS.md
│       ├── PAPER_RELATED_WORK_DRAFT.md
│       ├── PAPER_RESULTS_TABLES.md
│       ├── PAPER_DISCUSSION_POINTS.md
│       └── SIMPLE_RULE_VS_BERT_ANALYSIS.md
│
├── 🤖 模型和数据
│   ├── models/bert_improved/best_model/  # 训练好的模型 (781MB)
│   └── datasets/
│       ├── bert_debiased/                # 去偏后的训练数据 (18,170条)
│       ├── final/                        # AI生成文本 (9,170条)
│       └── human_texts/                  # 真实人类文本 (9,000条)
│
└── 🛠️ 脚本工具
    ├── scripts/data_cleaning/            # 格式去偏
    ├── scripts/training/                 # 模型训练
    └── scripts/evaluation/               # 测试和评估
        ├── test_single_text.py           # 单文本测试工具
        ├── format_adversarial_test.py    # 对抗测试
        └── format_bias_check.py          # 格式偏差检测
```

---

## 🎓 论文写作指南

### 第一步：阅读完整报告

阅读 **[PROJECT_COMPLETE_REPORT.md](PROJECT_COMPLETE_REPORT.md)** 了解：
- 数据集构建完整流程
- 格式偏差问题的发现和解决
- 模型训练过程和技术细节
- 实验结果和分析

### 第二步：使用快速参考

使用 **[PAPER_QUICK_REFERENCE.md](PAPER_QUICK_REFERENCE.md)** 快速查找：
- 关键数据和数字（直接引用）
- 图表建议和LaTeX模板
- Abstract/Introduction/Results模板
- 常见审稿意见及回应

### 第三步：引用实验结果

从 `docs/` 目录获取：
- 详细的实验数据表格
- Related Work文献综述
- Results章节的图表设计
- Discussion章节的讨论要点

---

## 🔬 测试模型

### 测试时注意事项

生成AI测试文本时：
- ✅ 必须用纯文本（不要用markdown格式）
- ✅ 长度300-600字最佳
- ✅ 提示词例子：
  ```
  请用350字纯文本（无格式）介绍人工智能。
  不要用markdown、列表、加粗等格式，像写普通文章一样。
  ```

### 运行测试

```bash
# 启动交互式测试
python start.py

# 或直接测试单个文本
python scripts/evaluation/test_single_text.py --text "你的文本"
```

---

## ❓ 常见问题

**Q: 虚拟环境不存在？**
```bash
python -m venv .venv
```

**Q: 没有GPU？**
编辑 `start.py` 第34行，改为：
```python
subprocess.run([sys.executable, test_script, "--interactive", "--device", "cpu"], check=True)
```

**Q: 如何复现实验？**
查看 **[PROJECT_COMPLETE_REPORT.md](PROJECT_COMPLETE_REPORT.md)** 第四章"模型训练"，包含完整的训练命令和配置。

**Q: 论文应该写什么？**
查看 **[PAPER_QUICK_REFERENCE.md](PAPER_QUICK_REFERENCE.md)**，包含完整的论文写作建议和模板。

---

## 📊 关键贡献

1. ✅ **发现格式偏差问题**：首次系统性量化中文AI检测的格式偏差（64%）
2. ✅ **提出去偏方案**：格式偏差从64%降至2.4%（减少96%）
3. ✅ **实现格式免疫模型**：测试准确率100%，对格式扰动完全鲁棒（最大下降0.05%）
4. ✅ **建立对抗测试框架**：4种格式扰动场景，自动化评估系统
5. ✅ **验证语义学习**：BERT vs 简单规则提升幅度翻倍（18.34% → 51.13%，2.79倍）

---

**所有数据、代码、文档均已准备就绪，可直接用于论文撰写！** 🎓📝

---

## 🆕 增强工具包（2026-01-24更新）

### 📦 新增工具

#### 数据收集
- **`scripts/collection/collect_standard_datasets.py`** - 标准数据集收集（HC3, THUCNews等）
- **`scripts/collection/collect_large_human_dataset.py`** - 大规模人类文本（5万+）
- **`scripts/generation/generate_large_ai_dataset.py`** - 大规模AI文本生成计划

#### 数据处理
- **`scripts/data_cleaning/balance_length_distribution.py`** - 长度偏差处理 ⚠️重要
- **`scripts/training/continue_training.py`** - 继续训练（不覆盖原模型）
- **`run_length_balanced_training.py`** - 一键完整训练流程

#### 特征提取
- **`scripts/features/statistical_features.py`** - 统计特征（10维）
- **`scripts/features/text_graph_builder.py`** - 图特征（6维）
- **`scripts/features/graph_neural_network.py`** - GCN深度特征（64维）
- **`scripts/features/extract_graph_features_batch.py`** - 批量图特征提取

#### 高级模型
- **`scripts/training/train_hybrid_model.py`** - 混合特征模型
- **`scripts/training/train_graph_enhanced_model.py`** - 图增强模型
- **`scripts/training/train_multitask_model.py`** - 多任务学习

### 📚 新增文档

- **`ENHANCEMENT_PLAN_6MONTHS.md`** - 6个月完整实施计划
- **`COMPLETE_TOOLKIT_SUMMARY.md`** - 完整工具包总结
- **`docs/LENGTH_BIAS_SOLUTION.md`** - 长度偏差解决方案 ⚠️必读
- **`docs/STANDARD_DATASETS_GUIDE.md`** - 标准数据集收集指南
- **`docs/GRAPH_ENHANCEMENT_GUIDE.md`** - 图增强详细指南
- **`docs/GRAPH_TOOLS_SUMMARY.md`** - 图增强工具总结

---

## ⚠️ 重要：长度偏差问题

**发现**：当前数据集存在严重长度偏差
- AI平均: 1680字符
- 人类平均: 942字符
- 差距: 78%

**影响**：模型可能学到"长文本=AI"的简单规则

**解决**：
```bash
# 1. 运行长度平衡
python scripts/data_cleaning/balance_length_distribution.py

# 2. 一键训练
python run_length_balanced_training.py
```

详见：`docs/LENGTH_BIAS_SOLUTION.md`

---

## 🚀 快速开始（增强版）

### 选项1：解决长度偏差（最优先）
```bash
python run_length_balanced_training.py
```

### 选项2：收集标准数据集
```bash
python scripts/collection/collect_standard_datasets.py
# 支持：HC3, THUCNews, MGTBench, NLPCC 2025
```

### 选项3：实现混合特征模型
```bash
# 提取统计特征
python scripts/features/statistical_features.py

# 训练混合模型
python scripts/training/train_hybrid_model.py
```

### 选项4：图增强（可选）
```bash
# 简单版：图统计特征
python scripts/features/extract_graph_features_batch.py \
  --input datasets/bert_debiased/train.csv \
  --output datasets/bert_debiased/train_graph.csv

# 完整版：GCN
python scripts/training/train_graph_enhanced_model.py
```

---

## 📊 增强后的模型架构

| 模型 | 特征 | 预期准确率 | 创新点 |
|------|------|-----------|--------|
| V1 (已完成) | BERT + 格式去偏 | 100% | ⭐核心创新 |
| V2 | +长度平衡 | 99.7% | 泛化能力 |
| V3 | +统计特征(10维) | 99.8% | 混合特征 |
| V4 | +图特征(6维) | 99.9% | 结构分析 |
| V5 | +GCN(64维) | 99.9% | 深度图学习 |
| V6 | +多任务 | 99.9% | 归属识别 |

---

## 🎯 6个月实施计划

| 阶段 | 时间 | 任务 | 产出 |
|------|------|------|------|
| 阶段1 | 1-3周 | 数据扩充 | 5万人类+5万AI |
| 阶段2 | 4-7周 | 混合特征 | BERT+统计+BiGRU |
| 阶段3 | 8-10周 | 多任务学习 | 检测+归属 |
| 阶段4 | 11-14周 | 图增强 | 图特征+GCN |
| 阶段5 | 15-20周 | 全面评估 | 鲁棒性测试 |
| 阶段6 | 21-24周 | 论文撰写 | 完整论文 |

详见：`ENHANCEMENT_PLAN_6MONTHS.md`

---

## 📖 推荐阅读顺序

### 立即阅读
1. **`docs/LENGTH_BIAS_SOLUTION.md`** - 解决当前问题
2. **`COMPLETE_TOOLKIT_SUMMARY.md`** - 了解所有工具

### 规划阶段
3. **`ENHANCEMENT_PLAN_6MONTHS.md`** - 6个月计划
4. **`docs/STANDARD_DATASETS_GUIDE.md`** - 数据集收集

### 实现阶段
5. **`docs/GRAPH_ENHANCEMENT_GUIDE.md`** - 图增强实现
6. **`PROJECT_COMPLETE_REPORT.md`** - 完整项目报告

### 论文撰写
7. **`PAPER_QUICK_REFERENCE.md`** - 论文快速参考
8. **`docs/PAPER_RESULTS_TABLES.md`** - 结果表格
