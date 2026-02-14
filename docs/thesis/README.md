# 📚 论文写作材料索引

> 最后更新: 2026-02-12
> 状态: ✅ **章节模板已完成，实验章有已填充版本，可直接进入答辩稿整合。**

---

## 📂 文件清单

| 章节 | 文件 | 状态 | 说明 |
|-----|------|------|------|
| 总览 | `thesis_outline.md` | ✅ 完成 | 完整论文框架和写作指导 |
| **第1章** | `chapter1_introduction_template.md` | ✅ 完成 | 绪论，研究背景与创新点 |
| **第2章** | `chapter2_related_work_template.md` | ✅ 完成 | 相关工作，技术综述 |
| **第3章** | `chapter3_dataset_template.md` | ✅ 完成 | 数据集构建，含真实统计数据 |
| **第4章** | `chapter4_method_template.md` | ✅ 完成 | 方法设计，含代码结构分析 |
| **第5章** | `chapter5_experiments_template.md` | 📝 模板 | 空白模板版本 |
| **第5章** | `chapter5_experiments_filled.md` | ✅ 大部分填充 | 已填入评估结果数据 |
| **第6章** | `chapter6_conclusion_template.md` | ✅ 完成 | 总结与展望 |
| **摘要** | `abstract_template.md` | ✅ 完成 | 中英文摘要模板 |

---

## 🎯 建议写作顺序

```
推荐顺序: 第5章 → 第4章 → 第3章 → 第2章 → 第1章 → 第6章 → 摘要

为什么从第5章开始？
├─ 实验数据最客观，不需要太多文字功底
├─ 填表格为主，完成快有成就感
├─ 写完实验才知道方法设计要强调什么
└─ 其他章节可以倒推来写
```

---

## ✅ 已填充的真实数据

### 数据集统计 (第3章、第5章)

| 数据项 | 数值 | 来源 |
|-------|------|------|
| 总样本数 | 58,563 | core_v1实际统计 |
| 训练集 | 46,849 (80%) | core_v1实际统计 |
| 验证集 | 5,856 (10%) | core_v1实际统计 |
| 测试集 | 5,858 (10%) | core_v1实际统计 |
| Human文本 | 27,181 (46.4%) | core_v1实际统计 |
| AI文本 | 31,382 (53.6%) | core_v1实际统计 |
| 平均长度 | 627.7字 | core_v1实际统计 |

### 模型性能 (第5章)

| 指标 | 数值 | 来源 |
|-----|------|------|
| 整体准确率 | 98.71% | FINAL_RESULTS.md |
| Human Precision | 98.98% | evaluation_results |
| Human Recall | 98.33% | evaluation_results |
| AI Precision | 98.47% | evaluation_results |
| AI Recall | 99.07% | evaluation_results |
| C2混合文本 | 93.84% (+14%) | final_report.txt |
| 长度准确率方差 | 0.0 | length_aware_evaluation.json |

### 模型配置 (第4章)

| 参数 | 值 | 来源 |
|-----|-----|------|
| 基础模型 | chinese-roberta-wwm-ext | model config |
| 隐藏维度 | 768 | model config |
| 注意力头 | 12 | model config |
| 层数 | 12 | model config |
| 参数量 | ~110M | model config |
| 学习率 | 2e-5 | train_bert_improved.py |
| Batch Size | 16 | train_bert_improved.py |

---

## ❌ 还需要填充的内容

### 第5章待补充

- [ ] 实验环境细节（CPU型号、GPU型号、内存）
- [ ] FastText、TextCNN基线实验结果
- [ ] 案例分析的具体文本内容
- [ ] 训练时间、推理速度实测数据
- [ ] 场景分布消融实验（需要schema转换后）

### 第3章待更新

- [ ] 新生成数据的统计（生成完成后）
- [ ] schema转换后的场景分布
- [ ] 模型分布更新

### 其他章节

- [ ] 第1章 绪论 - 需要写作
- [ ] 第2章 相关工作 - 需要文献调研
- [ ] 第6章 总结与展望 - 最后写

---

## 🔧 快速开始

### 1. 查看论文框架
```bash
cat docs/thesis/thesis_outline.md
```

### 2. 开始填写第5章
```bash
# 复制模板开始编辑
cp docs/thesis/chapter5_experiments_filled.md docs/thesis/chapter5_draft.md
# 用你喜欢的编辑器打开
```

### 3. 获取更多实验数据
```bash
# 运行评估获取详细指标
python scripts/evaluation/eval_complete.py

# 运行对比实验（如果有其他模型）
python scripts/evaluation/comprehensive_eval.py
```

---

## 📊 图表文件位置

已有的图表文件可直接用于论文：

| 图表 | 路径 | 用途 |
|-----|------|------|
| 训练曲线 | `evaluation_results/training_curves.png` | 图5-x |
| ROC曲线 | `evaluation_results/roc_curve.png` | 图5-x |
| 混淆矩阵 | `evaluation_results/confusion_matrix.png` | 图5-x |
| 置信度分布 | `evaluation_results/confidence_distribution.png` | 图5-x |

---

## 📝 写作小贴士

1. **数字要精确**：使用真实测量数据，不要编造
2. **表格清晰**：每个表格要有编号和标题
3. **分析有深度**：不只列数据，要解释原因
4. **前后呼应**：结论要回应引言的研究问题
5. **引用规范**：使用GB/T 7714或学校要求的格式

---

*创建时间: 2026-01-28*
*建议: 每完成一章就commit一次，方便版本管理*
