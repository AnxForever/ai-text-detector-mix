# 模型训练计划

> 更新时间: 2026-01-27
> 目标: 构建稳定的中文二分类基线并支持结构化文本场景

---

## 一、训练范围

- 主任务: 文档级二分类 (Human vs AI)
- 混合文本: 暂不进入主训练 (Mixed-Test 仅评测)

---

## 二、当前基线

- bert_improved (final_clean): 技术文档 95.2%，解释 81.1%，对话 95.5%
- bert_v2_with_sep (combined_v2): 技术文档 14.9%，解释 85.8%，对话 100%

结论:
- 结构化文本场景对数据质量更敏感
- [SEP] 混入导致模型走捷径

---

## 三、训练阶段设计

1. **阶段 A (高质量基座)**
   - 数据: final_clean + 结构化补齐样本 (不含混合)
   - 目标: 保持整体准确率，提升结构化风格

2. **阶段 B (配额均衡微调)**
   - 数据: core_v2 主训练集
   - 目标: 风格与长度分桶稳定

3. **阶段 C (难样本精调)**
   - 数据: 边界池 / 疑似错标池 / 对抗样本
   - 目标: 提升鲁棒性

---

## 四、超参数建议 (初始版本)

```json
{
  "model": "bert-base-chinese",
  "epochs": 3-5,
  "batch_size": 8-16,
  "learning_rate": 1e-5,
  "dropout": 0.3,
  "weight_decay": 0.01,
  "early_stopping_patience": 2
}
```

说明:
- 训练过拟合时优先降低学习率与增加 dropout
- 仅保存验证集最优模型

---

## 五、可用脚本

- scripts/training/train_bert_improved.py
- scripts/training/train_bert_bigru.py
- scripts/training/train_dpcnn.py
- scripts/training/train_span_detector.py (混合文本边界任务)

---

## 六、训练产出

- 模型权重: models/
- 日志与曲线: logs/ + scripts/evaluation/plot_training_curves.py
- 评估结果: evaluation_results/

---

## 七、通过标准

- 主任务 Accuracy >= 95%
- 技术文档/列表式 F1 >= 90%
- 结构化风格不出现明显崩溃

