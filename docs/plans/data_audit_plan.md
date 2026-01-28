# 数据审计计划

> 更新时间: 2026-01-27
> 目标: 识别数据质量问题、标签噪声、泄露与捷径特征

---

## 一、审计范围

- datasets/archive/final_clean
- datasets/archive/combined_v2
- datasets/mixed/hybrid

审计对象:
- 标签准确性
- 重复与近重复
- [SEP] / 混合样本污染
- 长度与风格分布偏差
- 训练/验证/测试交叉泄露

---

## 二、当前已知问题 (来源: docs/project/DATASET_ISSUES_FOR_AI.md, 2026-01-26)

- combined_v2 含 [SEP] 1,614 (3.1%)
- 结构化文本识别率异常低 (列表式 3%、技术文档 0.1%)
- <200 字符占比 38%，>1000 占比仅 20%
- 标签分布略偏 AI (52%)

---

## 三、审计清单

1. **统计概览**
   - 总样本、标签分布、长度分布、风格分布
2. **污染与捷径**
   - [SEP] / 明确拒绝词 / 自我声明
3. **重复与泄露**
   - 训练/验证/测试重复
   - 近重复段落
4. **标注噪声**
   - 低置信样本
   - 高置信反标签样本
5. **结构化风格覆盖**
   - 技术文档、列表式、README 的覆盖率

---

## 四、执行步骤

1. **生成基础统计**
   - 标签分布、长度分桶、风格占比
2. **[SEP] 与混合检测**
   - 统计并剔除主训练集
3. **去重与泄露检查**
   - 先全局去重，再切分
4. **标注噪声定位**
   - 使用基线模型推理，抽取边界样本
5. **审计报告输出**
   - 记录每一步样本变化与发现

可用脚本 (执行前检查参数):
- scripts/data_cleaning/evaluate_data_quality.py
- scripts/evaluation/comprehensive_data_quality.py
- scripts/evaluation/check_length_balance.py
- scripts/evaluation/format_bias_check.py

---

## 五、产出物

- 数据审计报告 (Markdown)
- 统计摘要 (CSV/JSON)
- 疑似错标与边界池列表

建议输出位置:
- docs/plans/audit_reports/
- datasets/analysis/metadata/audit_logs/

---

## 六、通过标准

- 主训练集 [SEP] = 0
- 重复/近重复比例 < 1%
- 标签分布接近 50/50
- 结构化风格占比满足配额

