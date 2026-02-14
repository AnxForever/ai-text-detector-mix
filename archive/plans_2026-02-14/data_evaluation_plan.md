# 数据评估计划

> 更新时间: 2026-01-27
> 目标: 评估主任务、场景/风格泛化、模型泛化、混合文本鲁棒性与安全场景表现

---

## 一、评估拆分

1. **ID Test**: 同分布测试集
2. **Style-OOD**: 留出 1 个文风（建议“说明书式”或“列表式”）
3. **Model-OOD**: 留出 1 个模型（建议 Gemini 或 GPT）
4. **Mixed-Test**: 混合文本评测集（不参与主训练）
5. **Security-Test（可选）**: 钓鱼/诈骗文本独立测试集（不参与主训练）
6. **OOD_HS（高风险域）**: 法律/金融独立测试子集（仅展示鲁棒性与风险边界）

---

## 二、评估指标

- Accuracy / Precision / Recall / F1
- Macro-F1（按场景/文风/长度分桶）
- Confusion Matrix
- 置信度分布（边界样本分析）

---

## 三、切片评估维度

- 场景: education / workplace / knowledge / community / commerce / news
- 文风: dialogue / explanation / list / report / guide / mixed
- 长度: 80-200 / 200-500 / 500-1000 / 1000-2000 / 2000+
- 生成模型: DeepSeek / Qwen / GLM / GPT / Gemini
- 领域(可选): tech / finance / medical / education / general
- 风险域(可选): domain_risk=high_stakes（legal/finance）

---

## 四、执行步骤

1. 运行标准评估（ID / Style-OOD / Model-OOD）
2. 输出分片指标（场景 × 文风 × 长度）
3. 生成错误分析与混淆矩阵
4. Mixed-Test 专项评估（仅报告，不参与训练调整）
5. Security-Test 专项评估（可选，单独报告）
6. OOD_HS 专项评估（legal/finance 分桶 + answer_type 分桶）

当前评测集版本:
- `datasets/eval/splits/v1/README.md`（含 ID 与 Mixed-Test）
- 待新增: `datasets/eval/splits/v2`（加入场景标签与 OOD）
- 待新增: `datasets/eval/security_phishing`（可选）

可用脚本 (执行前检查参数):
- scripts/evaluation/eval_complete.py
- scripts/evaluation/complete_evaluation.py
- scripts/evaluation/comprehensive_eval.py
- scripts/evaluation/format_adversarial_test.py

---

## 五、当前基线（旧版本记录）

- final_clean + bert_improved: 技术文档 95.2%，解释 81.1%，对话 95.5%
- combined_v2 + bert_v2_with_sep: 技术文档 14.9%，解释 85.8%，对话 100%

---

## 六、通过标准

- 主任务 Accuracy >= 95%
- 列表式/说明书式 F1 >= 90%
- Style-OOD 与 Model-OOD 无明显崩溃 (>= 85%)
- Mixed-Test / Security-Test 仅报告趋势，不作为主指标

---

## 七、轻量评价体系（无需 API）

### 7.1 数据质量硬指标
- 重复率（split 内与跨 split）
- [SEP] / 口癖 / 拒绝词比例
- 长度分桶分布（<80、80-200、200-500、500-1000、1000-2000、2000+）
- 非中英文字符占比异常样本（可选）

### 7.2 基础性能维度
- ID Test / Style-OOD / Model-OOD / Mixed-Test / Security-Test（可选）
- 按场景、文风、长度、模型来源进行分片统计

### 7.3 难样本池（低成本替代 Cartography）
- 低置信度样本（0.45–0.55）
- 高置信反标签样本（疑似错标）
- 作为后续人工审与增量清洗的优先集合

### 7.4 执行清单（轻量版）
1. 统计数据质量硬指标
2. 生成 ID / Style-OOD / Model-OOD / Mixed-Test / Security-Test（可选）指标
3. 输出分片指标（场景 / 文风 / 长度 / 模型来源）
4. 导出难样本池（低置信 + 高置信反标签）
5. 汇总到评估结果模板并存档

### 7.5 输出模板（轻量版）

```
# 评估结果汇总

> 日期:
> 模型:
> 数据集版本:

## 基础指标
- Accuracy:
- Precision:
- Recall:
- F1:

## OOD 表现
- Style-OOD F1:
- Model-OOD F1:
- Mixed-Test (参考):
- Security-Test (参考，可选):
- OOD_HS (参考，高风险域):

## 分片表现
- 列表式 F1:
- 说明书式 F1:
- 长文本(>1000) F1:

## 数据质量摘要
- [SEP] 比例:
- 口癖比例:
- 重复率:
```

---

## 八、重度评价体系（需 API 或额外算力）

### 8.1 统计语言学指标
- PPL 分布（人类与 AI 子集对比）
- Burstiness（句子长度方差 / 句子级 PPL 方差）

### 8.2 训练动力学评估
- Dataset Cartography（confidence/variability/correctness）
- AUM（边际下面积）用于疑似错标筛选

### 8.3 置信学习与标签诊断
- Cleanlab / find_label_issues（基于样本外预测概率）

### 8.4 LLM-as-a-Judge
- G-Eval / Pairwise 对比评分
- 需进行偏见校准（位置偏见/冗长偏见）
