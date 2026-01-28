# 数据评价系统计划 (Data Evaluation System Plan)

> 更新时间: 2026-01-27
> 目标: 建立“元数据标注 + 质量评分(Q) + 难度评分(D) + 分流策略”的可复用体系

---

## 一、核心目标

对每条样本回答四个问题:
1) 它是什么 (scenario/style/domain/provenance)
2) 它干净吗 (Quality, Q)
3) 标签靠谱吗 (y_conf + evidence)
4) 对训练有价值吗 (Difficulty, D)

---

## 二、最小统一样本 Schema

```json
{
  "id": "sample_000001",
  "text": "...",
  "y_main": "HUMAN",
  "scenario": "education",
  "scenario_id": "A",
  "style": "guide",
  "domain": "software",
  "length_bucket": "200-500",
  "provenance": {
    "source_type": "public_dataset",
    "source_name": "final_clean",
    "ai_model_family": "qwen",
    "prompt_id": "tech_api_023",
    "collect_time": "2026-01-27"
  },
  "quality": {
    "Q": 0.78,
    "q_flags": ["q_boilerplate"],
    "q_reasons": ["explicit_ai_phrase"]
  },
  "difficulty": {
    "D": 0.62,
    "d_flags": ["d_uncertain"],
    "d_reasons": ["low_margin"]
  },
  "label_confidence": {
    "y_conf": 0.91,
    "y_evidence": ["model_vote", "source_trust"]
  },
  "routing": {
    "target_pool": "core_train",
    "rule_version": "v1"
  }
}
```

---

## 三、两层标签体系

### 3.1 主标签 (y_main)
- HUMAN / AI / MIXED / UNCERTAIN

### 3.2 结构化元标签 (不直接给模型学)
- scenario: education / workplace / knowledge / community / commerce / news
- style: 对话式 / 连续解释式 / 列表要点式 / 报告总结式 / 说明书式 / 混合格式
- domain: tech / finance / medical / education / general / other（可选）
- length_bucket: 80–200 / 200–500 / 500–1000 / 1000–2000 / 2000+
- provenance: source_type / ai_model_family / prompt_id / collect_time / source_name

---

## 四、质量分 Q (0–1)

### 4.1 构成 (轻量版优先)
- Q_clean: 字符合法率、异常符号占比、重复字符/重复 ngram
- Q_length: 过短惩罚 (<80 直接 0)
- Q_provenance: 来源可信度加权

### 4.2 阈值建议
- Q < 0.3: Reject
- 0.3 <= Q < 0.6: Review
- Q >= 0.6: 进入下一步判断

---

## 五、难度分 D (0–1)

### 5.1 构成 (轻量版)
- D_uncertainty: 1 - |p(AI) - p(H)|
- D_style_prior: 技术文档/列表式 +0.05

### 5.2 分桶建议
- D >= 0.75: Hard
- 0.4 <= D < 0.75: Medium
- D < 0.4: Easy

---

## 六、标签置信度 y_conf (0–1)

### 6.1 证据来源 (轻量版)
- 现有模型概率 (校准后)
- 规则证据 (如显式“作为AI”)
- 来源证据 (生成记录/公开数据集)

### 6.2 分流规则
- y_conf < 0.6: Review / Reject
- y_conf >= 0.9 且 Q >= 0.6: Core
- 高置信反标签: 优先人工审

---

## 七、分流策略 (四个池)

1) Core Train: Q高 & y_conf高 & 非 MIXED
2) Hard Train: Q高但 D高
3) Audit/Review: Q中或 y_conf低
4) Reject: Q低/重复/截断/[SEP]污染

---

## 八、阶段化实施路线

### 阶段 A (无 API)
- 规则 + 统计计算 Q
- 现有模型输出计算 D/y_conf
- 生成四个池 + 桶统计报告

### 阶段 B (可离线增强)
- PPL / Burstiness
- EL2N

### 阶段 C (重度/训练过程)
- AUM / Forgetting / Cartography
- Cleanlab
- LLM-as-a-Judge

---

## 九、产出清单

- 评分结果 JSONL (逐样本)
- 分流池 CSV (Core/Hard/Review/Reject)
- 桶统计报告 (scenario × style × length)
- 版本日志 (rule_version + 参数记录)

---

## 十、立即可执行清单 (不动 API)

1) 定义 schema 与原因码字典
2) 生成 Q/D/y_conf 初版打分
3) 输出四个池
4) 输出桶统计与审计报告

参考模板:
- `docs/plans/data_eval_reason_codes.md`
- `docs/plans/data_eval_rules_log_template.md`
- `docs/plans/data_eval_bucket_report_template.md`

---

## 附录 A：原因码字典（建议）

### A.1 质量原因码（q_*）
- q_garbage: 乱码/编码异常
- q_truncated: 疑似截断
- q_duplicate: 重复/近重复
- q_boilerplate: 模板化/免责声明/口癖
- q_format_abnormal: 只有标点/只有列表符/只有代码块
- q_lang_mismatch: 中英文比例异常/语言不匹配
- q_pii_risk: 可能包含隐私信息
- q_sep_contamination: 含 [SEP] 或边界标记污染

### A.2 难度原因码（d_*）
- d_uncertain: 预测边际过小 (|pAI-pH| 低)
- d_disagreement: 多模型/多种子预测分歧
- d_ood: 嵌入距离训练分布过远
- d_style_prior: 技术文档/列表式先验加权

### A.3 标签置信原因码（y_evidence_*）
- y_model_vote: 多模型投票一致
- y_single_model: 单模型高置信
- y_rule_support: 规则证据支持
- y_source_trust: 来源可信度高
- y_source_unknown: 来源不明
- y_conflict: 多证据冲突

---

## 附录 B：评分权重与阈值建议

### B.1 质量分 Q（初始权重）
建议先用可解释权重，后续用小样本拟合调参：
- Q_clean = 0.4
- Q_length = 0.3
- Q_provenance = 0.3

示例：  
Q = 0.4 * Q_clean + 0.3 * Q_length + 0.3 * Q_provenance

### B.2 难度分 D（初始权重）
示例：  
D = 0.7 * D_uncertainty + 0.3 * D_style_prior  
若有多模型分歧，可替换为：  
D = 0.4 * (1-margin) + 0.4 * D_disagreement + 0.2 * D_ood (+ style_prior)

### B.3 标签置信 y_conf（初始权重）
示例：  
y_conf = 0.5 * model_prob + 0.3 * source_trust + 0.2 * rule_support

### B.4 阈值建议（可先固定）
- Q < 0.3 → Reject
- 0.3 ≤ Q < 0.6 → Review
- Q ≥ 0.6 → 进入训练分流
- y_conf < 0.6 → Review
- y_conf ≥ 0.9 且 Q ≥ 0.6 → Core
- D ≥ 0.75 → Hard

---

## 附录 C：桶统计与报告模板

### C.1 桶统计矩阵（样例）

```
scenario x style 统计表:
- count / avg_Q / avg_D / suspicious_rate
```

### C.2 汇总报告模板（样例）

```
# 数据评价系统汇总

> 日期:
> 数据集版本:
> 规则版本:

## 样本分流
- Core Train:
- Hard Train:
- Review:
- Reject:

## 质量/难度概览
- 平均 Q:
- 平均 D:
- 低 Q 比例:
- 高 D 比例:

## 桶统计摘要
- 技术文档 / 列表式 / 学术 关键桶数量与疑似错标率
```

