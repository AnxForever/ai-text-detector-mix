# 数据评价系统原因码字典

> 更新时间: 2026-01-27
> 用途: 统一 Q/D/y_conf 的可解释标记，方便审计与复现

---

## 一、质量原因码 (q_*)

- q_garbage: 乱码/编码异常
- q_truncated: 疑似截断
- q_duplicate: 重复/近重复
- q_boilerplate: 模板化/免责声明/口癖
- q_format_abnormal: 只有标点/只有列表符/只有代码块
- q_lang_mismatch: 中英文比例异常/语言不匹配
- q_pii_risk: 可能包含隐私信息
- q_sep_contamination: 含 [SEP] 或边界标记污染

---

## 二、难度原因码 (d_*)

- d_uncertain: 预测边际过小 (|pAI-pH| 低)
- d_disagreement: 多模型/多种子预测分歧
- d_ood: 嵌入距离训练分布过远
- d_style_prior: 技术文档/列表式先验加权

---

## 三、标签置信原因码 (y_evidence_*)

- y_model_vote: 多模型投票一致
- y_single_model: 单模型高置信
- y_rule_support: 规则证据支持
- y_source_trust: 来源可信度高
- y_source_unknown: 来源不明
- y_conflict: 多证据冲突

---

## 四、建议使用方式

- 允许多标签并存 (例如: ["q_duplicate", "q_boilerplate"])
- 规则版本更新时同步更新该字典
- 输出审计报告时必须包含原因码统计
