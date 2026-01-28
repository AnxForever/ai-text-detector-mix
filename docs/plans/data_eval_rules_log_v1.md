# 数据评价系统规则版本日志

> 规则版本: v1
> 日期: 2026-01-27
> 负责人: (待填)

---

## 变更摘要
- 初版规则落地，基于 Q/D/y_conf 轻量方案

## Q 评分权重
- Q_clean: 0.4
- Q_length: 0.3
- Q_provenance: 0.3

## D 评分权重
- D_uncertainty: 0.7
- D_disagreement: 0.0
- D_ood: 0.0
- D_style_prior: 0.3

## y_conf 权重
- model_prob: 0.5
- source_trust: 0.3
- rule_support: 0.2

## 阈值
- Q < 0.3 → Reject
- 0.3 ≤ Q < 0.6 → Review
- Q ≥ 0.6 → 进入训练分流
- y_conf < 0.6 → Review
- y_conf ≥ 0.9 且 Q ≥ 0.6 → Core
- D ≥ 0.75 → Hard

## 备注
- D_disagreement/D_ood 暂不启用（无多模型/嵌入距离支持）
- 后续引入 Cleanlab / AUM 后可升级到 v2
