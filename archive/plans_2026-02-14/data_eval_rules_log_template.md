# 数据评价系统规则版本日志模板

> 规则版本:
> 日期:
> 负责人:

---

## 变更摘要
- 

## Q 评分权重
- Q_clean:
- Q_length:
- Q_provenance:

## D 评分权重
- D_uncertainty:
- D_disagreement:
- D_ood:
- D_style_prior:

## y_conf 权重
- model_prob:
- source_trust:
- rule_support:

## 阈值
- Q < 0.3 → Reject
- 0.3 ≤ Q < 0.6 → Review
- Q ≥ 0.6 → 进入训练分流
- y_conf < 0.6 → Review
- y_conf ≥ 0.9 且 Q ≥ 0.6 → Core
- D ≥ 0.75 → Hard

## 备注
- 
