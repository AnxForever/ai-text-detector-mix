# 风险治理工作流（V10 -> V11）

See also: `risk_governance_addendum_2026-02-12.md` (review-aligned constraints).
更新时间: 2026-02-12

## 目标

在低算力条件下，优先降低 FN（AI 被判 Human）风险，先做数据治理再做轻量训练迭代。

## 1) 生成风险仪表盘

```bash
py -3 scripts/analysis/generate_risk_dashboard.py
```

产物:
- `docs/plans/risk_dashboard_v1.json`
- `docs/plans/risk_dashboard_v1.md`

覆盖维度:
- 长度分桶偏差
- source 完整性（unknown 比例）
- 模板/指令噪声
- 训练集与 fair_test 重叠
- V10 弱域来源

## 2) 对 unknown 样本分流

```bash
py -3 scripts/analysis/triage_unknown_source.py
```

产物目录:
- `datasets/analysis/routed/unknown_source_v1/keep_verified.csv`
- `datasets/analysis/routed/unknown_source_v1/review_needed.csv`
- `datasets/analysis/routed/unknown_source_v1/drop_candidate.csv`
- `datasets/analysis/routed/unknown_source_v1/summary.json`

## 3) 构建 v11 候选训练集

```bash
py -3 scripts/data_cleaning/build_train_v11_candidate.py
```

默认策略:
- 移除 hard pattern 样本
- unknown 仅保留 keep_verified
- 过滤极短/极长文本
- 去重

产物:
- `datasets/merged_v2/train_v11_candidate.csv`
- `datasets/merged_v2/train_v11_candidate_summary.json`

## 4) 可选：放宽 unknown 审核策略

如果你需要扩大样本规模，可把 `review_needed` 也纳入：

```bash
py -3 scripts/data_cleaning/build_train_v11_candidate.py --allow-review-unknown
```

## 5) 训练与对比（下一步）

建议后续用 `train_v11_candidate.csv` 做轻量训练，并复用 fair_test 三集做对比，重点看：
- `independent_data` FN 数是否下降
- 弱域（如 formal / llama）是否改善
- 三集平均是否稳定
