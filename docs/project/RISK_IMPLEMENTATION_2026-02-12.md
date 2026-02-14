# 风险治理实现记录 (2026-02-12)

## 已实现内容

1. API 风险可观测字段
- 在 `api/api.py` 新增:
  - `DETECTOR_DECISION_THRESHOLD` 环境变量
  - 响应字段 `modelVersion`, `decisionThreshold`, `riskFlags`, `domainHint`
  - 轻量域提示与风险标记逻辑（短文本、极长文本、模板化、低置信等）

2. 数据风险仪表盘脚本
- 新增 `scripts/analysis/generate_risk_dashboard.py`
- 输出:
  - `docs/plans/risk_dashboard_v1.json`
  - `docs/plans/risk_dashboard_v1.md`
- 覆盖:
  - 长度分桶偏差
  - source 完整性
  - 模板噪声
  - train/eval 重叠
  - 弱域来源

3. unknown source 分流脚本
- 新增 `scripts/analysis/triage_unknown_source.py`
- 输出目录:
  - `datasets/analysis/routed/unknown_source_v1/`
  - 含 `keep_verified.csv`, `review_needed.csv`, `drop_candidate.csv`, `summary.json`

4. v11 候选训练集构建脚本
- 新增 `scripts/data_cleaning/build_train_v11_candidate.py`
- 默认策略:
  - 过滤 hard pattern 样本
  - unknown 仅保留 keep_verified
  - 过滤极短/极长
  - 去重
- 输出:
  - `datasets/merged_v2/train_v11_candidate.csv`
  - `datasets/merged_v2/train_v11_candidate_summary.json`

5. 通用风险规则模块
- 新增 `scripts/utils/risk_patterns.py`，统一 hard/soft pattern。

6. 文档更新
- 新增流程文档 `docs/plans/risk_governance_workflow.md`
- 同步入口文档:
  - `docs/plans/README.md`
  - `docs/CLAUDE.md`
  - `docs/project/DOCS_INDEX.md`
  - `scripts/CLAUDE.md`
  - `api/CLAUDE.md`

## 本地执行结果（默认参数）

1. 风险仪表盘
- 风险标记: `unknown_source_ratio_high`, `length_bucket_bias_high`, `template_leakage_present`, `weak_domain_present`

2. unknown 分流
- keep_verified: 1,637
- review_needed: 1,767
- drop_candidate: 379

3. v11 候选集
- 行数: 62,980 -> 60,456
- unknown: 3,783 -> 1,637
- 过滤计数:
  - hard pattern: 750
  - unknown_unapproved: 1,767
  - length: 7

## 验证

- `py -3 -m py_compile api/api.py scripts/utils/risk_patterns.py scripts/analysis/generate_risk_dashboard.py scripts/analysis/triage_unknown_source.py scripts/data_cleaning/build_train_v11_candidate.py`
- `py -3 scripts/analysis/generate_risk_dashboard.py`
- `py -3 scripts/analysis/triage_unknown_source.py`
- `py -3 scripts/data_cleaning/build_train_v11_candidate.py`
- `pnpm -C frontend exec tsc --noEmit`

## Addendum alignment (review sync)

Additional updates were implemented after peer review:

- A1 now explicitly includes duplicate and near-duplicate checks in
  `scripts/analysis/generate_risk_dashboard.py`.
- B2 weak-domain supplement planning now has a hard lower bound of `>= 300` per weak domain
  via `scripts/analysis/plan_weak_domain_supplement.py`.
- API observability fields are now decoupled and controlled by
  `DETECTOR_INCLUDE_RISK_OBSERVABILITY` (default off) in `api/api.py`.
- C2 calibration for V11 is documented as mandatory rerun (no reuse of V10 temperature).
- Rollback policy is now codified in `scripts/evaluation/check_v11_regression_gate.py`
  with threshold `three_set_avg degradation > 0.5`.
