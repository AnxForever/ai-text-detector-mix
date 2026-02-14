# 项目文档说明

> 更新时间: 2026-02-13
> 当前生产模型: `models/bert_v11c_boundary_fix` (T=0.8165)

## 1) 先看这些（高频）

- `README.md` - 项目总览与快速入口
- `QUICKSTART.md` - 一键上手
- `docs/project/DEFENSE_CURRENT_STATUS.md` - 当前答辩口径（V11c）
- `docs/project/FINAL_RESULTS.md` - 关键结果汇总
- `docs/project/DATA_AND_MODELS.md` - 数据与模型对应关系

## 2) 运行与部署

- `api/api.py` - API 服务入口（默认 V11c）
- `api/README.md` - API 目录说明
- `api/API_KEYS.md` - API 配置说明

## 3) 评估与决策文档（V11主线）

- `docs/plans/v11_four_way_comparison.md` - V10/V11a/V11b/V11c 四方对比
- `docs/plans/v11c_regression_gate.md` - V11c 回归门结论
- `docs/plans/v11d_decision_2026-02-13.md` - V11d 不上线决策
- `docs/plans/v11d2_decision_2026-02-13.md` - V11d2 失败回滚决策
- `docs/plans/README.md` - plans 目录索引

## 4) 结构与治理文档

- `docs/project/WORKSPACE_STRUCTURE.md` - 工作区结构与活跃路径
- `archive/cleanup_2026-02-13/CLEANUP_MANIFEST.md` - 本轮清理清单与恢复命令
- `docs/project/CLEANUP_PHASE2_CANDIDATES.md` - 第二轮清理候选（提案）
- `docs/project/RISK_IMPLEMENTATION_2026-02-12.md` - 风险治理实现
- `docs/project/DEFENSE_FIX_CHECKLIST_2026-02-12.md` - 修复项总清单

## 5) 归档区（历史/失败实验）

- `archive/cleanup_2026-02-13/` - 已移出的 V11d/V11d2 模型与数据
- `archive/configs_legacy_2026-02-13/` - 已移出的历史生成配置
- `docs/plans/archive_failed_v11d/` - V11d/V11d2 的构建和交接文档
- `docs/archive/` - 旧版历史文档

---

如果只做当前版本开发，请优先聚焦：`api/`、`models/bert_v11c_boundary_fix/`、
`datasets/merged_v2/`、`scripts/{training,evaluation,data_cleaning}/`。
