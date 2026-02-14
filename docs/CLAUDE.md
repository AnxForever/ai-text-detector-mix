[根目录](../CLAUDE.md) > **docs**

# 文档模块

## 模块职责

集中管理项目所有文档，包括项目核心文档、计划文档和归档文档。

## 目录结构

```
docs/
├── README.md                    # 文档总览
├── project/                     # 项目核心文档
├── plans/                       # 计划与审计文档
└── archive/                     # 归档文档
```

## 项目文档 (docs/project/)

| 文档 | 说明 | 重要程度 |
|-----|------|---------|
| `DEFENSE_CURRENT_STATUS.md` | 答辩口径快照（最新） | 高 |
| `RISK_IMPLEMENTATION_2026-02-12.md` | 风险治理实现记录 | 高 |
| `FINAL_RESULTS.md` | 最终实验结果 | 高 |
| `DATASET_ISSUES_FOR_AI.md` | 数据集问题分析 | 高 |
| `TRAINING_PLAN.md` | 训练计划 | 中 |
| `DATA_AND_MODELS.md` | 数据与模型说明 | 中 |
| `EXPERIMENT_LOG.md` | 实验日志 | 中 |
| `TECHNICAL_SUMMARY_FOR_LITERATURE.md` | 技术总结 | 中 |
| `DOCS_INDEX.md` | 文档索引 | 辅助 |
| `CONTRIBUTING.md` | 贡献指南 | 辅助 |
| `CITATION.md` | 引用信息 | 辅助 |
| `GENERATION_TASK.md` | 生成任务说明 | 辅助 |

## 计划文档 (docs/plans/)

| 文档 | 说明 |
|-----|------|
| `README.md` | 计划文档索引 |
| `data_construction_plan.md` | 数据构建计划 |
| `data_audit_plan.md` | 数据审计计划 |
| `data_evaluation_plan.md` | 数据评估计划 |
| `model_training_plan.md` | 模型训练计划 |
| `data_evaluation_system_plan.md` | 数据评价系统计划 |
| `IMPROVEMENT_PLAN.md` | 改进计划 |
| `risk_governance_workflow.md` | 风险治理执行流程（V10->V11） |
| `risk_governance_addendum_2026-02-12.md` | Risk governance addendum (A1/B2/C2/rollback) |
| `audit_reports/` | 审计报告目录 |

## 其他文档

| 文档 | 说明 |
|-----|------|
| `DATASET_GUIDE.md` | 数据集使用指南 |
| `AI_TEXT_TESTING_GUIDE.md` | AI文本测试指南 |
| `MODEL_TESTING_GUIDE.md` | 模型测试指南 |
| `STANDARD_DATASETS_GUIDE.md` | 标准数据集指南 |
| `LENGTH_BIAS_SOLUTION.md` | 长度偏差解决方案 |
| `GRAPH_ENHANCEMENT_GUIDE.md` | 图增强指南 |

## 快速导航

### 了解项目成果

1. 查看 `docs/project/DEFENSE_CURRENT_STATUS.md`
2. 阅读 `docs/project/FINAL_RESULTS.md`（基线阶段）

### 了解数据问题

1. 阅读 `docs/project/DATASET_ISSUES_FOR_AI.md`
2. 查看 `docs/plans/audit_reports/`

### 了解开发计划

1. 阅读 `docs/plans/README.md`
2. 查看具体计划文档

## 文档规范

- 使用 Markdown 格式
- 文件名使用 UPPER_SNAKE_CASE
- 日期格式: YYYY-MM-DD
- 包含更新时间戳

## 相关文件清单

```
docs/
├── README.md
├── DATASET_GUIDE.md
├── AI_TEXT_TESTING_GUIDE.md
├── MODEL_TESTING_GUIDE.md
├── dataset_audit.md
├── project/
│   ├── DOCS_INDEX.md
│   ├── FINAL_RESULTS.md
│   ├── DATA_AND_MODELS.md
│   ├── TRAINING_PLAN.md
│   └── ...
├── plans/
│   ├── README.md
│   ├── data_construction_plan.md
│   ├── audit_reports/
│   └── ...
└── archive/
    └── ...
```

## 变更记录 (Changelog)

### 2026-02-12
- 新增 `DEFENSE_CURRENT_STATUS.md` 入口
- 更新“项目成果”导航顺序（最新口径优先）
- 新增 `risk_governance_workflow.md` 计划文档入口
- Added `risk_governance_addendum_2026-02-12.md` entry
- 新增 `RISK_IMPLEMENTATION_2026-02-12.md` 实现记录

### 2026-01-28
- 初始化模块文档

---

*文档更新时间: 2026-02-12*
