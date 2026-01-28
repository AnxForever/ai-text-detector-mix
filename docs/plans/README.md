# 计划文档索引

> 更新时间: 2026-01-27
> 用途: 统一管理数据构建/审计/评估/训练的执行计划与当前状态

## 文档列表
- data_construction_plan.md - 场景驱动的数据构建与配额、元数据规范、采集/生成流程
- data_audit_plan.md - 数据质量审计与风险排查流程
- data_evaluation_plan.md - 评测集设计与评估流程
- model_training_plan.md - 训练阶段与配置策略
- dataset_inventory.md - 数据集文件盘点（Step 1）
- human_data_collection_2026-01-27.md - 人类数据采集清单（优先级与执行方案）
- ai_generation_template_framework_2026-01-27.md - AI 生成模板框架（样式/领域/长度对齐）
- dataset_schema_template_2026-01-27.md - 数据集统一 Schema 模板（CSV/Parquet）
- data_fill_execution_plan_2026-01-27.md - 数据补齐执行清单（按桶行动）
- schema_sample_report_2026-01-27.md - Schema 转换示例输出
- data_fill_pipeline_guide_2026-01-27.md - Data fill pipeline guide (dry-run)
- data_fill_pipeline_targets_2026-01-27.json - Quota-aligned targets config
- IMPROVEMENT_PLAN.md - 改进方案（V2）
- evaluation_result_template.md - 评估结果汇总模板
- evaluation_result_2026-01-27.md - 当前评估汇总（占位）
- data_evaluation_system_plan.md - 数据评价系统总计划
- data_eval_reason_codes.md - 数据评价系统原因码字典
- data_eval_rules_log_template.md - 规则版本日志模板
- data_eval_bucket_report_template.md - 桶统计报告模板
- data_eval_rules_log_v1.md - 规则版本日志 v1
- label_review_template.md - 疑似错标样本审计清单模板
- data_eval_bucket_report_2026-01-27.md - 桶统计报告（占位）
- label_review_2026-01-27.md - 疑似错标审计清单（占位）
- audit_reports/ - 审计输出报告与日志

## 当前状态快照 (来源: docs/project/DATASET_ISSUES_FOR_AI.md, 2026-01-26)
- final_clean: 44,350 样本，技术文档 AI 识别 95.2%，解释 81.1%，对话 95.5%
- combined_v2: 总样本 66,001，训练样本 52,800，包含 [SEP] 1,614 (3.1%)，技术文档 AI 识别 14.9%
- 文本长度: <200 字符占 38%，>1000 字符占 20%，中位 277，均值 608
- 标签分布: Human 48%，AI 52%

## 最新执行记录
- 2026-01-27: 完成 combined_v2 清洗并生成 `datasets/archive/_to_delete/sources_20260127/combined_v2_clean`
- 2026-01-27: 输出审计补充报告 `docs/plans/audit_reports/2026-01-27_combined_v2_clean_report.md`
- 2026-01-27: 输出跨数据集重复报告 `docs/plans/audit_reports/2026-01-27_cross_overlap_report.md`
- 2026-01-27: 生成 Mixed-Test 去重候选集 `datasets/mixed/candidates/hybrid_expanded_clean.csv`
- 2026-01-27: 输出 Mixed-Test 去重报告 `docs/plans/audit_reports/2026-01-27_mixed_test_clean_report.md`
- 2026-01-27: 输出 Mixed-Test 统计报告 `docs/plans/audit_reports/2026-01-27_mixed_test_stats_report.md`
- 2026-01-27: 清理 Mixed-Test 口癖样本 `datasets/mixed/candidates/hybrid_expanded_clean_no_phrases.csv`
- 2026-01-27: 输出 Mixed-Test 口癖清理报告 `docs/plans/audit_reports/2026-01-27_mixed_test_phrase_clean_report.md`
- 2026-01-27: 生成 Mixed-Test 候选集(仅去 final_clean 重叠) `datasets/mixed/candidates/hybrid_expanded_clean_vs_final_clean.csv`
- 2026-01-27: 生成 Mixed-Test 类别均衡集 `datasets/mixed/candidates/mixed_test_balanced_by_category.csv`
- 2026-01-27: 生成 Mixed-Test 长度均衡集 `datasets/mixed/candidates/mixed_test_balanced_by_category_length.csv`
- 2026-01-27: 输出 Mixed-Test 新增报告 `docs/plans/audit_reports/2026-01-27_mixed_test_candidate_report.md`
- 2026-01-27: 输出 Mixed-Test 类别均衡报告 `docs/plans/audit_reports/2026-01-27_mixed_test_category_balance_report.md`
- 2026-01-27: 输出 Mixed-Test 长度均衡报告 `docs/plans/audit_reports/2026-01-27_mixed_test_length_balance_report.md`
- 2026-01-27: 生成 final_clean 口癖清理候选集 `datasets/archive/_to_delete/sources_20260127/final_clean_phrase_clean`
- 2026-01-27: 输出 final_clean 口癖清理报告 `docs/plans/audit_reports/2026-01-27_final_clean_phrase_clean_report.md`
- 2026-01-27: 生成评分与分流结果 `datasets/analysis/routed/combined_v2_clean` `datasets/analysis/routed/final_clean_phrase_clean`
- 2026-01-27: 输出评分分流报告 `docs/plans/audit_reports/combined_v2_clean_score_route_report.md`
- 2026-01-27: 输出评分分流报告 `docs/plans/audit_reports/final_clean_phrase_clean_score_route_report.md`
- 2026-01-27: 创建评测集版本 `datasets/eval/splits/v1`
- 2026-01-27: 完成 combined_v2_clean 规则分类（style/domain/length）并输出桶统计报告 `docs/plans/bucket_report_combined_v2_clean_2026-01-27.md`
- 2026-01-27: 完成 final_clean_phrase_clean 规则分类（style/domain/length）并输出桶统计报告 `docs/plans/bucket_report_final_clean_phrase_clean_2026-01-27.md`
- 2026-01-27: 生成缺口汇总报告 `docs/plans/gap_report_2026-01-27.md`
- 2026-01-27: 生成数据补齐优先级清单 `docs/plans/data_fill_priority_2026-01-27.md`
- 2026-01-27: 生成数据补齐配额表 `docs/plans/data_fill_quota_2026-01-27.md`
- 2026-01-27: 生成人类数据采集清单 `docs/plans/human_data_collection_2026-01-27.md`
- 2026-01-27: 生成 AI 生成模板框架 `docs/plans/ai_generation_template_framework_2026-01-27.md`
- 2026-01-27: 生成数据集统一 Schema 模板 `docs/plans/dataset_schema_template_2026-01-27.md`
- 2026-01-27: 生成数据补齐执行清单 `docs/plans/data_fill_execution_plan_2026-01-27.md`
- 2026-01-27: 生成 Schema 转换示例 `docs/plans/schema_sample_report_2026-01-27.md`
- 2026-01-27: 生成数据补齐脚本框架与配置模板 `scripts/generation/data_fill_pipeline.py` `configs/data_fill_pipeline_template.json`
- 2026-01-27: 生成配额对齐的 data fill targets 配置 `configs/data_fill_pipeline_targets_2026-01-27.json`
- 2026-01-27: 生成 data fill 执行计划 `datasets/planning/data_fill_runs/data_fill_run_20260127_152912`
- 2026-01-27: 生成数据集分类索引 `datasets/README.md` `datasets/registry.json`
- 2026-01-27: 重整 datasets 目录结构并删除重复 mixed-test 文件
- 2026-01-27: 合并 active 数据集并生成 core_v1 `datasets/active/core_v1`
- 2026-01-27: 重复来源已移动至 `datasets/archive/_to_delete/sources_20260127`（待手动删除）

## 统一约束
- 主训练集不含 [SEP] / 混合文本
- 混合文本进入 Mixed-Test
- 安全钓鱼/诈骗文本可选进入 Security-Test（不进主训练）
- 先去重后切分，确保 splits 互斥
