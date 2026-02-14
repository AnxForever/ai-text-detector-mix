# 评估结果汇总

> 日期: 2026-01-27
> 模型: 现有基线 (bert_improved / bert_v2_with_sep)
> 数据集版本: final_clean, combined_v2, combined_v2_clean, mixed_test_candidates

## 基础指标
- Accuracy: (待填充，需运行评测脚本)
- Precision: (待填充)
- Recall: (待填充)
- F1: (待填充)

## OOD 表现
- Style-OOD F1: (待填充)
- Model-OOD F1: (待填充)
- Mixed-Test (参考): `datasets/mixed/candidates/mixed_test_balanced_by_category_length.csv` 已生成

## 分片表现
- 技术文档 F1: final_clean + bert_improved 95.2%
- 列表式 F1: (待填充)
- 长文本(>1000) F1: (待填充)

## 数据质量摘要
- [SEP] 比例: combined_v2 3.1%，combined_v2_clean 0
- 口癖比例: final_clean train 移除 720 条；combined_v2_clean 移除 906 条
- 重复率: combined_v2 train 去重移除 3,254；clean 后 train/val/test 交叉重复 = 0

来源参考:
- `docs/plans/audit_reports/2026-01-27_audit_report.md`
- `docs/plans/audit_reports/2026-01-27_combined_v2_clean_report.md`
- `docs/plans/audit_reports/2026-01-27_cross_overlap_report.md`
- `docs/plans/audit_reports/2026-01-27_mixed_test_length_balance_report.md`

## 最近产出
- Mixed-Test 候选集 (去重+口癖清理): datasets/mixed/candidates/hybrid_expanded_clean_vs_final_clean.csv
- Mixed-Test 类别均衡: datasets/mixed/candidates/mixed_test_balanced_by_category.csv
- Mixed-Test 长度均衡: datasets/mixed/candidates/mixed_test_balanced_by_category_length.csv
- combined_v2_clean: datasets/archive/_to_delete/sources_20260127/combined_v2_clean
- final_clean_phrase_clean: datasets/archive/_to_delete/sources_20260127/final_clean_phrase_clean

## 下一步（待你确认后执行）
1. 运行评测脚本填充 Accuracy/F1 等核心指标
2. 基于现有数据生成 Style-OOD / Model-OOD 测试集（需要来源/风格字段）
3. 用 Mixed-Test 长度均衡集做鲁棒性对比评测
