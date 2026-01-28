# combined_v2_clean 审计补充报告

> 生成日期: 2026-01-27
> 来源: scripts/data_cleaning/clean_combined_v2.py 结果

## 清洗结果概览

- 输入: datasets/archive/combined_v2 (train/val/test)
- 输出: datasets/archive/_to_delete/sources_20260127/combined_v2_clean
- 总样本(清洗前): 66,001
- 移除 [SEP] 样本: 2,034
- 移除显式 AI/拒绝词样本: 906
- 去重移除: 4,498
- 清洗后总样本: 58,563
- 切分: train 46,850 / val 5,856 / test 5,857

## 关键验证

- Train/Val/Test 交叉重复: 0
- [SEP] 残留: 0

## 产出文件

- datasets/archive/_to_delete/sources_20260127/combined_v2_clean/train.csv
- datasets/archive/_to_delete/sources_20260127/combined_v2_clean/val.csv
- datasets/archive/_to_delete/sources_20260127/combined_v2_clean/test.csv
- datasets/archive/_to_delete/sources_20260127/combined_v2_clean/mixed_sep.csv (移除的混合文本)
- datasets/archive/_to_delete/sources_20260127/combined_v2_clean/phrase_removed.csv (移除的口癖样本)
- datasets/archive/_to_delete/sources_20260127/combined_v2_clean/cleaning_log.json

## 后续建议

1. mixed_sep.csv 迁入 Mixed-Test 构建池
2. 若口癖删除过重，可在评估阶段单独回放
3. 下一步补齐结构化风格样本后再进行 Style/Model-OOD 切分
