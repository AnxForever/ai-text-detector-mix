# 数据集分类与桶统计报告

> 数据集: final_clean_phrase_clean
> 生成时间: 2026-01-27 15:00:09

## 基本统计

- 总样本数: 54536

### 标签分布

- AI: 27355
- HUMAN: 27181

### Style 分布

- explanation: 32619
- academic: 6931
- readme: 6698
- list: 3488
- technical_doc: 2778
- dialogue: 2022

### Domain 分布

- general: 25815
- finance: 6415
- medical: 5164
- education: 4857
- software: 4695
- law: 3178
- ml_ai: 2814
- ops: 1598

### Length Bucket 分布

- 80-200: 18243
- 200-500: 14520
- 1000-2000: 7296
- 500-1000: 6539
- 2000+: 4326
- lt_80: 3612

## 缺失桶提示 (style × domain × length)

- dialogue / ml_ai / lt_80
- dialogue / ml_ai / 80-200
- dialogue / software / lt_80
- dialogue / ops / lt_80
- dialogue / ops / 80-200
- dialogue / medical / lt_80
- dialogue / education / lt_80
- dialogue / law / lt_80
- dialogue / law / 80-200
- list / ml_ai / lt_80
- list / ml_ai / 80-200
- list / software / lt_80
- list / software / 80-200
- list / ops / lt_80
- list / ops / 80-200
- list / ops / 200-500
- list / ops / 500-1000
- list / finance / lt_80
- list / finance / 80-200
- list / finance / 200-500
- list / medical / lt_80
- list / medical / 80-200
- list / education / lt_80
- list / education / 80-200
- list / law / lt_80
- list / general / lt_80
- technical_doc / education / lt_80
