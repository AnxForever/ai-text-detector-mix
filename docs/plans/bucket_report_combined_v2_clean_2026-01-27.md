# 数据集分类与桶统计报告

> 数据集: combined_v2_clean
> 生成时间: 2026-01-27 14:55:39

## 基本统计

- 总样本数: 61503

### 标签分布

- AI: 34313
- HUMAN: 27190

### Style 分布

- explanation: 35673
- readme: 7606
- academic: 7462
- list: 5395
- technical_doc: 3219
- dialogue: 2148

### Domain 分布

- general: 29267
- finance: 7175
- medical: 5690
- education: 5553
- software: 5122
- law: 3587
- ml_ai: 3348
- ops: 1761

### Length Bucket 分布

- 80-200: 19056
- 200-500: 16294
- 500-1000: 9540
- 1000-2000: 8308
- 2000+: 4367
- lt_80: 3938

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
- list / finance / lt_80
- list / medical / lt_80
- list / education / lt_80
- list / education / 80-200
- list / law / lt_80
- list / general / lt_80
- technical_doc / education / lt_80
