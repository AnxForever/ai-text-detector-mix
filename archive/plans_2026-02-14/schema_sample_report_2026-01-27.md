# Schema 转换示例输出

> 生成时间: 2026-01-27  
> 说明: 示例用于验证字段对齐，不代表最终训练集

## 输出文件

- `datasets/samples/schema_sample/combined_v2_clean_train_sample.csv`（1000 rows）
- `datasets/samples/schema_sample/final_clean_phrase_clean_train_sample.csv`（1000 rows）

## 转换说明

- 使用 `scripts/data_cleaning/convert_to_schema.py` 进行转换
- 输入为已分类版本（含 style/domain/length_bucket）
- `collected_at` 默认填充为 `unknown`（原始数据未提供）
