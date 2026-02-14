# 数据清洗脚本

> 用于数据预处理、清洗、转换的核心脚本集

## 核心脚本

| 脚本 | 用途 | 命令示例 |
|-----|------|---------|
| `final_merge.py` | **主合并脚本** - 合并 legacy 和生成数据为 core_v2 | `python final_merge.py` |
| `convert_to_unified_schema.py` | Schema v2 转换 | `python convert_to_unified_schema.py` |
| `deduplicate_samples.py` | 基于 MD5 的去重 | `python deduplicate_samples.py <input> <output>` |
| `add_sep_markers.py` | 添加 [SEP] 边界标记 | `python add_sep_markers.py` |
| `prepare_span_labels.py` | 准备 Span 检测标签 | `python prepare_span_labels.py` |
| `clean_prompt_residue.py` | 清理 AI 提示词残留 | `python clean_prompt_residue.py` |

## 工具脚本

| 脚本 | 用途 |
|-----|------|
| `classify_scenario.py` | 自动场景分类（A-F） |
| `evaluate_data_quality.py` | 数据质量评估报告 |

## 数据处理流程

```
原始数据 → 清洗 → 去重 → Schema转换 → 合并 → 划分
   ↓         ↓       ↓        ↓         ↓       ↓
legacy/   clean_*  dedup   convert   final   train/val/test
generated           icate   to_v2    merge
```

## Schema v2 格式

```json
{
  "text": "文本内容",
  "label": 0,           // 0=Human, 1=AI
  "scenario_id": "A",   // A-F 场景
  "style": "report",    // 风格标签
  "model": "gpt-4",     // 生成模型（AI样本）
  "length_bucket": "200-500",
  "source": "来源标识"
}
```

## 归档脚本

旧版本脚本已移至 `archive/scripts/data_cleaning/`

---

*更新时间: 2026-01-28*
