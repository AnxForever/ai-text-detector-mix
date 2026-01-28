# 数据集统一 Schema 模板（CSV/Parquet）
> 生成时间: 2026-01-27
> 目标: 统一字段约束，支持“场景主轴 + 风格副轴 + 二分类/Mixed 扩展”。
> 格式建议: CSV（采集/生成阶段） / Parquet（训练与分析阶段）

---

## 1. 主表字段（必备）

| 字段名 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| id | string | 是 | 全局唯一 ID（建议 `source_id + row_index`） |
| text | string | 是 | 原始文本 |
| y_main | string | 是 | HUMAN / AI / MIXED / UNCERTAIN |
| label | string | 否 | 原始标签（若存在） |
| scenario | string | 是 | education / workplace / knowledge / community / commerce / news |
| scenario_id | string | 是 | A / B / C / D / E / F |
| style | string | 是 | dialogue / explanation / list / report / guide / mixed |
| answer_type | string | 否 | list / guide / report / explanation / dialogue / mixed |
| domain | string | 否 | 主题/领域（可选：tech/finance/medical/education/general 等） |
| domain_risk | string | 否 | normal / high_stakes |
| hs_domain | string | 否 | legal / finance（高风险子域，可选） |
| length_bucket | string | 是 | 80-200 / 200-500 / 500-1000 / 1000-2000 / 2000+ |
| length_chars | int | 是 | 字符长度（去首尾空白） |
| source_type | string | 是 | public_dataset / open_source / web / ai_generated / internal |
| source_id | string | 是 | 来源标识（repo/url/doc_id 等） |
| collected_at | string | 是 | 采集日期（YYYY-MM-DD） |
| split | string | 否 | train / val / test / ood_style / ood_model / mixed_test / security_test |
| ood_split | string | 否 | OOD_HS（高风险域专用，固定值） |
| eval_suite | string | 否 | id / style_ood / model_ood / mixed / security |
| has_sep | bool | 否 | 是否含 [SEP] 或显式边界标记（主训练集必须 false） |

---

## 2. 生成样本字段（AI 专用）

| 字段名 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| model_family | string | 否 | gpt / gemini / deepseek / qwen / glm / other |
| model_name | string | 否 | 具体模型名称 |
| model_version | string | 否 | 版本或日期 |
| prompt_id | string | 否 | 模板 ID |
| decoding | string | 否 | 例如 `temp=0.7,top_p=0.95,max_tokens=1200` |
| seed | string | 否 | 随机种子（如有） |
| created_at | string | 否 | 生成日期（YYYY-MM-DD） |

---

## 3. 数据审计/评价字段（可选）

| 字段名 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| q_score | float | 否 | 质量分 Q (0–1) |
| d_score | float | 否 | 难度分 D (0–1) |
| y_conf | float | 否 | 标签置信度 (0–1) |
| q_flags | string | 否 | 质量问题标记（逗号分隔） |
| y_evidence | string | 否 | 标签证据码（逗号分隔） |
| routed_pool | string | 否 | core / hard / review / reject |

---

## 4. 混写扩展字段（MIXED 专用）

| 字段名 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| segment_annotations | string | 否 | JSON 序列，记录 span 边界与作者 |
| boundary_metrics | string | 否 | JSON（Pk / WindowDiff 等指标） |

---

## 5. CSV 表头示例

```
id,text,y_main,label,scenario,scenario_id,style,answer_type,domain,domain_risk,hs_domain,length_bucket,length_chars,source_type,source_id,collected_at,split,ood_split,eval_suite,has_sep,model_family,model_name,model_version,prompt_id,decoding,seed,created_at,q_score,d_score,y_conf,q_flags,y_evidence,routed_pool,segment_annotations,boundary_metrics
```

---

## 6. 最小落地规范

- **最少必填字段**: `id,text,y_main,scenario,scenario_id,style,length_bucket,length_chars,source_type,source_id,collected_at`
- 字段缺失允许为空，但必须保留列名
- 文本统一 UTF-8 编码
- 不移除列表/表格符号（结构是检测信号）
