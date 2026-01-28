# 生成任务配置

本目录包含 AI 文本生成任务的配置文件，用于数据填充管道 (Data Fill Pipeline) 和场景生成脚本。

## 配置类型

### 1. 数据填充管道配置

| 文件 | 用途 |
|-----|------|
| `data_fill_pipeline_template.json` | 管道配置模板 |
| `data_fill_pipeline_targets_*.json` | 特定日期的目标配置 |

### 2. 场景填充配置

| 文件模式 | 说明 |
|---------|------|
| `scenario_fill_smoke*.json` | 冒烟测试配置 (快速验证) |
| `scenario_fill_p0_*.json` | P0 优先级任务配置 |
| `scenario_fill_10h_*.json` | 长时间运行任务 (~10小时) |

### 3. 特殊配置

| 文件 | 用途 |
|-----|------|
| `ood_hs_ai_generate_*.json` | OOD (分布外) 高中 AI 文本生成 |

## 配置结构 (模板)

```json
{
  "run_name": "data_fill_run",
  "output_root": "datasets/planning/data_fill_runs",

  "human_collection": {
    "enabled": true,
    "sources": [
      {
        "source_type": "open_source",
        "notes": "说明文字"
      }
    ]
  },

  "ai_generation": {
    "enabled": false,
    "models": [
      {
        "family": "gpt",
        "name": "gpt-4o",
        "provider": "openai"
      }
    ],
    "decoding_profiles": [
      {
        "name": "conservative",
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 1200
      },
      {
        "name": "diverse",
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 1200
      }
    ],
    "templates": [
      {
        "prompt_id": "TD-SW-001",
        "style": "technical_doc",
        "domain": "software",
        "length_bucket": "500-1000"
      }
    ]
  },

  "targets": [
    {
      "label": "HUMAN",
      "style": "list",
      "domain": "software",
      "length_bucket": "200-500",
      "target_count": 200,
      "priority": "P0",
      "notes": "备注"
    }
  ]
}
```

## 字段说明

### 顶级字段

| 字段 | 类型 | 说明 |
|-----|------|------|
| `run_name` | string | 运行标识名 |
| `output_root` | string | 输出目录 |

### human_collection

| 字段 | 类型 | 说明 |
|-----|------|------|
| `enabled` | boolean | 是否启用人类文本收集 |
| `sources` | array | 数据来源列表 |
| `sources[].source_type` | string | 来源类型 (open_source, etc.) |

### ai_generation

| 字段 | 类型 | 说明 |
|-----|------|------|
| `enabled` | boolean | 是否启用 AI 生成 |
| `models` | array | 使用的模型列表 |
| `models[].family` | string | 模型家族 (gpt, deepseek, qwen, etc.) |
| `models[].name` | string | 具体模型名 |
| `models[].provider` | string | 提供商 (openai, nvidia, etc.) |
| `decoding_profiles` | array | 解码参数配置 |
| `templates` | array | 提示词模板配置 |

### targets

| 字段 | 类型 | 说明 |
|-----|------|------|
| `label` | string | 标签 (HUMAN / AI) |
| `style` | string | 文本风格 (list, technical_doc, etc.) |
| `domain` | string | 领域 (software, ops, ml_ai, etc.) |
| `length_bucket` | string | 长度区间 (200-500, 500-1000, etc.) |
| `target_count` | integer | 目标数量 |
| `priority` | string | 优先级 (P0, P1, P2) |

## 使用示例

```bash
# 运行场景填充生成
python scripts/generation/scenario_fill_generate.py \
  --config configs/scenario_fill_smoke.json

# 运行数据填充管道
python scripts/generation/data_fill_pipeline.py \
  --config configs/data_fill_pipeline_template.json
```

## 命名约定

- **日期后缀**: `_2026-01-27.json` 表示配置创建日期
- **smoke**: 冒烟测试 (少量数据，快速验证)
- **10h**: 长时间任务 (约 10 小时)
- **v2/v3**: 版本迭代

---

*最后更新: 2026-01-28*
