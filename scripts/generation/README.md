# 数据生成脚本

> 用于生成 AI 文本样本的核心脚本集

## 核心脚本

| 脚本 | 用途 | 命令示例 |
|-----|------|---------|
| `scenario_fill_generate.py` | **主力生成脚本** - 基于配置文件的场景填充生成 | `python scenario_fill_generate.py --config <config.json>` |
| `monitor_progress.py` | 实时监控生成进度 | `python monitor_progress.py` |
| `monitor_generation.py` | 生成过程监控与日志 | `python monitor_generation.py` |

## 辅助脚本

| 脚本 | 用途 |
|-----|------|
| `parallel_generation.py` | 并行生成（多进程） |
| `data_fill_pipeline.py` | 数据填充管道 |
| `auto_batch_runner.py` | 自动批处理运行器 |

## 使用流程

### 1. 准备配置文件

配置文件位于 `configs/scenario_fill_*.json`，包含：
- `proxy_pool` - API 代理池配置
- `tasks` - 生成任务列表（场景、风格、长度桶、目标数量）
- `concurrency` - 并发数
- `decoding_profiles` - 解码参数

### 2. 启动生成

```bash
cd /mnt/c/datacollection

# 使用指定配置
python scripts/generation/scenario_fill_generate.py \
  --config configs/scenario_fill_v5_BEF_priority_2026-01-28.json
```

### 3. 监控进度

```bash
# 方式1：简单统计
python scripts/generation/monitor_progress.py

# 方式2：实时监控（Linux/macOS）
watch -n 60 'python scripts/generation/monitor_progress.py'

# 方式3：PowerShell 监控
while ($true) { python scripts/generation/monitor_progress.py; Start-Sleep 60 }
```

## 输出目录

生成的数据保存在：
```
datasets/generated/scenario_fill/<run_name>/
├── ai_scenario_fill_<timestamp>_part001.jsonl  # 有效样本
├── ai_scenario_fill_<timestamp>_rejected.jsonl # 被拒样本
└── logs/                                        # 运行日志
```

## 归档脚本

旧版本脚本已移至 `archive/scripts/generation/`，包括：
- `gen_local*.py` 系列 - 本地生成旧版本
- `gen_c*.py` 系列 - 特定场景旧版本
- `generate_*.py` 系列 - 特定用途生成器

---

*更新时间: 2026-01-28*
