# API Manager - 多API中转站管理与批量数据生成工具

管理多个 OpenAI 兼容 API 并批量生成 AI 文本数据集。

## 功能

- **可视化TUI界面**：交互式终端界面，方便操作
- **API 配置管理**：添加/删除/测试多个 API 中转站
- **模型查询**：查看各 API 可用模型，按厂商分类
- **批量生成**：多 API 并行生成，支持断点续传
- **数据集类型**：支持多种文本风格（新闻、学术、社交媒体等）
- **安全存储**：支持环境变量存储 API 密钥

## 安装依赖

```bash
pip install rich httpx click textual
```

## 使用方式

### 可视化界面（推荐）

```bash
# 启动TUI界面
python -m tools.api_manager.cli

# 或明确指定
python -m tools.api_manager.cli tui
```

**TUI界面功能：**
- **API配置**：填写API名称、URL、Key，保存配置
- **模型选择**：显示可用模型，按厂商（OpenAI/Anthropic/阿里云等）分类
- **数据生成**：选择数据集类型、输入主题、开始生成

### 命令行模式

```bash
# 查看所有 API
python -m tools.api_manager.cli list-apis

# 添加 API（推荐：使用环境变量）
export MY_API_KEY="sk-xxx"
python -m tools.api_manager.cli add-api \
  --name my_api \
  --url https://api.example.com/v1 \
  --key-env MY_API_KEY \
  --rate-limit 10

# 添加 API（交互式输入密钥）
python -m tools.api_manager.cli add-api \
  --name my_api \
  --url https://api.example.com/v1
# 将提示输入 API Key（密码模式，不显示）

# 删除 API
python -m tools.api_manager.cli remove-api my_api

# 测试 API 连接
python -m tools.api_manager.cli test-api my_api  # 指定API
python -m tools.api_manager.cli test-api         # 测试所有

# 查询模型
python -m tools.api_manager.cli list-models            # 所有API
python -m tools.api_manager.cli list-models my_api     # 指定API
python -m tools.api_manager.cli list-models -r         # 强制刷新

# 批量生成
python -m tools.api_manager.cli generate \
  --topic "人工智能" --topic "科技发展" \
  --count 20 \
  --concurrent 5 \
  --output output/ai_data.csv
```

## 数据集类型

| 类型 | 说明 |
|------|------|
| 通用文章 | 常规主题文章，200-300字 |
| 新闻报道 | 客观、简洁的新闻风格 |
| 学术论文 | 严谨、专业的学术语言 |
| 社交媒体 | 轻松活泼的帖子风格 |
| 产品评测 | 包含优缺点分析的评测 |
| 故事创作 | 引人入胜的短篇故事 |
| 教程指南 | 步骤清晰的教程 |
| 自定义 | 完全自定义模板 |

## 模型厂商分类

TUI界面会自动将模型按厂商分类显示：

- **OpenAI**: GPT-4o, GPT-4, GPT-3.5-Turbo 等
- **Anthropic**: Claude-3.5-Sonnet, Claude-3-Opus 等
- **Google**: Gemini-Pro, PaLM 等
- **DeepSeek**: DeepSeek-Chat 等
- **阿里云**: Qwen-Plus, 通义千问 等
- **智谱AI**: GLM-4, ChatGLM 等
- **百度**: ERNIE, 文心一言 等
- **Meta**: LLaMA 等
- **Mistral**: Mistral, Mixtral 等
- **零一万物**: Yi 系列
- **月之暗面**: Moonshot, Kimi 等
- **讯飞**: Spark 等

## 配置文件

配置存储在 `tools/configs/apis.json`：

```json
{
  "apis": [
    {
      "name": "my_proxy",
      "base_url": "https://api.example.com/v1",
      "api_key": "",
      "api_key_env": "MY_API_KEY",
      "enabled": true,
      "rate_limit": 10,
      "models_cache": ["gpt-4o", "gpt-3.5-turbo"]
    }
  ]
}
```

**安全说明**：
- 推荐使用 `api_key_env` 字段指定环境变量名
- 如果设置了 `api_key_env`，`api_key` 字段将为空，密钥从环境变量读取
- 避免将实际 API 密钥提交到版本控制

## 输出格式

支持三种输出格式（根据文件扩展名自动选择）：

- `.csv`：CSV 格式，含 text, label, category, source 列
- `.json`：JSON 格式，含元数据和完整记录
- `.jsonl`：JSONL 格式，每行一条记录

## 快捷键（TUI界面）

| 快捷键 | 功能 |
|--------|------|
| `q` | 退出 |
| `r` | 刷新 |
| `Ctrl+S` | 保存配置 |
| `Tab` | 切换标签页 |

## 文件结构

```
tools/
├── api_manager/
│   ├── __init__.py    # 模块入口
│   ├── config.py      # API配置管理
│   ├── models.py      # 模型查询
│   ├── generator.py   # 数据生成
│   ├── tui.py         # 可视化TUI界面
│   └── cli.py         # 命令行界面
├── configs/
│   └── apis.json      # API配置文件
└── README.md          # 本文档
```
