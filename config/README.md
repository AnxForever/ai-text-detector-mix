# 配置文件说明

本目录包含项目运行所需的 API 配置文件。

## 文件清单

| 文件 | 说明 |
|-----|------|
| `api.local.json` | 本地配置文件（包含实际密钥，**不要提交到Git**） |
| `api.local.json.example` | 配置模板（安全可提交） |

## 配置结构

```json
{
  "nvidia": {
    "api_key": "YOUR_NVIDIA_KEY",
    "base_url": "https://integrate.api.nvidia.com/v1"
  },
  "local_proxy": {
    "base_url": "http://192.168.60.105:8317/v1",
    "api_key": "YOUR_LOCAL_PROXY_KEY"
  },
  "remote_proxy": {
    "base_url": "https://api.hotaruapi.top/v1",
    "api_key": "YOUR_REMOTE_PROXY_KEY"
  },
  "hybgzs_proxy": {
    "base_url": "https://ai.hybgzs.com/v1",
    "api_key": "YOUR_HYBGZS_PROXY_KEY"
  }
}
```

## 字段说明

### nvidia

| 字段 | 类型 | 说明 |
|-----|------|------|
| `api_key` | string | NVIDIA API 密钥 |
| `base_url` | string | NVIDIA API 端点 |

**用途**: 调用 NVIDIA NIM 平台上的模型（如 DeepSeek、Llama 等）

### local_proxy

| 字段 | 类型 | 说明 |
|-----|------|------|
| `api_key` | string | 本地代理密钥 |
| `base_url` | string | 本地 OpenAI 兼容代理地址 |

**用途**: 连接本地部署的 LLM 服务（如 vLLM、Ollama 等）

### remote_proxy

| 字段 | 类型 | 说明 |
|-----|------|------|
| `api_key` | string | 远程代理密钥 |
| `base_url` | string | 远程 OpenAI 兼容代理地址 |

**用途**: 连接第三方 API 代理服务

### hybgzs_proxy

| 字段 | 类型 | 说明 |
|-----|------|------|
| `api_key` | string | hybgzs 代理密钥 |
| `base_url` | string | hybgzs API 端点 |

**用途**: 连接 hybgzs.com 代理服务

## 快速开始

```bash
# 复制模板文件
cp api.local.json.example api.local.json

# 编辑配置，填入实际的 API 密钥
nano api.local.json  # 或使用其他编辑器
```

## 在脚本中使用

```python
from scripts.utils.api_config import load_api_config

config = load_api_config()

# 获取特定提供商配置
nvidia_config = config.get("nvidia", {})
api_key = nvidia_config.get("api_key")
base_url = nvidia_config.get("base_url")
```

## 安全提示

1. **永远不要提交** `api.local.json` 到版本控制
2. `.gitignore` 中已包含此文件的忽略规则
3. 如需分享配置结构，请使用 `.example` 文件
4. 定期轮换 API 密钥

---

*最后更新: 2026-01-28*
