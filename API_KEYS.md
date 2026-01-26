# API配置列表 - 完整版

## 当前活跃API (hotaruapi.top)

### 1. DeepSeek API (推荐用于数据集生成)
- **API Key**: `sk-WPzv2WwpDbnLp02uro0DyPUy0LyI3VjIRmngMj8fm7BLQqSq`
- **Base URL**: `https://api.hotaruapi.top/v1`
- **可用模型**:
  - `deepseek-ai/deepseek-v3.1` ✅ 已测试
  - `deepseek-ai/deepseek-r1`
  - `gemini-2.5-flash`
  - `gemini-2.5-pro`
  - `gpt-4.1-mini`
  - `gpt-5-codex-mini`

### 2. OpenAI API (Codex专用)
- **API Key**: `sk-NWUVDRLLpGRQvK1nAqSzLM5mn6TA5F5wgHMWi8XxJ6v1fRSY`
- **Base URL**: `https://api.hotaruapi.top`
- **可用模型**:
  - `gpt-5-codex`
  - `gpt-5.1-codex-max`
  - `gpt-5.2-codex`

### 3. Claude API
- **API Key**: `sk-cGIZ4ovB13Salr7l7J5cpsjBwml9QxPIs5JGi4I9cnqN9l2J`
- **Base URL**: `https://api.hotaruapi.top/v1`
- **可用模型**: Claude系列（待测试）

---

## 旧方案API (wzw.pp.ua) - 多模型代理

### 4. 多模型代理API (支持80+模型)
- **API Key**: `sk-***` (需要你提供完整key)
- **Base URL**: `https://wzw.pp.ua/v1`
- **可用模型**:
  - `deepseek-v3.2-chat` - DeepSeek V3.2
  - `claude-sonnet-4-5` - Claude Sonnet 4.5
  - `qwen-max-latest` - 通义千问最新版
  - `GLM-4.7` - 智谱GLM 4.7
  - `gpt-4o-mini` - GPT-4o-mini

---

## 早期API配置 (已归档)

### 5. 自定义API端点 (china.184772.xyz)
- **API Key**: `sk-***` (需要你提供完整key)
- **Base URL**: `https://china.184772.xyz/v1`
- **可用模型**: `gpt-4o-mini`

### 6. DeepSeek 官方API
- **API Key**: `sk-***` (需要你提供完整key)
- **Base URL**: `https://api.deepseek.com`
- **可用模型**: `deepseek-chat`

### 7. 通义千问官方API (DashScope)
- **API Key**: `sk-***` (需要你提供完整key)
- **Base URL**: `https://dashscope.aliyuncs.com/compatible-mode/v1`
- **可用模型**: `qwen3-max`

---

## 使用建议

### 数据集生成
推荐使用 **DeepSeek API (Key #1)**:
- 模型质量好
- 已验证可用
- 适合生成中文文本

### 配置方法

**环境变量方式**:
```bash
export OPENAI_API_KEY="sk-WPzv2WwpDbnLp02uro0DyPUy0LyI3VjIRmngMj8fm7BLQqSq"
export OPENAI_BASE_URL="https://api.hotaruapi.top/v1"
```

**Python代码方式**:
```python
import requests

API_KEY = "sk-WPzv2WwpDbnLp02uro0DyPUy0LyI3VjIRmngMj8fm7BLQqSq"
API_BASE = "https://api.hotaruapi.top/v1"
MODEL = "deepseek-ai/deepseek-v3.1"

response = requests.post(
    f"{API_BASE}/chat/completions",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    },
    json={
        "model": MODEL,
        "messages": [{"role": "user", "content": "你好"}],
        "max_tokens": 500
    }
)
```

---

## 测试脚本

运行完整API测试:
```bash
python scripts/testing/test_all_apis.py
```

---

**更新时间**: 2026-01-25 (完整版恢复)
