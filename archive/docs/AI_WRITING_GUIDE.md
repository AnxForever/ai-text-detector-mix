# AI续写和润色功能使用指南

## 功能说明

前端已集成AI续写和润色功能，通过调用本地反代API实现：
- **润色功能**: 优化文本表达，使其更流畅专业
- **续写功能**: 根据已有内容自然延续写作

## 快速启动

### 1. 启动后端API服务

```bash
cd /mnt/c/datacollection
source .venv/bin/activate
python api.py
```

后端服务将在 `http://localhost:8000` 启动，包含以下端点：
- `/api/detect` - AI文本检测
- `/v1/chat/completions` - AI续写和润色（OpenAI兼容接口）

### 2. 启动前端服务

```bash
cd /mnt/c/datacollection/frontend
npm run dev
```

前端将在 `http://localhost:3000` 启动

### 3. 使用功能

1. 访问 `http://localhost:3000/demo`
2. 输入文本
3. 点击右侧的按钮：
   - **✨ AI润色** - 优化文本表达
   - **✍️ AI续写** - 继续写作
4. 查看对比结果

## API配置

### 默认配置
- **API端点**: `http://localhost:8000/v1/chat/completions`
- **模型**: DeepSeek V3.1
- **API Key**: 已内置在后端（hotaruapi.top）

### 自定义配置
在前端页面点击 **⚙️ API设置** 可以修改：
- API端点地址
- 使用的模型
- API密钥（可选）

## 技术实现

### 后端 (api.py)
```python
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    # 转发请求到 hotaruapi.top
    # 使用 DeepSeek V3.1 模型
    # 返回OpenAI兼容格式
```

### 前端 (app/demo/page.tsx)
- 润色：使用系统提示词优化文本
- 续写：根据上下文自然延续
- 对比视图：显示原文和处理后的差异

## 支持的模型

通过 hotaruapi.top 反代，支持：
- `deepseek-ai/deepseek-v3.1` (默认)
- `deepseek-ai/deepseek-r1`
- `gemini-2.5-flash`
- `gpt-4.1-mini`

## 故障排查

### 后端无法启动
```bash
# 检查依赖
pip install fastapi uvicorn requests

# 检查端口占用
lsof -i :8000
```

### API调用失败
1. 检查后端服务是否运行
2. 查看浏览器控制台错误
3. 验证API密钥是否有效

### 前端无法连接
1. 确认后端在 `http://localhost:8000` 运行
2. 检查CORS配置（已在api.py中配置）
3. 查看网络请求状态

## 示例

### 润色示例
**原文**：
```
这个项目很好用，功能也挺多的，我觉得还不错。
```

**润色后**：
```
该项目具有出色的实用性，功能丰富全面，整体表现令人满意。
```

### 续写示例
**原文**：
```
随着人工智能技术的快速发展，AI文本检测变得越来越重要。
```

**续写**：
```
特别是在学术诚信、内容审核等领域，准确识别AI生成的文本已成为一项关键需求。本系统通过深度学习技术，实现了高精度的检测能力...
```

## 注意事项

1. **API配额**: hotaruapi.top有使用限制，请合理使用
2. **响应时间**: 首次调用可能较慢（模型加载）
3. **文本长度**: 建议单次处理不超过5000字
4. **网络要求**: 需要稳定的网络连接

## 更新日志

- 2026-01-26: 添加AI续写和润色功能
- 集成本地反代API
- 实现对比视图
- 支持自定义API配置
