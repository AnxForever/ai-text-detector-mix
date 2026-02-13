[根目录](../CLAUDE.md) > **api**

# API 服务模块

## 模块职责

提供 FastAPI 后端服务，实现 AI 文本检测 API 和 OpenAI 兼容的聊天接口。

## 入口与启动

```bash
# 启动服务
cd /mnt/c/datacollection
python api/api.py
# 服务监听: http://0.0.0.0:8000
```

## 对外接口

### 1. 文本检测 API

**POST** `/api/detect`

请求:
```json
{
  "text": "待检测的文本内容"
}
```

响应:
```json
{
  "type": "human|ai|mixed",
  "confidence": 95.5,
  "humanPercentage": 30,
  "aiPercentage": 70,
  "boundary": 2,
  "sentences": [
    {"text": "句子1", "isAI": false, "confidence": 98.0},
    {"text": "句子2", "isAI": true, "confidence": 95.0}
  ],
  "processingTime": 150
}
```

### 2. OpenAI 兼容聊天接口

**POST** `/v1/chat/completions`

用于 AI 续写和润色功能，转发到配置的代理 API。

## 关键依赖与配置

### Python 依赖

- `torch` - PyTorch 深度学习框架
- `transformers` - Hugging Face 模型库
- `fastapi` - Web 框架
- `uvicorn` - ASGI 服务器
- `pydantic` - 数据验证

### 模型依赖

- `models/bert_v11c_boundary_fix/` - 分类器模型 (默认)
- `models/bert_span_detector/` - 边界检测器模型

### 环境变量

```bash
DETECTOR_CLASSIFIER_MODEL=models/bert_v11c_boundary_fix
DETECTOR_SPAN_MODEL=models/bert_span_detector
DETECTOR_MAX_LENGTH=256
DETECTOR_TEMPERATURE=0.8165
DETECTOR_DECISION_THRESHOLD=0.8
DETECTOR_INCLUDE_RISK_OBSERVABILITY=0   # 1 to include risk fields in /api/detect response
OPENAI_API_KEY=sk-xxx           # OpenAI兼容接口密钥
OPENAI_BASE_URL=https://...     # 代理API地址
```

## 数据模型

### DetectRequest
```python
class DetectRequest(BaseModel):
    text: str
```

### DetectionResponse
```python
class DetectionResponse(BaseModel):
    type: str                    # "human" | "ai" | "mixed"
    confidence: float
    humanPercentage: int
    aiPercentage: int
    boundary: int | None
    sentences: list[SentenceResult]
    processingTime: int
    modelVersion: str | None     # 模型版本标识
    decisionThreshold: float | None
    riskFlags: list[str] | None  # 风险提示 (short_text/template_like 等)
    domainHint: str | None       # 粗粒度文本域提示
```

### HybridTextDetector

核心检测类，包含:
- `classify(text)` - 文本分类
- `detect_boundary(text)` - 边界检测

## 测试与质量

### 测试文件

- `api/tests/test_v0_api.py` - API 配置检查

### 运行测试

```bash
python api/tests/test_v0_api.py
```

## 常见问题 (FAQ)

**Q: 模型加载失败?**
A: 确保 `models/bert_v11c_boundary_fix/` 和 `models/bert_span_detector/` 目录存在且包含完整模型文件。

**Q: CUDA 不可用?**
A: 系统会自动回退到 CPU 模式，性能会下降但功能正常。

**Q: API 密钥无效?**
A: 检查 `config/api.local.json` 或环境变量配置。

## 相关文件清单

```
api/
├── api.py                 # 主服务入口
├── tests/
│   └── test_v0_api.py     # API测试
└── API_KEYS.md            # API密钥说明 (敏感)
```

## 变更记录 (Changelog)

### 2026-02-13
- 分类器默认路径更新为 `models/bert_v11c_boundary_fix`
- Temperature 更新为 0.8165

### 2026-02-12
- 分类器默认路径更新为 `models/bert_v10_augmented`
- 补充检测模型相关环境变量说明
- 检测响应新增 `modelVersion/decisionThreshold/riskFlags/domainHint`

### 2026-01-28
- 初始化模块文档

---

*文档更新时间: 2026-02-13*
