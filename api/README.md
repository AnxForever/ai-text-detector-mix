# API 目录说明

本目录用于存放 API 相关代码与配置。

## 内容
- `api.py` : API 服务入口
- `API_KEYS.md` : API 配置与密钥（内部使用）
- `tests/` : API 测试相关文件

## 启动示例

```bash
python api/api.py
```

## 项目知识问答 Agent

新增接口：`POST /api/project-qa`

用途：
- 从当前仓库自动检索答辩相关资料
- 优先命中 `README.md`、`docs/project/*.md`、`docs/plans/*.md`、
  `docs/thesis/*.md`、`api/api.py` 等项目知识源
- 有 `OPENAI_API_KEY` 时走检索增强回答
- 没有上游模型时回退到本地抽取式回答

请求示例：

```json
{
  "question": "当前推荐模型是什么？三集平均准确率是多少？",
  "topK": 5,
  "useLLM": true,
  "forceRefresh": false
}
```

返回要点：
- `answer`: 问题回答
- `mode`: `rag` 或 `extractive`
- `sources`: 命中的证据片段与来源路径

适合场景：
- 导师临时追问项目细节
- 本地做答辩模拟问答
- 后续接前端或 Coze/工作流平台

补充材料接口：
- `GET /api/project-qa/materials`：查看已上传到知识库的材料
- `POST /api/project-qa/materials`：上传 `pdf/docx/pptx/md/txt/json` 到项目知识库

