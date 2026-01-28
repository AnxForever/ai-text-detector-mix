# 全自动多维文本数据集生成系统

## 项目简介

本系统用于生成高质量的中文文本训练数据，采用**六维组合策略**（属性 × 话题 × 文体 × 角色 × 风格 × 约束）生成多样化的提示词，并调用多个 LLM API 生成文本数据。

---

## 目录结构

```
C:\datacollection\
├── new data collection.py      # 主程序（核心文件）
├── START_GENERATION.bat        # Windows 一键启动脚本
├── .venv/                      # Python 虚拟环境
├── auto_generated_datasets/    # 生成的数据集输出目录
│   ├── auto_dataset_*.csv      # 各模型生成的数据集
│   ├── AUTO_COMBINED_*.csv     # 合并后的完整数据集
│   └── auto_generation_plan.json # 生成计划缓存
├── dimension_cache/            # 维度缓存（加速后续生成）
├── dataset_generation.log      # 运行日志
├── CLAUDE.md                   # AI 助手指导文件
└── AGENTS.md                   # AI Agent 指导文件
```

---

## 快速开始

### 方式一：双击启动（推荐）

直接双击 `START_GENERATION.bat` 即可运行。

### 方式二：命令行启动

```bash
cd C:\datacollection
python "new data collection.py"
```

---

## 配置说明

### API 配置（推荐：config/api.txt）

在 `config/api.txt` 中维护 API 列表，例如：

```
name
key:sk-***
url:https://your-endpoint/v1
```

**修改 API**：直接在 `config/api.txt` 中增删配置，避免在代码中硬编码密钥。

### 模型配置（第 18-44 行）

当前配置了 4 个模型：

| 模型标识 | 模型名称 | 说明 |
|---------|---------|------|
| deepseek | deepseek-v3.2-chat | 高性价比，推荐优先使用 |
| claude | claude-sonnet-4-5 | 高质量输出 |
| qwen | qwen-max-latest | 通义千问最新版 |
| glm | GLM-4.7 | 智谱 GLM |

**添加/修改模型**：在 `MODEL_CONFIGS` 字典中添加新配置。

### 生成参数（第 771 行）

```python
SAMPLES_PER_MODEL = 800  # 每个模型生成的样本数
```

---

## 核心组件

### 1. AutoDimensionGenerator（自动维度生成器）

- **位置**：第 64-405 行
- **功能**：使用 LLM 自动生成六个维度的内容
- **维度**：话题、文体、角色、风格、约束
- **特性**：带本地缓存机制，避免重复生成

### 2. IntelligentCombinationGenerator（智能组合生成器）

- **位置**：第 409-665 行
- **功能**：智能组合六维内容，确保组合合理性
- **特性**：自动计算组合质量评分，优先生成高质量组合

### 3. TextQualityAssessor（文本质量评估器）

- **位置**：第 702-739 行
- **功能**：多维度评估生成文本质量
- **评估维度**：长度、结构、内容、AI 模板检测

### 4. generate_text_with_retry（API 调用函数）

- **位置**：第 669-697 行
- **功能**：带指数退避的 API 调用
- **特性**：自动重试、多模型故障转移

---

## 数据流程

```
┌─────────────────┐
│ 1. 维度自动生成  │  AutoDimensionGenerator
│   (带缓存)      │  生成话题、文体、角色、风格、约束
└────────┬────────┘
         ↓
┌─────────────────┐
│ 2. 智能组合      │  IntelligentCombinationGenerator
│   (质量评分)    │  生成 800+ 高质量组合
└────────┬────────┘
         ↓
┌─────────────────┐
│ 3. 提示词构建    │  基于六维组合生成具体提示词
└────────┬────────┘
         ↓
┌─────────────────┐
│ 4. API 调用      │  调用多模型 API
│   (带重试)      │  支持故障转移
└────────┬────────┘
         ↓
┌─────────────────┐
│ 5. 质量评估      │  TextQualityAssessor
│                 │  评估生成文本质量
└────────┬────────┘
         ↓
┌─────────────────┐
│ 6. 数据保存      │  保存到 CSV 文件
│   (自动保存)    │  每 50 条自动保存一次
└─────────────────┘
```

---

## 输出文件说明

### CSV 数据集字段

| 字段名 | 说明 |
|-------|------|
| text_id | 唯一标识符 |
| text_content | 生成的文本内容 |
| source_model | 生成模型标识 |
| attribute | 文本属性（描写/记叙/说明/抒情/议论） |
| topic | 话题 |
| genre | 文体 |
| role | 角色 |
| style | 风格 |
| constraint | 约束条件 |
| prompt | 使用的提示词 |
| combination_quality | 组合质量评分 |
| generation_quality | 生成质量评分 |
| timestamp | 生成时间戳 |

---

## 断点续传

系统支持断点续传：

1. 生成计划保存在 `auto_generated_datasets/auto_generation_plan.json`
2. 程序中断后重新运行会自动从中断处继续
3. 已生成的数据不会重复生成

---

## 扩展指南

### 添加新模型

1. 在 `MODEL_CONFIGS` 中添加配置：

```python
"new_model": {
    "client_class": openai.OpenAI,
    "api_key": API_KEY,
    "base_url": API_BASE,
    "model_name": "model-name-here"
}
```

2. 在 `clients` 字典中添加：

```python
clients = {..., "new_model": True}
```

### 修改六维内容

修改 `AutoDimensionGenerator` 类中对应的 `generate_*()` 方法和 `_get_default_*()` 备用方法。

### 修改组合逻辑

修改 `IntelligentCombinationGenerator` 类中的 `generate_combinations()` 方法。

---

## 常见问题

### Q: API 调用失败怎么办？

A: 系统内置重试机制，会自动重试 3 次。如果持续失败，检查：
- API 密钥是否有效
- API 端点是否可访问
- 网络连接是否正常

### Q: 如何修改每个模型的生成数量？

A: 修改第 771 行的 `SAMPLES_PER_MODEL` 变量。

### Q: 生成的数据在哪里？

A: 在 `auto_generated_datasets/` 目录下，文件名格式为 `auto_dataset_{模型名}_{时间}.csv`。

### Q: 如何清除缓存重新生成维度？

A: 删除 `dimension_cache/` 目录和 `auto_generated_datasets/auto_generation_plan.json` 文件。

---

## 依赖安装

```bash
pip install openai pandas requests
```

或使用虚拟环境：

```bash
cd C:\datacollection
.venv\Scripts\activate
pip install openai pandas requests
```

---

## 注意事项

1. **API 费用**：生成大量数据会产生 API 调用费用，请注意控制
2. **速率限制**：系统内置 0.5 秒延迟，避免触发 API 速率限制
3. **存储空间**：每 1000 条数据约占用 5-10 MB 存储空间
4. **编码格式**：CSV 文件使用 UTF-8-SIG 编码，Excel 可直接打开

---

## 联系方式

如有问题，请参考 `CLAUDE.md` 或 `AGENTS.md` 文件中的项目说明。
