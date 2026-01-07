# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个**全自动多维文本数据集生成系统**，用于生成高质量的中文文本训练数据。系统通过六维组合（属性 × 话题 × 文体 × 角色 × 风格 × 约束）生成多样化的提示词，并调用多个 LLM API 生成文本数据。

## 运行命令

```bash
# 运行主程序（需要配置 API 密钥）
python "new data collection.py"
```

**注意**：运行前需要在代码中配置以下 API 密钥：
- DeepSeek API Key（第 12 行）
- 通义千问 API Key（第 23 行）

## 核心架构

### 主要类和职责

| 类名 | 位置 | 职责 |
|------|------|------|
| `AutoDimensionGenerator` | 第 56-405 行 | 自动生成六个维度的内容：话题、文体、角色、风格、约束，带缓存机制 |
| `IntelligentCombinationGenerator` | 第 409-665 行 | 智能组合生成器，确保维度组合的合理性和质量评分 |
| `TextQualityAssessor` | 第 702-739 行 | 多维度文本质量评估：长度、结构、内容、AI模板检测 |

### 数据流

```
维度生成 → 智能组合 → 提示词构建 → API调用(带重试) → 质量评估 → 数据保存
```

### 关键函数

- `generate_text_with_retry()` (第 669-697 行)：带指数退避的 API 调用，支持多模型故障转移
- `main_auto()` (第 744-960 行)：主控制器，管理整个生成流程

## 输出文件

- `auto_dataset_*.csv`：生成的数据集（UTF-8-SIG 编码）
- `generation_plan_*.json`：组合计划缓存
- `dataset_generation.log`：运行日志
- `dataset_metadata_*.json`：数据集元信息

## 配置参数

在 `main_auto()` 函数中可调整：
- `samples_per_model`：每个模型生成的样本数（默认 800）
- `save_interval`：自动保存间隔（默认每 50 条）

## 扩展指南

### 添加新模型

1. 在文件开头添加新的 API 客户端初始化
2. 在 `main_auto()` 的 `model_configs` 列表中添加配置
3. 确保新客户端支持 OpenAI 兼容接口

### 添加新维度

1. 在 `AutoDimensionGenerator` 中添加对应的 `generate_*()` 方法
2. 添加对应的 `_get_default_*()` 备用方法
3. 在 `IntelligentCombinationGenerator` 中更新组合逻辑

## Spec Workflow 框架

项目集成了 Spec Workflow 框架，模板位于 `.spec-workflow/templates/`：
- `requirements-template.md`：需求文档模板
- `design-template.md`：设计文档模板
- `tasks-template.md`：任务列表模板

自定义模板可放入 `.spec-workflow/user-templates/`，将覆盖默认模板。
