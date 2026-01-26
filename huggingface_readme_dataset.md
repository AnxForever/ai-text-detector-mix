---
language:
- zh
license: mit
tags:
- chinese
- ai-detection
- text-classification
- dataset
size_categories:
- 10K<n<100K
---

# Chinese AI Detection Dataset

中文AI文本检测数据集

## 数据集简介

用于训练中文AI生成文本检测模型的综合数据集，包含纯人类、纯AI以及混合文本（人类+AI）。

**核心特色**：使用`[SEP]`标记显式标注混合文本的人类/AI边界。

## 数据统计

| 类型 | 样本数 | 说明 |
|------|--------|------|
| 总计 | 66,001 | 训练/验证/测试集 |
| 纯人类 | 27,719 | 多领域人类文本 |
| 纯AI | 27,719 | 多模型生成 |
| C2 (续写) | 3,781 | 人类开头+AI续写 |
| C3 (改写) | 3,781 | AI改写人类文本 |
| C4 (润色) | 3,001 | AI润色人类文本 |

## 数据格式

```json
{
  "text": "文本内容（混合文本包含[SEP]标记）",
  "label": 0,  // 0=Human, 1=AI
  "category": "C2",  // Human/AI/C2/C3/C4
  "source": "数据来源"
}
```

## 快速使用

```python
from datasets import load_dataset

# 加载数据集
dataset = load_dataset("AnxForever/chinese-ai-detection-dataset")

# 查看样本
print(dataset['train'][0])

# 训练/验证/测试集
train_data = dataset['train']
val_data = dataset['validation']
test_data = dataset['test']
```

## 数据来源

- **人类文本**: 知乎、百度知道、新闻、论文等
- **AI文本**: GPT-3.5/4、Claude、文心一言、通义千问等
- **混合文本**: 人工标注 + 自动生成

## 边界标记说明

混合文本使用`[SEP]`标记人类/AI边界：

```
人类写的部分[SEP]AI生成的部分
```

这个创新显著提升了混合文本检测准确率（从79.84%提升到93.84%）。

## 配套模型

- **分类器**: [chinese-ai-detector-bert](https://huggingface.co/AnxForever/chinese-ai-detector-bert)
- **边界检测**: [chinese-ai-detector-span](https://huggingface.co/AnxForever/chinese-ai-detector-span)

## 数据质量

- 人工审核去重
- 多轮清洗过滤
- 平衡各类别分布
- Token级边界标注（2,034条）

## 引用

```bibtex
@misc{chinese-ai-detection-dataset-2026,
  author = {AnxForever},
  title = {Chinese AI Detection Dataset with Boundary Markers},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/AnxForever/chinese-ai-detection-dataset}}
}
```

## 许可证

MIT License

## 项目链接

- GitHub: [ai-text-detector-mix](https://github.com/AnxForever/ai-text-detector-mix)
- 完整文档: [DATA_AND_MODELS.md](https://github.com/AnxForever/ai-text-detector-mix/blob/master/DATA_AND_MODELS.md)

## 使用限制

仅供学术研究使用，不得用于商业目的。
