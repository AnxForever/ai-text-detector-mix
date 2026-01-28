---
language:
- zh
license: mit
tags:
- token-classification
- chinese
- boundary-detection
- bert
metrics:
- accuracy
pipeline_tag: token-classification
---

# Chinese AI Text Detector - Span Detector

中文AI文本检测器 - 边界检测模型

## 模型简介

基于BERT的Token级边界检测模型，用于精确定位混合文本（人类+AI）中的人类/AI边界位置。

## 性能指标

| 指标 | 数值 |
|------|------|
| Token分类准确率 | **96.69%** |
| 实际边界误差 | <10字符 |

## 快速使用

```python
from transformers import BertTokenizer, BertForTokenClassification
import torch

# 加载模型
model_name = "AnxForever/chinese-ai-detector-span"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForTokenClassification.from_pretrained(model_name)

# 检测边界
text = "人类写的部分[SEP]AI生成的部分"
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)

# 找到边界位置
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
labels = predictions[0].tolist()

boundary_idx = None
for i, (token, label) in enumerate(zip(tokens, labels)):
    if label == 1:  # AI部分开始
        boundary_idx = i
        break

print(f"边界位置: Token {boundary_idx}")
```

## 标签说明

- **0**: Human (人类写的部分)
- **1**: AI (AI生成的部分)

## 配合使用

建议先使用分类器判断文本类型：
[chinese-ai-detector-bert](https://huggingface.co/AnxForever/chinese-ai-detector-bert)

如果检测为混合文本，再使用本模型定位边界。

## 训练细节

- **基础模型**: bert-base-chinese
- **标注数据**: 2,034条Token级标注
- **最大长度**: 512 tokens
- **学习率**: 2e-5

## 数据集

[chinese-ai-detection-dataset](https://huggingface.co/datasets/AnxForever/chinese-ai-detection-dataset)

## 引用

```bibtex
@misc{chinese-ai-detector-span-2026,
  author = {AnxForever},
  title = {Chinese AI Text Boundary Detection},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/AnxForever/chinese-ai-detector-span}}
}
```

## 许可证

MIT License

## 项目链接

- GitHub: [ai-text-detector-mix](https://github.com/AnxForever/ai-text-detector-mix)
