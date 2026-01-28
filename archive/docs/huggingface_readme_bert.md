---
language:
- zh
license: mit
tags:
- text-classification
- chinese
- ai-detection
- bert
metrics:
- accuracy
- f1
pipeline_tag: text-classification
---

# Chinese AI Text Detector - BERT Classifier

中文AI文本检测器 - BERT分类模型

## 模型简介

基于BERT的中文AI生成文本检测模型，支持检测纯人类、纯AI以及混合文本（人类+AI）。

**核心创新**：使用`[SEP]`标记显式标注人类/AI边界，显著提升混合文本检测准确率。

## 性能指标

| 指标 | 数值 |
|------|------|
| 整体准确率 | **98.71%** |
| C2 (续写) | 93.84% |
| C3 (改写) | 100% |
| C4 (润色) | 92.89% |

## 快速使用

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 加载模型
model_name = "AnxForever/chinese-ai-detector-bert"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# 检测文本
text = "这是一段需要检测的中文文本..."
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    
label = "AI" if probs[0][1] > 0.5 else "Human"
confidence = probs[0][1].item() if label == "AI" else probs[0][0].item()

print(f"检测结果: {label} (置信度: {confidence:.2%})")
```

## 模型架构

- **基础模型**: bert-base-chinese
- **分类层**: 2分类 (Human/AI)
- **最大长度**: 512 tokens
- **训练数据**: 66,001条样本

## 训练细节

- **优化器**: AdamW
- **学习率**: 2e-5
- **Batch Size**: 16
- **Epochs**: 5
- **损失函数**: CrossEntropyLoss

## 数据集

训练数据包含：
- 纯人类文本
- 纯AI文本（多个模型生成）
- 混合文本（C2/C3/C4类型）

数据集: [chinese-ai-detection-dataset](https://huggingface.co/datasets/AnxForever/chinese-ai-detection-dataset)

## 边界检测

如需精确定位混合文本的人类/AI边界，请配合使用：
[chinese-ai-detector-span](https://huggingface.co/AnxForever/chinese-ai-detector-span)

## 引用

```bibtex
@misc{chinese-ai-detector-2026,
  author = {AnxForever},
  title = {Chinese AI Text Detection with Boundary Markers},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/AnxForever/chinese-ai-detector-bert}}
}
```

## 许可证

MIT License

## 项目链接

- GitHub: [ai-text-detector-mix](https://github.com/AnxForever/ai-text-detector-mix)
- 完整文档: [FINAL_RESULTS.md](https://github.com/AnxForever/ai-text-detector-mix/blob/master/FINAL_RESULTS.md)
