---
language:
- zh
license: mit
library_name: transformers
pipeline_tag: text-classification
tags:
- ai-generated-text-detection
- chinese
- bert
- text-classification
- binary-classification
- thesis
- academic
base_model: bert-base-chinese
metrics:
- accuracy
- precision
- recall
- f1
model-index:
- name: chinese-ai-detector-bert-v11c
  results:
  - task:
      type: text-classification
      name: Text Classification
    dataset:
      type: AnxForever/chinese-ai-detection-dataset
      name: Validation Set
      split: validation
    metrics:
    - type: accuracy
      value: 0.9875
      name: Accuracy
    - type: f1
      value: 0.9883
      name: F1
  - task:
      type: text-classification
      name: Text Classification
    dataset:
      type: AnxForever/chinese-ai-detection-dataset
      name: Independent Evaluation (910 samples)
      split: test
    metrics:
    - type: accuracy
      value: 0.9857
      name: Accuracy
    - type: f1
      value: 0.9579
      name: F1
  - task:
      type: text-classification
      name: Text Classification
    dataset:
      type: AnxForever/chinese-ai-detection-dataset
      name: Three-Set Average
    metrics:
    - type: accuracy
      value: 0.9856
      name: Accuracy
  - task:
      type: token-classification
      name: Token Boundary Detection
    dataset:
      type: AnxForever/chinese-ai-detection-dataset
      name: Boundary Token Eval
    metrics:
    - type: accuracy
      value: 0.9669
      name: Token Accuracy
---

# Chinese AI-Generated Text Detector — BERT v11c (Boundary-Fix)

> 中文 AI 生成文本检测器（本科毕业设计最终版）
>
> A fine-tuned BERT model that classifies Chinese text as either **human-written (0)** or **AI-generated (1)**, with a boundary-fix training strategy designed to handle long AI-generated passages and mixed human/AI segments.

---

## 📌 模型概述 / Overview

**中文**：本模型是基于 `bert-base-chinese` 微调的中文 AI 生成文本二分类器，为本科毕业设计「基于 BERT 微调的中文 AI 生成文本检测系统」的最终生产模型（v11c boundary-fix 版本）。项目创新性地引入 `[SEP]` 边界标记机制与双层检测架构（分类器 + Token 级边界检测器），显著提升了对人类/AI 混合文本边界的定位能力。

**English**: A binary classifier fine-tuned on `bert-base-chinese` for detecting AI-generated Chinese text. This is the final production checkpoint (v11c boundary-fix) of an undergraduate thesis project featuring `[SEP]` boundary markers and a two-stage detection architecture (classifier + token-level boundary detector).

---

## 📊 评估指标 / Evaluation

| Dataset | Samples | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Validation set | 7,452 | **98.75 %** | 98.30 % | 99.37 % | 98.83 % |
| `core_v1_test_clean` | 545 | 97.98 % | 97.87 % | 98.77 % | 98.32 % |
| Independent eval (910) | 910 | **98.57 %** | 93.08 % | 98.67 % | 95.79 % |
| **Three-set average** | — | **98.56 %** | — | — | — |
| Token-level boundary | — | 96.69 % | — | — | — |

### Independent eval by source (selected)

| Source | Samples | Accuracy |
|---|---:|---:|
| Toutiao News (all) | 377 | 100.0 % |
| Wikipedia CN | 119 | 99.16 % |
| formal_collected | 200 | 96.5 % |
| real_ai_gemini-3-pro-preview | 24 | 100.0 % |
| real_ai_deepseek-v3.2 | 8 | 100.0 % |

---

## 🏗️ 架构 / Architecture

- **Base model**: `bert-base-chinese` (12 layers, hidden 768, 12 heads, vocab 21,128)
- **Head**: `BertForSequenceClassification` (2 labels: `0 = human`, `1 = AI`)
- **Max sequence length**: 256 tokens (train), 512 (supported)
- **Framework**: `transformers 4.57.3`, PyTorch 2.0+
- **Parameters**: ~102M

### Training configuration

| Setting | Value |
|---|---|
| Base model | `bert-base-chinese` (via `bert_v7_improved` intermediate checkpoint) |
| Train samples | 63,113 |
| Validation samples | 7,452 |
| Epochs | 5 (best at epoch 2) |
| Batch size | 8 × 4 grad accum |
| Learning rate | 1e-5 |
| Label smoothing | 0.05 |
| Max length | 256 |
| Early stopping patience | 2 |

### Data changes vs. v10 baseline

- Removed 750 hard patterns + 1,767 unapproved samples + 7 length violations
- Added 300 formal-collected weak-domain samples
- Added 300 Llama-405B weak-domain samples
- Added **2,131 long-AI boundary-fix samples** (the key v11c contribution)
- Net change: +207 rows vs. v10

---

## 🚀 使用方法 / Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_ID = "AnxForever/chinese-ai-detector-bert"
TEMPERATURE = 0.8165  # Temperature scaling, calibrated on 910 samples (ECE=0.0034)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()

text = "这是一段需要检测的中文文本。"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)

with torch.no_grad():
    logits = model(**inputs).logits
    # Apply temperature scaling for calibrated confidence
    probs = torch.softmax(logits / TEMPERATURE, dim=-1)[0]

pred_idx = int(probs.argmax())
label = model.config.id2label[pred_idx]   # "human-written" or "AI-generated"
print(f"{label}  (confidence: {probs[pred_idx].item():.2%})")
```

### Label mapping
- `0` → human-written (人类撰写)
- `1` → AI-generated (AI 生成)

> **Note on Temperature Scaling**: `T = 0.8165` was calibrated on a held-out 910-sample set
> and brings ECE from 0.0121 down to **0.0034**. For uncalibrated probabilities, set `TEMPERATURE = 1.0`.

---

## 🎯 技术创新 / Contributions

1. **`[SEP]` 边界标记机制 / Boundary-marker mechanism**
   在人类/AI 混合文本的分界处显式插入 `[SEP]`，使模型对 C2 类混合文本的检测准确率提升约 14 %。

2. **双层检测架构 / Two-stage detection**
   - Stage 1: 本模型 — 篇章级二分类
   - Stage 2: 配套的 span detector — Token 级边界定位
   - 参见 [`AnxForever/chinese-ai-detector-span`](https://huggingface.co/AnxForever/chinese-ai-detector-span)

3. **Long-AI boundary-fix (v11c)**
   针对长 AI 段落在边界处易被误判的问题，补充 2,131 条长 AI 边界样本，使 256+ token 桶的准确率恢复到 V10 水平。

---

## ⚠️ 局限性 / Limitations

- 仅针对**中文**文本；对英文或其他语言无保证。
- 训练语料偏新闻/百科/技术/正式文体，对**诗歌、古文、社交媒体短文本**可能欠拟合。
- 对**人机混写**的检测效果优于纯分类，但建议配合 span detector 获取更细粒度结果。
- 训练数据主要来自 DeepSeek、Gemini、GPT、Llama-405B 等主流模型；对**经过重度改写**的 AI 文本仍有遗漏风险。

---

## 🗂️ 相关资源 / Related

- 📊 训练数据集 / Dataset: [`AnxForever/chinese-ai-detection-dataset`](https://huggingface.co/datasets/AnxForever/chinese-ai-detection-dataset)
- 🎯 边界检测器 / Span detector: [`AnxForever/chinese-ai-detector-span`](https://huggingface.co/AnxForever/chinese-ai-detector-span)

---

## 📜 License

MIT License

## ✍️ Citation

```bibtex
@misc{anxforever2026chineseaidetectorbert,
  title  = {Chinese AI-Generated Text Detector with Boundary Markers (BERT v11c)},
  author = {AnxForever},
  year   = {2026},
  howpublished = {\url{https://huggingface.co/AnxForever/chinese-ai-detector-bert}},
  note   = {Undergraduate thesis project}
}
```
