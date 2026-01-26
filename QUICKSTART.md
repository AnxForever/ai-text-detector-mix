# 快速开始指南

## 安装依赖

```bash
pip install transformers torch
```

## 使用模型

### 1. 文本分类（检测是否AI生成）

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

print(f"检测结果: {label}")
print(f"置信度: {confidence:.2%}")
```

### 2. 边界检测（定位混合文本边界）

```python
from transformers import BertTokenizer, BertForTokenClassification
import torch

# 加载边界检测模型
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

## 加载数据集

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

## 本地运行演示

```bash
# 克隆仓库
git clone https://github.com/AnxForever/ai-text-detector-mix.git
cd ai-text-detector-mix

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements_training.txt

# 运行可视化演示
python scripts/demo/visualize_detection.py

# 运行完整评估
python scripts/evaluation/eval_complete.py
```

## 性能指标

| 指标 | 数值 |
|------|------|
| 整体准确率 | 98.71% |
| C2 (续写) | 93.84% |
| C3 (改写) | 100% |
| C4 (润色) | 92.89% |
| Token分类 | 96.69% |

## 更多信息

- 完整成果报告: [FINAL_RESULTS.md](FINAL_RESULTS.md)
- 数据集说明: [DATA_AND_MODELS.md](DATA_AND_MODELS.md)
- 训练计划: [TRAINING_PLAN.md](TRAINING_PLAN.md)
