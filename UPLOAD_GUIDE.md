# 上传到Hugging Face指南

## 前置准备

### 1. 注册Hugging Face账号
访问: https://huggingface.co/join

### 2. 获取Access Token
1. 访问: https://huggingface.co/settings/tokens
2. 点击 "New token"
3. 名称: `datacollection-upload`
4. 权限: 选择 **Write**
5. 复制生成的token

### 3. 登录
```bash
cd /mnt/c/datacollection
source .venv/bin/activate
huggingface-cli login
# 粘贴你的token
```

---

## 方法1: 自动化脚本 (推荐)

```bash
cd /mnt/c/datacollection
./upload_to_huggingface.sh
```

脚本会自动：
- 创建3个仓库
- 上传2个模型 (~779MB)
- 上传数据集 (~106MB)

---

## 方法2: 手动上传

### 创建仓库
```bash
huggingface-cli repo create chinese-ai-detector-bert --type model
huggingface-cli repo create chinese-ai-detector-span --type model
huggingface-cli repo create chinese-ai-detection-dataset --type dataset
```

### 上传模型
```bash
# BERT分类器
huggingface-cli upload YOUR_USERNAME/chinese-ai-detector-bert \
    models/bert_v2_with_sep/

# Span检测器
huggingface-cli upload YOUR_USERNAME/chinese-ai-detector-span \
    models/bert_span_detector/
```

### 上传数据集
```bash
huggingface-cli upload YOUR_USERNAME/chinese-ai-detection-dataset \
    datasets/combined_v2/
```

---

## 方法3: 使用Python API

```python
from huggingface_hub import HfApi

api = HfApi()

# 上传模型
api.upload_folder(
    folder_path="models/bert_v2_with_sep",
    repo_id="YOUR_USERNAME/chinese-ai-detector-bert",
    repo_type="model"
)

# 上传数据集
api.upload_folder(
    folder_path="datasets/combined_v2",
    repo_id="YOUR_USERNAME/chinese-ai-detection-dataset",
    repo_type="dataset"
)
```

---

## 上传后的链接

上传完成后，你的资源将在：

- **模型1**: `https://huggingface.co/YOUR_USERNAME/chinese-ai-detector-bert`
- **模型2**: `https://huggingface.co/YOUR_USERNAME/chinese-ai-detector-span`
- **数据集**: `https://huggingface.co/datasets/YOUR_USERNAME/chinese-ai-detection-dataset`

---

## 添加README (推荐)

在Hugging Face网页上为每个仓库添加README.md：

### 模型README模板
```markdown
---
language: zh
license: mit
tags:
- text-classification
- chinese
- ai-detection
---

# Chinese AI Text Detector

## Model Description
BERT-based classifier for detecting AI-generated Chinese text.

## Performance
- Accuracy: 98.71%
- C2 Detection: 93.84%

## Usage
\`\`\`python
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained("YOUR_USERNAME/chinese-ai-detector-bert")
model = BertForSequenceClassification.from_pretrained("YOUR_USERNAME/chinese-ai-detector-bert")
\`\`\`

## Citation
See [CITATION.md](https://github.com/AnxForever/ai-text-detector-mix/blob/master/CITATION.md)
```

### 数据集README模板
```markdown
---
language: zh
license: cc-by-nc-sa-4.0
task_categories:
- text-classification
---

# Chinese AI Detection Dataset

## Dataset Description
66,001 samples for Chinese AI-generated text detection.

## Dataset Structure
- Train: 52,800
- Val: 6,600
- Test: 6,601

## Usage
\`\`\`python
from datasets import load_dataset

dataset = load_dataset("YOUR_USERNAME/chinese-ai-detection-dataset")
\`\`\`
```

---

## 常见问题

### Q: 上传速度慢？
A: 总大小~1.4GB，根据网速可能需要10-30分钟

### Q: 上传失败？
A: 检查：
1. Token权限是否为Write
2. 网络连接是否稳定
3. 磁盘空间是否足够

### Q: 如何更新已上传的文件？
A: 重新运行上传命令即可覆盖

---

## 下载使用

上传后，其他人可以这样使用：

```python
# 下载模型
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained("YOUR_USERNAME/chinese-ai-detector-bert")
model = BertForSequenceClassification.from_pretrained("YOUR_USERNAME/chinese-ai-detector-bert")

# 下载数据集
from datasets import load_dataset
dataset = load_dataset("YOUR_USERNAME/chinese-ai-detection-dataset")
```

或使用CLI：
```bash
huggingface-cli download YOUR_USERNAME/chinese-ai-detector-bert
```

---

*最后更新: 2026-01-26*
