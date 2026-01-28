[根目录](../CLAUDE.md) > **models**

# 模型模块

## 模块职责

存储训练好的 BERT 模型文件，包括分类器和边界检测器。

**重要**: 此目录为只读，不要删除模型文件!

## 模型列表

| 模型目录 | 用途 | 准确率 | 大小 |
|---------|------|-------|------|
| `bert_v2_with_sep/` | 主分类器 | 98.71% | ~391MB |
| `bert_span_detector/` | 边界检测器 | 96.69% (Token) | ~391MB |
| `bert_improved/best_model/` | 改进版分类器 | 100%* | ~391MB |
| `bert_improved/final_model/` | 最终训练模型 | - | ~391MB |

*在特定测试集上

## 模型结构

### bert_v2_with_sep (主分类器)

```
bert_v2_with_sep/
├── config.json              # 模型配置
├── model.safetensors        # 模型权重
├── tokenizer_config.json    # 分词器配置
├── special_tokens_map.json  # 特殊token映射
└── vocab.txt                # 词表
```

### bert_span_detector (边界检测器)

```
bert_span_detector/
├── config.json
├── model.safetensors
├── tokenizer_config.json
├── special_tokens_map.json
└── vocab.txt
```

## 模型加载

### Python 加载方式

```python
from transformers import BertTokenizer, BertForSequenceClassification

# 加载分类器
tokenizer = BertTokenizer.from_pretrained('models/bert_v2_with_sep')
model = BertForSequenceClassification.from_pretrained('models/bert_v2_with_sep')

# 加载边界检测器
from transformers import BertForTokenClassification
span_tokenizer = BertTokenizer.from_pretrained('models/bert_span_detector')
span_model = BertForTokenClassification.from_pretrained('models/bert_span_detector')
```

### 推理示例

```python
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

text = "待检测的文本"
inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    label = 'AI' if pred == 1 else 'Human'
```

## 数据模型

### 分类器输出

- **num_labels**: 2 (Human=0, AI=1)
- **输入格式**: `[CLS] text [SEP]`
- **最大长度**: 512 tokens

### 边界检测器输出

- **num_labels**: 2 (Human=0, AI=1)
- **输出**: 每个token的标签序列
- **边界定位**: 找到 0→1 的转换点

## 保护措施

1. 模型文件设置为只读 (`chmod 444`)
2. Git LFS 管理大文件
3. `.gitignore` 中排除敏感权重

## 常见问题 (FAQ)

**Q: 模型加载报错 "FileNotFoundError"?**
A: 检查模型目录是否完整，可能需要从备份恢复。

**Q: 模型太大无法加载?**
A: 使用 `low_cpu_mem_usage=True` 参数或减小批次大小。

**Q: 如何更新模型?**
A: 运行训练脚本后，新模型会保存到对应目录。

## 相关文件清单

```
models/
├── README_IMPORTANT.md         # 重要说明 (不要删除!)
├── bert_v2_with_sep/           # 主分类器
│   ├── config.json
│   ├── model.safetensors
│   └── vocab.txt
├── bert_span_detector/         # 边界检测器
│   ├── config.json
│   ├── model.safetensors
│   └── vocab.txt
└── bert_improved/              # 改进版模型
    ├── best_model/
    ├── final_model/
    └── test_results.json
```

## 变更记录 (Changelog)

### 2026-01-28
- 初始化模块文档

---

*文档生成时间: 2026-01-28T12:42:53+0800*
