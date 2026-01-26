# 中文AI文本检测系统 - 数据集与模型

## 📊 数据集

### 1. Combined v2 (主训练数据集)
- **规模**: 66,001条样本
- **分布**: 
  - 训练集: 52,800 (80%)
  - 验证集: 6,600 (10%)
  - 测试集: 6,601 (10%)
- **类别**: Human (32,869) / AI (33,132)
- **格式**: CSV (text, label)
- **大小**: ~106MB

### 2. 混合文本数据集 (Hybrid Dataset)
- **规模**: 7,563条样本
- **类别分布**:
  - C2 (续写): 2,034条 - 人类开头+AI续写
  - C3 (改写): 1,594条 - AI改写人类文本
  - C4 (润色): 2,435条 - AI润色人类文本
  - Human: 1,500条 - 纯人类文本
- **特殊标记**: C2样本包含`[SEP]`边界标记
- **格式**: CSV (text, label, category, boundary)
- **大小**: ~86MB

### 3. Span标注数据集
- **规模**: 2,034条 (C2类别)
- **标注**: Token级别 (0=Human, 1=AI)
- **用途**: 训练边界检测器
- **格式**: JSON
- **大小**: ~27MB

### 4. 基础数据集 (Final Clean)
- **规模**: 55,438条样本
- **来源**: 
  - 人类文本: 知乎、百度知道、新闻
  - AI文本: DeepSeek, Qwen, GLM等
- **大小**: ~264MB

**总计**: ~483MB

---

## 🤖 模型

### 1. BERT分类器 (bert_v2_with_sep)
- **基础模型**: chinese-roberta-wwm-ext
- **任务**: 二分类 (Human vs AI)
- **性能**:
  - 整体准确率: **98.71%**
  - Human: Precision=98.98%, Recall=98.33%
  - AI: Precision=98.47%, Recall=99.07%
  - C2检测: **93.84%** (提升14%)
- **创新**: 使用`[SEP]`标记标注混合文本边界
- **大小**: ~390MB
- **文件**: 
  - `model.safetensors` - 模型权重
  - `config.json` - 模型配置
  - `vocab.txt` - 词表
  - `tokenizer_config.json` - 分词器配置

### 2. Span边界检测器 (bert_span_detector)
- **基础模型**: bert_combined (fine-tuned)
- **任务**: Token级分类 (边界检测)
- **性能**:
  - Token分类准确率: **96.69%**
  - 边界定位准确率: **49.51%** (±5 tokens)
  - 实际误差: **<10字符**
- **大小**: ~389MB
- **文件**: 同上

**总计**: ~779MB

---

## 📥 下载方式

### 方案1: 百度网盘 (推荐)
```
链接: https://pan.baidu.com/s/[待上传后填写]
提取码: [待填写]
```

**包含内容**:
- `models.zip` (779MB) - 两个训练好的模型
- `datasets.zip` (483MB) - 完整数据集

### 方案2: 阿里云盘
```
链接: https://www.alipan.com/s/[待上传后填写]
```

### 方案3: Google Drive
```
链接: https://drive.google.com/[待上传后填写]
```

### 方案4: Hugging Face Hub (待上传)
```bash
# 下载模型
huggingface-cli download AnxForever/chinese-ai-detector-bert
huggingface-cli download AnxForever/chinese-ai-detector-span

# 下载数据集
huggingface-cli download AnxForever/chinese-ai-detection-dataset
```

**注**: 由于网络限制，Hugging Face上传暂时无法完成。推荐使用百度网盘下载。

---

## 📂 文件结构

下载后解压到项目根目录：

```
datacollection/
├── models/
│   ├── bert_v2_with_sep/
│   │   ├── model.safetensors
│   │   ├── config.json
│   │   ├── vocab.txt
│   │   └── tokenizer_config.json
│   └── bert_span_detector/
│       └── (同上)
│
└── datasets/
    ├── combined_v2/
    │   ├── train.csv
    │   ├── val.csv
    │   ├── test.csv
    │   └── test_hybrid_only.csv
    ├── hybrid/
    │   ├── hybrid_dataset_with_sep.csv
    │   └── c2_span_labels.json
    └── final_clean/
        ├── train.csv
        ├── val.csv
        └── test.csv
```

---

## 🔧 使用方法

### 1. 下载数据和模型
```bash
# 从百度网盘/Google Drive下载
# 解压到项目根目录
```

### 2. 运行演示
```bash
cd /mnt/c/datacollection
source .venv/bin/activate
export HF_HUB_OFFLINE=1
python scripts/demo/visualize_detection.py
```

### 3. 评估模型
```bash
python scripts/evaluation/eval_complete.py
```

### 4. 重新训练
```bash
# 训练分类器
python scripts/training/train_v2_simple.py

# 训练边界检测器
python scripts/training/train_span_detector.py
```

---

## 📊 数据集统计

### Combined v2
| 分割 | Human | AI | 总计 |
|------|-------|-----|------|
| Train | 26,400 | 26,400 | 52,800 |
| Val | 3,300 | 3,300 | 6,600 |
| Test | 3,169 | 3,432 | 6,601 |
| **总计** | **32,869** | **33,132** | **66,001** |

### 混合数据
| 类别 | 样本数 | 说明 |
|------|--------|------|
| C2 | 2,034 | 人类开头+AI续写 (含[SEP]) |
| C3 | 1,594 | AI改写人类文本 |
| C4 | 2,435 | AI润色人类文本 |
| Human | 1,500 | 纯人类文本 |
| **总计** | **7,563** | |

---

## 📄 数据格式

### CSV格式 (combined_v2)
```csv
text,label
"这是一段人类写的文本...",0
"这是一段AI生成的文本...",1
```

### CSV格式 (hybrid)
```csv
text,label,category,boundary
"人类部分[SEP]AI部分",1,"C2",20
```

### JSON格式 (span labels)
```json
{
  "text": "人类部分[SEP]AI部分",
  "boundary": 20,
  "token_labels": [0,0,0,...,1,1,1],
  "category": "C2",
  "label": 1
}
```

---

## 🔬 数据来源

### 人类文本
- 知乎问答
- 百度知道
- 新闻文章
- HC3-Chinese数据集

### AI文本
- DeepSeek API
- Qwen API
- GLM API
- 公开数据集

---

## 📜 引用

如使用本数据集或模型，请引用：

```bibtex
@misc{chinese-ai-detector-2026,
  title={Chinese AI-Generated Text Detection with Boundary Markers},
  author={AnxForever},
  year={2026},
  url={https://github.com/AnxForever/ai-text-detector-mix}
}
```

---

## ⚖️ 许可证

- **数据集**: CC BY-NC-SA 4.0 (仅供学术研究使用)
- **模型**: MIT License
- **代码**: MIT License

---

## 📧 联系方式

如有问题或需要数据集/模型，请：
- 提交Issue: https://github.com/AnxForever/ai-text-detector-mix/issues
- 发送邮件: [待添加]

---

*最后更新: 2026-01-26*
