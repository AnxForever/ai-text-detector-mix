# 技术总结 - 用于文献检索

## 一、数据构建技术

### 1.1 数据来源与采集
**技术关键词**:
- Web scraping / 网络爬虫
- Public dataset collection / 公开数据集收集
- Chinese text corpus / 中文文本语料库

**具体方法**:
- 人类文本: 知乎、百度知道、新闻网站
- AI文本: 调用API生成（DeepSeek, Qwen, GLM等）
- 数据集: HC3-Chinese等公开数据集

**相关文献方向**:
- Chinese text dataset construction
- Web-based corpus collection
- Crowdsourcing for text annotation

---

### 1.2 混合文本生成（核心创新）
**技术关键词**:
- Hybrid text generation / 混合文本生成
- Human-AI collaborative writing / 人机协同写作
- Text continuation / 文本续写
- Text rewriting / 文本改写
- Text polishing / 文本润色

**具体方法**:
```
C2 (Continuation): 人类写开头 → AI续写
C3 (Rewriting): 人类文本 → AI改写
C4 (Polishing): 人类文本 → AI润色
```

**相关文献方向**:
- Mixed authorship detection
- Partial AI-generated text detection
- Human-AI co-creation
- Text manipulation detection

---

### 1.3 边界标记机制（核心创新）
**技术关键词**:
- Boundary marker / 边界标记
- Special token insertion / 特殊标记插入
- [SEP] token / 分隔符标记
- Explicit boundary annotation / 显式边界标注

**具体方法**:
```
原始: "人类部分AI部分"
标记: "人类部分[SEP]AI部分"
```

**相关文献方向**:
- Special tokens in transformers
- Boundary detection in text
- Segment-level annotation
- Token-based text segmentation

---

### 1.4 Token级标注
**技术关键词**:
- Token-level labeling / Token级标注
- Sequence labeling / 序列标注
- BIO tagging / BIO标注
- Span annotation / 片段标注

**具体方法**:
```json
{
  "text": "人类部分[SEP]AI部分",
  "token_labels": [0,0,0,...,1,1,1],
  "boundary": 20
}
```

**相关文献方向**:
- Token classification
- Named entity recognition (NER) techniques
- Sequence tagging for text analysis
- Fine-grained text annotation

---

### 1.5 数据平衡与增强
**技术关键词**:
- Data balancing / 数据平衡
- Class imbalance / 类别不平衡
- Data augmentation / 数据增强
- Stratified sampling / 分层采样

**具体方法**:
- 训练/验证/测试: 8:1:1分割
- Human/AI样本平衡
- 混合文本类别平衡

**相关文献方向**:
- Imbalanced dataset handling
- Stratified cross-validation
- Data augmentation for NLP

---

## 二、模型训练技术

### 2.1 预训练模型
**技术关键词**:
- Pre-trained language model / 预训练语言模型
- BERT (Bidirectional Encoder Representations from Transformers)
- Chinese BERT / 中文BERT
- RoBERTa (Robustly Optimized BERT)
- Whole Word Masking (WWM) / 全词掩码

**具体模型**:
- `chinese-roberta-wwm-ext`
- 基于HuggingFace Transformers库

**相关文献方向**:
- BERT for text classification
- Chinese language models
- Transfer learning in NLP
- Fine-tuning pre-trained models

---

### 2.2 双层检测架构（核心创新）
**技术关键词**:
- Two-stage detection / 两阶段检测
- Hierarchical classification / 层次化分类
- Coarse-to-fine approach / 粗到细方法
- Multi-task learning / 多任务学习

**具体架构**:
```
Layer 1: BertForSequenceClassification
  → Binary classification (Human vs AI)
  
Layer 2: BertForTokenClassification
  → Token-level labeling (boundary detection)
```

**相关文献方向**:
- Hierarchical text classification
- Two-stage detection systems
- Cascade classifiers
- Multi-level text analysis

---

### 2.3 序列分类任务
**技术关键词**:
- Sequence classification / 序列分类
- Binary classification / 二分类
- Text classification / 文本分类
- Sentiment analysis techniques / 情感分析技术

**具体方法**:
- 任务: Human (0) vs AI (1)
- 模型: BertForSequenceClassification
- 输出: Softmax概率分布

**相关文献方向**:
- BERT for sequence classification
- Binary text classification
- Authorship attribution
- AI-generated text detection

---

### 2.4 Token分类任务（边界检测）
**技术关键词**:
- Token classification / Token分类
- Sequence labeling / 序列标注
- Boundary detection / 边界检测
- Span detection / 片段检测

**具体方法**:
- 任务: 每个token标记为Human(0)或AI(1)
- 模型: BertForTokenClassification
- 输出: 每个token的类别概率

**相关文献方向**:
- Token-level classification with BERT
- Boundary detection in NLP
- Sequence tagging models
- Span-based text analysis

---

### 2.5 损失函数与优化
**技术关键词**:
- Cross-entropy loss / 交叉熵损失
- AdamW optimizer / AdamW优化器
- Learning rate scheduling / 学习率调度
- Gradient clipping / 梯度裁剪

**具体配置**:
```python
loss = CrossEntropyLoss()
optimizer = AdamW(lr=2e-5)
batch_size = 8
epochs = 3
max_length = 512
```

**相关文献方向**:
- Optimization for transformers
- AdamW optimizer
- Learning rate strategies
- Training stability techniques

---

### 2.6 Fine-tuning策略
**技术关键词**:
- Fine-tuning / 微调
- Transfer learning / 迁移学习
- Domain adaptation / 领域自适应
- Few-shot learning / 少样本学习

**具体方法**:
- 冻结底层，微调顶层
- 全模型微调
- 学习率: 2e-5 (较小)

**相关文献方向**:
- Fine-tuning strategies for BERT
- Transfer learning in NLP
- Domain-specific model adaptation
- Efficient fine-tuning methods

---

### 2.7 评估指标
**技术关键词**:
- Accuracy / 准确率
- Precision / 精确率
- Recall / 召回率
- F1-score / F1分数
- Confusion matrix / 混淆矩阵
- ROC curve / ROC曲线

**具体指标**:
- 整体准确率: 98.71%
- Per-class precision/recall
- Token-level accuracy: 96.69%
- Boundary accuracy: 49.51%

**相关文献方向**:
- Evaluation metrics for text classification
- Performance measurement in NLP
- Imbalanced classification metrics
- Multi-metric evaluation

---

## 三、关键技术创新点

### 3.1 边界标记机制
**创新点**: 使用[SEP]标记显式标注人类/AI边界
**效果**: C2检测率从79.82%提升到93.84% (+14%)
**文献方向**:
- Special token usage in transformers
- Explicit vs implicit boundary detection
- Marker-based text segmentation

---

### 3.2 双层检测架构
**创新点**: 分类+边界定位的两阶段方法
**效果**: 实现粗粒度分类和细粒度定位
**文献方向**:
- Hierarchical detection systems
- Coarse-to-fine text analysis
- Multi-stage classification

---

### 3.3 Token级边界定位
**创新点**: 精确到Token级别的边界检测
**效果**: 实际误差<10字符
**文献方向**:
- Fine-grained text segmentation
- Token-level boundary detection
- Span-based text analysis

---

## 四、相关研究领域

### 4.1 AI文本检测
**关键词**:
- AI-generated text detection
- Machine-generated text identification
- Synthetic text detection
- GPT detection
- LLM-generated content detection

**代表性工作**:
- DetectGPT
- GPTZero
- AI text classifiers

---

### 4.2 作者归属识别
**关键词**:
- Authorship attribution
- Author identification
- Stylometry / 文体学
- Writing style analysis

**相关技术**:
- N-gram analysis
- Linguistic features
- Deep learning for authorship

---

### 4.3 文本真实性检测
**关键词**:
- Text authenticity detection
- Fake text detection
- Deepfake text detection
- Content verification

---

### 4.4 混合作者检测
**关键词**:
- Mixed authorship detection
- Collaborative writing analysis
- Multi-author text segmentation
- Intrinsic plagiarism detection

**相关工作**:
- PAN competition (plagiarism detection)
- Multi-author document analysis

---

### 4.5 中文NLP
**关键词**:
- Chinese natural language processing
- Chinese text classification
- Chinese language models
- Simplified Chinese / Traditional Chinese

**代表性模型**:
- BERT-Chinese
- RoBERTa-wwm-ext-Chinese
- ERNIE (Baidu)
- GLM (Tsinghua)

---

## 五、文献检索建议

### 5.1 核心检索词组合

**组合1: AI文本检测**
```
"AI-generated text detection" AND "Chinese"
"machine-generated text" AND "classification"
"GPT detection" AND "BERT"
```

**组合2: 混合文本检测**
```
"mixed authorship detection"
"human-AI collaborative writing" AND "detection"
"partial AI-generated text"
"hybrid text" AND "boundary detection"
```

**组合3: 边界检测**
```
"boundary detection" AND "text segmentation"
"token-level classification" AND "BERT"
"span detection" AND "NLP"
"[SEP] token" AND "transformers"
```

**组合4: 中文NLP**
```
"Chinese text classification" AND "BERT"
"Chinese language model" AND "fine-tuning"
"RoBERTa" AND "Chinese"
```

---

### 5.2 推荐数据库

**学术数据库**:
- Google Scholar
- IEEE Xplore
- ACL Anthology
- arXiv (cs.CL)
- Semantic Scholar

**中文数据库**:
- 中国知网 (CNKI)
- 万方数据
- 维普网

---

### 5.3 推荐会议/期刊

**顶级会议**:
- ACL (Association for Computational Linguistics)
- EMNLP (Empirical Methods in NLP)
- NAACL (North American Chapter of ACL)
- COLING (International Conference on Computational Linguistics)

**相关会议**:
- AAAI (Artificial Intelligence)
- IJCAI (International Joint Conference on AI)
- WWW (World Wide Web Conference)
- SIGIR (Information Retrieval)

**中文会议**:
- CCL (Chinese Computational Linguistics)
- NLPCC (Natural Language Processing and Chinese Computing)

**期刊**:
- Computational Linguistics
- TACL (Transactions of ACL)
- Natural Language Engineering
- 中文信息学报

---

## 六、技术栈总结

### 6.1 编程语言与框架
- Python 3.12
- PyTorch
- HuggingFace Transformers
- scikit-learn
- pandas, numpy

### 6.2 模型与工具
- chinese-roberta-wwm-ext
- BertTokenizer
- BertForSequenceClassification
- BertForTokenClassification

### 6.3 数据处理
- CSV/JSON格式
- Tokenization
- Padding & Truncation
- Train/Val/Test split

### 6.4 训练环境
- CUDA GPU
- Batch size: 8
- Mixed precision training (可选)
- Gradient accumulation (可选)

---

## 七、论文写作建议

### 7.1 相关工作章节
**需要引用的方向**:
1. AI文本检测的现有方法
2. BERT及其变体在文本分类中的应用
3. 混合作者检测的相关工作
4. 中文NLP的预训练模型
5. Token级序列标注方法

### 7.2 方法章节
**需要引用的技术**:
1. BERT原理和架构
2. Fine-tuning策略
3. [SEP] token的作用
4. Token classification方法
5. 两阶段检测系统

### 7.3 实验章节
**需要引用的评估方法**:
1. 标准评估指标 (Accuracy, Precision, Recall, F1)
2. 消融实验设计
3. 统计显著性检验
4. 混淆矩阵分析

---

## 八、关键文献类型

### 8.1 必读基础文献
- BERT原论文 (Devlin et al., 2019)
- RoBERTa论文 (Liu et al., 2019)
- Chinese BERT相关论文

### 8.2 AI检测相关
- DetectGPT及其变体
- GPT生成文本检测方法
- 最新的LLM检测工作

### 8.3 边界检测相关
- 文本分割方法
- 序列标注技术
- Span detection方法

### 8.4 混合文本相关
- 协同写作分析
- 多作者文档分析
- 内在抄袭检测 (intrinsic plagiarism)

---

## 九、检索策略

### 9.1 时间范围
- 重点: 2018-2024 (BERT之后)
- 补充: 2015-2017 (深度学习基础)
- 最新: 2023-2024 (LLM检测)

### 9.2 引用数量
- 高引用论文 (>100): 基础方法
- 中等引用 (10-100): 相关技术
- 最新论文 (<10): 前沿工作

### 9.3 检索顺序
1. 综述论文 (Survey/Review)
2. 经典方法论文
3. 最新进展论文
4. 相关应用论文

---

*最后更新: 2026-01-26*
*用途: 论文文献检索*
