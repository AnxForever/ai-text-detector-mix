# AI文本检测 - 数据构建与训练计划

> 更新时间: 2026-01-24

---

## 一、当前数据集状态

### 中文数据集 (datasets/final_clean/)
| 文件 | 数量 | 来源 |
|------|------|------|
| all_ai.csv | 28,250 | GPT-4, Claude, DeepSeek, ChatGPT等 |
| all_human.csv | 27,188 | THUCNews + HC3人类回答 |
| train.csv | 44,350 | 80% |
| val.csv | 5,544 | 10% |
| test.csv | 5,544 | 10% |

### 英文数据集 (datasets/english/)
| 文件 | 数量 | 来源 |
|------|------|------|
| train.jsonl | 23,707 | GPT-3.5/4, LLaMA, Falcon等 |
| val.jsonl | 3,589 | 同上 |

---

## 二、数据构建方案

### 2.1 短期目标 (本周)

#### 数据增强
```python
# 1. 回译增强 (中→英→中)
# 目标: 每类增加5000条
python scripts/data_augmentation/back_translate.py \
  --input datasets/final_clean/train.csv \
  --output datasets/augmented/back_translated.csv

# 2. 同义词替换
# 目标: 每类增加3000条
python scripts/data_augmentation/synonym_replace.py \
  --input datasets/final_clean/train.csv \
  --output datasets/augmented/synonym_replaced.csv
```

#### 硬负样本构建
```python
# 3. AI文本人工润色 (标签仍为AI)
# 4. 人类文本AI改写 (标签仍为Human)
# 目标: 各2000条
```

### 2.2 中期目标 (2周内)

#### 多模型AI文本生成
使用API生成更多样化的AI文本:

| 模型 | 目标数量 | API |
|------|----------|-----|
| DeepSeek-V3 | 5,000 | sk-WPzv2... |
| GPT-4 | 3,000 | 待配置 |
| Claude-3 | 3,000 | 待配置 |
| Qwen-Max | 2,000 | 待配置 |

#### 生成策略
```python
# 使用详细提示词约束，模仿人类写作风格
prompts = [
    "以普通记者口吻写一篇400字新闻报道...",
    "以学生视角写一篇500字议论文...",
    "模仿知乎回答风格写一篇科普文章...",
]
```

### 2.3 长期目标 (1个月)

#### 跨领域数据收集
| 领域 | 目标 | 来源 |
|------|------|------|
| 学术论文 | 10,000 | arXiv中文摘要 |
| 社交媒体 | 10,000 | 微博/知乎 |
| 新闻报道 | 10,000 | 新闻网站 |
| 文学创作 | 5,000 | 小说/散文 |

#### 最终数据集目标
- AI文本: 50,000+
- Human文本: 50,000+
- 总计: 100,000+

---

## 三、模型训练计划

### 3.1 阶段一: BERT基线 (1天)

```bash
python scripts/training/train_bert_improved.py \
  --model-name bert-base-chinese \
  --train-csv datasets/final_clean/train.csv \
  --val-csv datasets/final_clean/val.csv \
  --test-csv datasets/final_clean/test.csv \
  --output-dir models/bert_v2 \
  --batch-size 8 --num-epochs 3 --learning-rate 2e-5
```

**预期**: 准确率 95-98%

### 3.2 阶段二: BERT-BiGRU (2天)

```
架构: BERT(768) → BiGRU(256*2) → Attention → FC(2)
```

**实现要点**:
- BERT提取语义特征
- BiGRU捕捉序列依赖
- 注意力机制聚合
- SHAP可解释性分析

**预期**: 准确率 96-99%

### 3.3 阶段三: DPCNN (2天)

```
架构: Embedding → Region Embedding → [Conv + Pooling]×N → FC
```

**优势**:
- 处理长文本效率高
- 参数量小，训练快
- 捕捉局部特征

**预期**: 准确率 94-97%

### 3.4 阶段四: 图增强模型 (3天)

```
架构: 文本 → BERT嵌入 → 构建文本图 → GCN/GAT → 分类
```

**图构建**:
- 节点: 句子/段落
- 边: 语义相似度 > 阈值
- 使用PyTorch Geometric

**预期**: 准确率 97-99%

### 3.5 阶段五: LoRA大模型 (4天)

```python
# 使用Qwen-7B + LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    task_type="SEQ_CLS"
)
```

**8GB GPU适配**:
- batch_size=1
- gradient_accumulation=8
- 4bit量化

**预期**: 准确率 98-99%+

---

## 四、评估方案

### 4.1 基础指标
- Accuracy, Precision, Recall, F1
- AUC-ROC, AUC-PR

### 4.2 鲁棒性测试
- 格式扰动 (去除markdown等)
- 长度分布测试 (短/中/长文本)
- 跨领域测试 (新闻/学术/社交)

### 4.3 对抗测试
- AI文本人工润色后检测
- 人类文本AI改写后检测
- 混合文本检测

---

## 五、时间线

| 周 | 任务 | 产出 |
|----|------|------|
| 第1周 | BERT基线 + BERT-BiGRU | 2个模型 |
| 第2周 | DPCNN + 图增强 | 2个模型 |
| 第3周 | LoRA大模型 + 数据增强 | 1个模型 + 增强数据 |
| 第4周 | 综合评估 + 论文撰写 | 实验报告 |

---

## 六、立即行动

### 今晚 (21:40-23:00)
1. ✅ 项目整理完成
2. [ ] 启动BERT基线训练 (新数据集55,438条)
3. [ ] 监控训练进度

### 明天
1. [ ] 评估BERT基线结果
2. [ ] 实现BERT-BiGRU模型
3. [ ] 开始数据增强脚本

---

**查看API配置**: `API_KEYS.md`
**查看项目结构**: `README.md`
