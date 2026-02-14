# 自构建AI生成数据集 - 详细说明

> 本文档专门介绍项目中**自己生成**的AI文本数据
> 更新日期: 2026-01-29

---

## 一、数据总览

| 类别 | 样本数 | 说明 |
|-----|-------|------|
| **已用于训练** (generated_v2) | 11,116 | 已整合到平衡训练集 |
| **scenario_fill 原始生成** | ~81,000 | 多批次生成的原始数据 |
| **混合文本 (C2/C3/C4)** | 7,563 | 人机混合写作场景 |

---

## 二、已用于训练的数据

### 2.1 在训练集中的位置

```
datasets/bert_v2_overnight/train_balanced.csv
└── source='generated_v2' → 11,116 条 AI 文本
```

### 2.2 统计信息

| 指标 | 数值 |
|-----|------|
| 样本数 | 11,116 条 |
| 标签 | 全部为 AI (label=1) |
| 平均长度 | 728 字符 |

**长度分布:**

| 区间 | 样本数 |
|-----|-------|
| <100字符 | 4 |
| 100-200 | 427 |
| 200-300 | 2,097 |
| 300-500 | 2,575 |
| 500-1000 | 3,171 |
| 1000-2000 | 2,481 |
| >2000 | 361 |

---

## 三、Scenario Fill 生成数据

### 3.1 目录结构

```
datasets/generated/scenario_fill/
├── 2026-01-27_10h_multi_proxies/   # 主要批次 ⭐
│   ├── ai_scenario_fill_*_part*.jsonl    # 原始生成
│   ├── deduplicated_all.jsonl            # 去重后 (4,856条)
│   └── by_scenario/                       # 按场景分类
│       ├── A_education/  (632条)
│       ├── B_workplace/
│       ├── C_knowledge/  (563条)
│       ├── D_community/  (96条)
│       ├── E_commerce/
│       └── F_news/
├── cleaned/                          # 清洗后数据 (12,126条)
├── smoke/                            # 冒烟测试
├── smoke_glm/                        # GLM模型测试
└── smoke_qwen/                       # Qwen模型测试
```

### 3.2 生成配置

使用脚本: `scripts/generation/scenario_fill_generate.py`

场景模板位于: `docs/plans/ai_generation_templates_v2_mapped_*.yaml`

**生成场景分类:**
- **A_education**: 教育学习场景
- **B_workplace**: 职场办公场景
- **C_knowledge**: 知识问答场景
- **D_community**: 社区交流场景
- **E_commerce**: 电商购物场景
- **F_news**: 新闻媒体场景

### 3.3 数据格式 (JSONL)

```json
{
  "text": "生成的AI文本内容...",
  "label": 1,
  "source": "generated_v2",
  "model": "deepseek-v3",
  "scenario": "A_education",
  "sub_scenario": "论文写作",
  "timestamp": "2026-01-28T12:30:00"
}
```

### 3.4 使用的AI模型

| 模型 | API | 说明 |
|-----|-----|------|
| DeepSeek-V3 | OpenRouter | 主要生成模型 |
| GLM-4 | 智谱AI | 备选模型 |
| Qwen-2.5 | 阿里云 | 备选模型 |

---

## 四、混合文本数据 (C2/C3/C4)

### 4.1 位置

```
datasets/mixed/hybrid/
├── c2_*.json          # C2: 人类开头+AI续写
├── c3_*.json          # C3: AI改写人类文本
├── c4_*.json          # C4: AI润色人类文本
├── merged_all.json    # 合并数据 (5,063条)
├── hybrid_dataset.csv # CSV格式 (5,063条)
└── hybrid_dataset_with_sep.csv  # 带[SEP]标记 (7,563条)
```

### 4.2 类别说明

| 类别 | 样本数 | 描述 |
|-----|-------|------|
| **C2 (续写)** | 4,068 | 人类写开头，AI续写后面 |
| **C3 (改写)** | 1,594 | AI改写人类的原始文本 |
| **C4 (润色)** | 2,435 | AI对人类文本进行润色 |

### 4.3 C2 详细文件

| 文件 | 样本数 | 说明 |
|-----|-------|------|
| c2_batch.json | 1,000 | 批量生成 |
| c2_continuation.json | 225 | 续写数据 |
| c2_fast.json | 200 | 快速生成 |
| c2_final.json | 100 | 最终版本 |
| c2_local.json | 100 | 本地模型生成 |
| c2_local_v2.json | 309 | 本地模型v2 |
| c2_local_v3.json | 100 | 本地模型v3 |
| c2_span_labels.json | 2,034 | 带边界标注 |

### 4.4 C3 详细文件

| 文件 | 样本数 | 说明 |
|-----|-------|------|
| c3_batch.json | 1,000 | 批量生成 |
| c3_edited.json | 94 | 编辑版 |
| c3_edited_kfc.json | 50 | KFC风格 |
| c3_final.json | 200 | 最终版本 |
| c3_local_v2.json | 200 | 本地模型 |
| c3_public.json | 50 | 公开数据 |

### 4.5 C4 详细文件

| 文件 | 样本数 | 说明 |
|-----|-------|------|
| c4_batch.json | 500 | 批量生成 |
| c4_fast.json | 300 | 快速生成 |
| c4_final.json | 305 | 最终版本 |
| c4_local.json | 100 | 本地模型 |
| c4_local_v2.json | 400 | 本地模型v2 |
| c4_local_v3.json | 444 | 本地模型v3 |
| c4_polished.json | 122 | 润色版 |
| c4_polished_kfc.json | 100 | KFC风格润色 |
| c4_polished_x666.json | 64 | x666风格 |
| c4_public.json | 100 | 公开数据 |

---

## 五、数据加载示例

### 5.1 加载已训练数据

```python
import pandas as pd

# 从训练集中筛选自生成数据
train = pd.read_csv('datasets/bert_v2_overnight/train_balanced.csv', encoding='utf-8-sig')
my_generated = train[train['source'] == 'generated_v2']
print(f"自生成AI数据: {len(my_generated)} 条")
```

### 5.2 加载原始生成数据

```python
import json

# 加载去重后的数据
with open('datasets/generated/scenario_fill/2026-01-27_10h_multi_proxies/deduplicated_all.jsonl', 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]
print(f"去重后数据: {len(data)} 条")
```

### 5.3 加载混合文本

```python
import pandas as pd

# 带[SEP]标记的混合文本
hybrid = pd.read_csv('datasets/mixed/hybrid/hybrid_dataset_with_sep.csv', encoding='utf-8-sig')
print(f"混合文本: {len(hybrid)} 条")
```

---

## 六、生成流程说明

### 6.1 生成脚本

主脚本: `scripts/generation/scenario_fill_generate.py`

```bash
# 运行生成
python scripts/generation/scenario_fill_generate.py \
    --config configs/scenario_fill_v5_BEF_priority_2026-01-28.json \
    --output datasets/generated/scenario_fill/new_batch/
```

### 6.2 配置文件

位于 `configs/` 目录:
- `scenario_fill_v5_BEF_priority_2026-01-28.json`
- `scenario_fill_v6_ACD_priority_2026-01-28.json`
- `scenario_fill_v7_type_balance_2026-01-28.json`
- `scenario_fill_v8_thesis_demo_2026-01-28.json`

### 6.3 清洗流程

1. **原始生成** → JSONL 文件
2. **质量筛选** → 移除低质量/重复内容
3. **去重** → `deduplicated_all.jsonl`
4. **格式转换** → CSV / 统一schema

---

## 七、文件大小参考

```
datasets/generated/scenario_fill/2026-01-27_10h_multi_proxies/
├── deduplicated_all.jsonl                    ~2MB
├── ai_scenario_fill_*_combined.jsonl         ~500KB
└── cleaned/                                  ~50MB (总计)

datasets/mixed/hybrid/
├── merged_all.json                           ~5MB
├── hybrid_dataset_with_sep.csv               ~8MB
└── c2_*/c3_*/c4_*.json                       ~20MB (总计)
```

---

## 八、注意事项

1. **数据用途**:
   - `generated_v2` 已整合到训练集，直接用 `train_balanced.csv` 即可
   - 原始生成数据可用于分析或扩展

2. **数据质量**:
   - `cleaned/` 目录的数据经过清洗
   - `*_rejected.jsonl` 是被拒绝的低质量数据

3. **混合文本**:
   - C2/C3/C4 主要用于边界检测任务
   - `hybrid_dataset_with_sep.csv` 包含 `[SEP]` 边界标记

---

*文档生成时间: 2026-01-29*
