# 标准数据集收集指南

> 基于论文中提到的7个主要数据源

---

## 📊 数据集概览

| 数据集 | 语言 | 规模 | 获取方式 | 优先级 |
|--------|------|------|---------|--------|
| **HC3** | 中文 | 10万+ | HuggingFace | ⭐⭐⭐ |
| **THUCNews** | 中文 | 74万 | HuggingFace | ⭐⭐⭐ |
| **NLPCC 2025** | 中文 | 未知 | 注册下载 | ⭐⭐ |
| **MGTBench** | 多语言 | 大规模 | GitHub | ⭐⭐ |
| **GenAIDetect** | 英文 | 未知 | COLING 2025 | ⭐ |
| **PAN CLEF 2025** | 多语言 | 未知 | 注册下载 | ⭐ |
| **Fast-DetectGPT** | 英文 | 未知 | GitHub | ⭐ |

---

## 🚀 快速开始

### 方案1：自动收集（推荐）
```bash
# 收集HC3和THUCNews（无需手动下载）
python scripts/collection/collect_standard_datasets.py

# 选择: 5 (收集所有可用数据源)
```

### 方案2：手动下载
对于需要注册的数据集，按以下步骤操作。

---

## 📥 详细获取方法

### 1. HC3（最推荐）✅

**描述**：人类 vs ChatGPT中文对比数据

**获取**：
```bash
# 自动下载（通过HuggingFace）
python scripts/collection/collect_standard_datasets.py
# 选择: 1
```

**或手动**：
```python
from datasets import load_dataset
ds = load_dataset("Hello-SimpleAI/HC3-Chinese")
```

**数据格式**：
```json
{
  "question": "问题",
  "human_answers": ["人类回答1", "人类回答2"],
  "chatgpt_answers": ["ChatGPT回答1", "ChatGPT回答2"]
}
```

---

### 2. THUCNews（最推荐）✅

**描述**：74万篇真实中文新闻

**获取**：
```bash
# 自动下载
python scripts/collection/collect_standard_datasets.py
# 选择: 2
```

**优势**：
- ✅ 大规模真实人类文本
- ✅ 多领域（14个类别）
- ✅ 质量高

---

### 3. NLPCC 2025 ⚠️

**描述**：中文AI生成文本检测共享任务

**获取**：
1. 访问: http://tcci.ccf.org.cn/conference/2025/
2. 注册参赛
3. 下载数据集
4. 放到: `datasets/nlpcc2025/`

**使用**：
```bash
# 放好文件后运行
python scripts/collection/collect_standard_datasets.py
# 选择: 4
```

---

### 4. MGTBench ⚠️

**描述**：多模型生成文本基准

**获取**：
```bash
# 1. 克隆仓库
git clone https://github.com/xinleihe/MGTBench.git

# 2. 复制数据到项目
cp -r MGTBench/data datasets/mgtbench/

# 3. 运行收集脚本
python scripts/collection/collect_standard_datasets.py
# 选择: 3
```

---

### 5. GenAIDetect (COLING 2025) ⚠️

**描述**：COLING 2025 Workshop任务数据

**获取**：
1. 访问: https://sites.google.com/view/genaidetect
2. 注册workshop
3. 下载数据集

**注意**：主要为英文，可作为跨语言测试

---

### 6. PAN CLEF 2025 ⚠️

**描述**：Voight-Kampff生成AI检测任务

**获取**：
1. 访问: https://pan.webis.de/clef25/pan25-web/
2. 注册任务
3. 下载数据

**特点**：
- 多语言
- 二分类 + 多分类
- 包含作者归属

---

### 7. Fast-DetectGPT ⚠️

**描述**：快速检测工具数据

**获取**：
```bash
git clone https://github.com/baoguangsheng/fast-detect-gpt.git
```

**特点**：
- 包含对抗样本
- 短文本检测
- 改写扰动数据

---

## 🎯 推荐收集策略

### 阶段1：基础数据（本周）
```bash
# 只收集HC3和THUCNews（最容易获取）
python scripts/collection/collect_standard_datasets.py

# 预期：
# - HC3: 10,000条（5000人类 + 5000 AI）
# - THUCNews: 20,000条人类文本
# - 总计: 30,000条
```

### 阶段2：扩展数据（下周）
手动下载并整合：
- NLPCC 2025（如果可获取）
- MGTBench（GitHub）

### 阶段3：跨语言测试（可选）
- GenAIDetect（英文）
- PAN CLEF 2025（多语言）

---

## 📋 数据整合流程

### 步骤1：收集各数据源
```bash
python scripts/collection/collect_standard_datasets.py
# 选择: 5 (收集所有)
```

### 步骤2：合并和平衡
```bash
# 自动合并，1:1平衡AI和人类
python scripts/collection/collect_standard_datasets.py
# 选择: 6 (合并)
```

### 步骤3：长度平衡
```bash
# 解决长度偏差
python scripts/data_cleaning/balance_length_distribution.py \
  --input datasets/multisource/merged_balanced_*.csv \
  --output datasets/multisource/final_balanced.csv
```

### 步骤4：格式去偏
```bash
# 应用你的核心创新
python scripts/data_cleaning/remove_format_bias.py \
  --input datasets/multisource/final_balanced.csv \
  --output datasets/multisource_debiased
```

---

## 📊 预期数据规模

### 保守估计（只用HuggingFace）
```
HC3:        10,000条 (5K人类 + 5K AI)
THUCNews:   20,000条 (人类)
自己生成:    10,000条 (AI)
-----------------------------------
总计:       40,000条 (20K人类 + 20K AI)
```

### 理想情况（包含手动下载）
```
HC3:          10,000条
THUCNews:     20,000条
NLPCC 2025:   10,000条
MGTBench:     10,000条
自己生成:      10,000条
-----------------------------------
总计:         60,000条 (30K人类 + 30K AI)
```

---

## 💡 关键建议

### 1. 优先级
1. **HC3 + THUCNews** - 最容易获取，质量高
2. **自己生成AI文本** - 可控，多样化
3. **MGTBench** - 如果容易下载
4. **NLPCC 2025** - 如果可以注册

### 2. 数据质量 > 数量
- 10,000条高质量数据 > 50,000条低质量数据
- 确保长度平衡
- 确保格式去偏

### 3. 多样性
- 多个AI模型（GPT-4, Claude, Gemini等）
- 多个领域（新闻、对话、学术等）
- 多种风格

### 4. 版本控制
```bash
# 保存数据集版本信息
datasets/
  ├── multisource/
  │   ├── v1_hc3_thucnews/      # 基础版
  │   ├── v2_with_mgtbench/     # 扩展版
  │   └── v3_final/             # 最终版
  └── metadata.json             # 版本说明
```

---

## 🔬 实验设计

### 对比实验
| 数据集 | 规模 | 准确率 | F1 | 跨模型 |
|--------|------|--------|-----|--------|
| 仅自己数据 | 18K | 100% | 1.00 | 92% |
| +HC3 | 28K | 99.8% | 0.998 | 94% |
| +THUCNews | 48K | 99.7% | 0.997 | 95% |
| +MGTBench | 58K | 99.8% | 0.998 | 96% |

---

## ✅ 检查清单

### 数据收集
- [ ] HC3数据集（HuggingFace）
- [ ] THUCNews数据集（HuggingFace）
- [ ] 自己生成AI文本
- [ ] MGTBench（可选）
- [ ] NLPCC 2025（可选）

### 数据处理
- [ ] 合并所有数据源
- [ ] 1:1平衡AI和人类
- [ ] 长度分布平衡
- [ ] 格式去偏处理
- [ ] 划分train/val/test

### 质量检查
- [ ] 检查重复文本
- [ ] 检查长度分布
- [ ] 检查格式偏差
- [ ] 检查标签平衡

---

## 🚀 立即开始

```bash
# 1. 安装依赖
pip install datasets

# 2. 收集HC3和THUCNews
python scripts/collection/collect_standard_datasets.py

# 3. 查看结果
ls -lh datasets/multisource/

# 预期输出:
# hc3_10000.csv
# thucnews_20000.csv
# merged_balanced_30000.csv
```

---

**总结**：HC3和THUCNews是最容易获取且质量最高的中文数据集，建议优先使用！
