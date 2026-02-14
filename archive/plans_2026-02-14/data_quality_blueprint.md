# 数据集质量保障蓝图

## 📊 当前状况分析

### 正在生成的数据 (v4批次)

| 指标 | 当前值 | 目标值 | 状态 |
|-----|--------|-------|------|
| 总数 | 5,163 | 60,000 | ⏳ 生成中 |
| 场景覆盖 | 5/6 (缺F) | 6/6 | ❌ 需修复 |
| 唯一主题 | 65 | 200+ | ⚠️ 严重不足 |
| 主题复用率 | 79x | <20x | ⚠️ 需降低 |

### 场景分布问题

```
目标: A(12K) B(10K) C(12K) D(10K) E(10K) F(6K)
当前: A(31%)  B(11%) C(26%) D(27%) E(5%)  F(0%)
问题: ❌F完全缺失  ⚠️E严重不足  ⚠️B偏少
```

### 主题多样性问题

```
当前: 65个唯一主题，平均每主题79条
问题: 主题高度重复，模型可能学到主题而非AI特征
```

---

## 🎯 质量目标

### 1. 数据多样性

| 维度 | 最低要求 | 理想目标 |
|-----|---------|---------|
| 场景覆盖 | 6/6 (100%) | 6/6 (100%) |
| 唯一主题 | 150+ | 300+ |
| 主题复用率 | <30x | <15x |
| 模型覆盖 | 5+ | 10+ |
| 长度分布 | 4个桶均有 | 均匀分布 |

### 2. 数据质量

| 维度 | 标准 |
|-----|------|
| 提示词残留 | 0% |
| 重复率 | <5% |
| 长度合规率 | >90% |
| 语法错误率 | <1% |

### 3. 数据平衡

| 类别 | 目标比例 |
|-----|---------|
| Human : AI | 1:1 (50%:50%) |
| 各场景 | 按计划比例 |
| 各长度桶 | 尽量均匀 |

---

## 🔧 改进方案

### 方案1: 扩展主题库 (优先级: 🔴高)

**问题**: 65个主题 → 需要300+

**解决方案**:

```python
# 每个场景25个主题 × 6场景 = 150个主题 (最低)
# 理想: 每个场景50个主题 × 6场景 = 300个主题

# 主题生成策略:
# 1. 按场景分类
# 2. 按子领域细分
# 3. 避免重复和近义
```

**具体主题扩展 (待实施)**:

| 场景 | 当前主题数 | 目标 | 需新增 |
|-----|----------|------|-------|
| A 教育 | ~15 | 50 | 35 |
| B 职场 | ~10 | 50 | 40 |
| C 知识 | ~15 | 50 | 35 |
| D 社区 | ~15 | 50 | 35 |
| E 商业 | ~5 | 50 | 45 |
| F 新闻 | ~5 | 50 | 45 |

### 方案2: 修复F场景缺失 (优先级: 🔴高)

**问题**: F(新闻/媒体)场景0条数据

**检查点**:
1. 配置文件是否包含F场景任务
2. 生成脚本是否支持F场景模板
3. 任务队列顺序是否合理

### 方案3: 平衡场景分布 (优先级: 🟡中)

**当前vs目标**:

```
场景  当前%   目标%   差距
A     30.6%   20%    +10.6% (过多)
B     11.4%   16.7%  -5.3%  (不足)
C     26.0%   20%    +6%    (过多)
D     26.7%   16.7%  +10%   (过多)
E     5.3%    16.7%  -11.4% (严重不足)
F     0%      10%    -10%   (完全缺失)
```

### 方案4: 提升模型多样性 (优先级: 🟡中)

**当前模型分布**:
- llama-3.1-405b: 29%
- gpt-oss-120b: 21%
- deepseek-v3.2: 14%
- gpt-4: 11%
- glm-4.7: 11%

**建议**: 增加更多模型变体，特别是:
- Claude系列
- Gemini系列
- 国产模型 (文心、通义)

---

## 📋 数据清洗流水线

### 阶段1: 基础清洗

```bash
# 1. 提示词残留清洗
python scripts/data_cleaning/clean_prompt_residue.py \
  --input datasets/generated/... \
  --output datasets/generated/.../cleaned.jsonl

# 2. 去重
python scripts/data_cleaning/deduplicate_samples.py \
  --input datasets/generated/.../cleaned.jsonl \
  --output datasets/generated/.../dedup.jsonl
```

### 阶段2: 质量过滤

```python
# 质量过滤规则
filters = {
    "min_length": 80,
    "max_length": 3000,
    "min_unique_chars": 20,  # 避免重复字符
    "max_repetition_ratio": 0.3,  # 句子重复率
    "forbidden_patterns": [
        r"分析请求",
        r"作为.*我",
        r"^\d+\.\s*$",  # 纯数字列表
    ]
}
```

### 阶段3: 分布检查

```python
# 检查并报告不平衡
def check_distribution(dataset):
    issues = []

    # 场景平衡
    scenario_counts = Counter(d['scenario_id'] for d in dataset)
    if any(c < expected * 0.5 for c in scenario_counts.values()):
        issues.append("场景分布严重不平衡")

    # 主题多样性
    topic_counts = Counter(d['topic'] for d in dataset)
    if len(topic_counts) < 100:
        issues.append(f"主题多样性不足: {len(topic_counts)}")

    return issues
```

---

## ✅ 质量检查清单

### 生成前检查

- [ ] 配置文件包含所有6个场景
- [ ] 每个场景有充足的主题(50+)
- [ ] 模板覆盖所有场景和样式组合
- [ ] 任务队列打乱顺序

### 生成中监控

- [ ] 定期运行 `monitor_generation.py`
- [ ] 检查场景分布是否均衡
- [ ] 检查rejected比例(<10%)
- [ ] 检查主题使用情况

### 生成后验证

- [ ] 运行提示词清洗
- [ ] 运行去重
- [ ] 验证最终分布
- [ ] 抽样人工检查(100条)

---

## 📈 质量指标仪表板

创建一个自动化质量报告:

```bash
python scripts/evaluation/comprehensive_data_quality.py \
  --input datasets/generated/.../final.jsonl \
  --output reports/data_quality_report.html
```

报告内容:
- 总体统计
- 场景分布饼图
- 长度分布直方图
- 主题词云
- 模型覆盖热力图
- 质量问题列表

---

## 🚨 当前紧急行动

### 1. 立即检查F场景 (5分钟)

```bash
# 检查配置是否包含F
grep -r "scenario.*F" configs/scenario_fill*.json

# 检查模板是否有F
grep -r '"F"' scripts/generation/scenario_fill_generate.py
```

### 2. 扩展主题库 (30分钟)

修改 `scripts/generation/scenario_fill_generate.py` 中的 `SCENARIO_TOPICS`:
- 每个场景从25个扩展到50个
- 确保主题不重复
- 主题要具体、多样

### 3. 监控当前生成 (持续)

```bash
python scripts/generation/monitor_generation.py \
  datasets/generated/scenario_fill/2026-01-27_10h_multi_proxies \
  --watch --interval 60
```

---

*创建时间: 2026-01-28*
*状态: 生成进行中，需持续监控*
