# 数据集整合方案：旧数据补全 + 新旧合并

## 一、问题分析

### 旧数据 (core_v1) 现状
```
规模: 46,849条
字段: text, label, source, length, category, origin_dataset, origin_split

source分布:
├─ hc3_human: 14,230 (30.4%) → Human, 知识问答
├─ hc3_chatgpt: 12,805 (27.3%) → AI(ChatGPT), 知识问答
├─ thucnews: 6,938 (14.8%) → Human, 新闻
├─ parallel_*: ~12,000 (25%) → AI(多模型), 各类
└─ auto_*: ~1,300 (2.8%) → AI, 自动生成

category分布 (仅10%有值):
├─ C4: 1,951 (4.2%)
├─ Human: 1,533 (3.3%)
├─ C3: 1,256 (2.7%)
└─ 无值: ~42,000 (90%)
```

### 新数据 (generated) 现状
```
规模: 8,367+ (生成中)
字段: text, label, scenario_id, scenario, style_plan, model,
      length_bucket, topic, prompt_id, ...

特点: 元数据完整，可追溯
```

### 核心问题
1. 旧数据缺少 scenario/style/model 字段
2. 新旧数据schema不统一
3. 合并后需要保持一致性

---

## 二、方案一：旧数据元数据补全

### 2.1 Source → Scenario 映射规则

```python
SOURCE_TO_SCENARIO = {
    # HC3数据 → C (知识百科)
    "hc3_human": ("C", "knowledge"),
    "hc3_chatgpt": ("C", "knowledge"),

    # THUCNews → F (新闻资讯)
    "thucnews": ("F", "news"),

    # Parallel生成数据 → A (教育学术) 为主
    # 因为parallel通常是学术/专业文本的AI改写
    "parallel_*": ("A", "education"),

    # Auto生成 → C (知识百科)
    "auto_*": ("C", "knowledge"),

    # 默认
    "default": ("C", "knowledge"),
}
```

### 2.2 Source → Model 映射规则

```python
SOURCE_TO_MODEL = {
    "hc3_chatgpt": "chatgpt-3.5",
    "parallel_gpt-4.1-mini": "gpt-4.1-mini",
    "parallel_deepseek-v3.2": "deepseek-v3.2",
    "parallel_Kimi-K2": "kimi-k2",
    "parallel_gemini-2.5-flash": "gemini-2.5-flash",
    "parallel_claude-haiku-4-5-20251001": "claude-haiku-4.5",
    "parallel_cursor2-gpt-5": "gpt-5",
    "parallel_claude-sonnet-4-5": "claude-sonnet-4.5",
    "parallel_qwen-max-latest": "qwen-max",
    "auto_deepseek": "deepseek",
    "auto_custom": "unknown",
    # Human数据
    "hc3_human": None,
    "thucnews": None,
}
```

### 2.3 文本特征 → Style 推断规则

```python
def infer_style(text: str) -> str:
    """基于文本特征推断style"""

    # 列表特征
    list_patterns = [
        r'^\s*[1-9]\.',          # 1. 2. 3.
        r'^\s*[一二三四五六七八九十]、',  # 一、二、
        r'^\s*[-•●]',            # 无序列表
        r'^\s*第[一二三四五六七八九十]',  # 第一、第二
    ]

    # 对话特征
    dialogue_patterns = [
        r'[""「」].*?[""」」]',   # 引号对话
        r'^\s*[甲乙丙丁AB][:：]',  # 角色对话
        r'问[:：]|答[:：]',        # 问答形式
    ]

    # 报告/正式特征
    report_patterns = [
        r'摘要|引言|背景|结论|总结',
        r'研究表明|数据显示|分析发现',
        r'综上所述|由此可见',
    ]

    import re
    lines = text.split('\n')

    # 检查列表
    list_lines = sum(1 for line in lines if any(re.match(p, line) for p in list_patterns))
    if list_lines >= 3:
        return "list"

    # 检查对话
    if any(re.search(p, text) for p in dialogue_patterns):
        return "dialogue"

    # 检查报告
    if any(re.search(p, text) for p in report_patterns):
        return "report"

    # 默认
    return "explanation"
```

### 2.4 Length → Length Bucket 映射

```python
def get_length_bucket(length: int) -> str:
    if length < 80:
        return "0-80"
    elif length < 200:
        return "80-200"
    elif length < 500:
        return "200-500"
    elif length < 1000:
        return "500-1000"
    elif length < 2000:
        return "1000-2000"
    else:
        return "2000+"
```

---

## 三、方案二：统一Schema设计

### 3.1 目标Schema (v2)

```python
UNIFIED_SCHEMA = {
    # === 核心字段 ===
    "text_id": str,          # 唯一ID (hash)
    "text": str,             # 文本内容
    "label": int,            # 0=Human, 1=AI

    # === 场景/风格 ===
    "scenario_id": str,      # A/B/C/D/E/F
    "scenario": str,         # education/workplace/knowledge/community/commerce/news
    "style": str,            # list/guide/report/explanation/dialogue/mixed

    # === 长度 ===
    "length": int,           # 字符数
    "length_bucket": str,    # 80-200/200-500/500-1000/1000-2000/2000+

    # === 来源追溯 ===
    "source": str,           # 原始来源标识
    "source_type": str,      # "legacy" | "generated" | "collected"
    "model": str,            # AI模型名 (Human为null)

    # === 元数据 ===
    "created_at": str,       # ISO时间戳
    "schema_version": str,   # "v2"
}
```

### 3.2 旧数据转换函数

```python
def convert_legacy_to_v2(row: dict) -> dict:
    """将core_v1格式转换为v2 schema"""
    import hashlib

    text = row['text']
    source = row['source']
    label = int(row['label'])
    length = int(row.get('length', len(text)))

    # 推断scenario
    scenario_id, scenario = infer_scenario_from_source(source)

    # 推断style
    style = infer_style(text)

    # 推断model
    model = SOURCE_TO_MODEL.get(source)
    if model is None and label == 1:
        model = "unknown"

    return {
        "text_id": hashlib.md5(text.encode()).hexdigest()[:12],
        "text": text,
        "label": label,
        "scenario_id": scenario_id,
        "scenario": scenario,
        "style": style,
        "length": length,
        "length_bucket": get_length_bucket(length),
        "source": source,
        "source_type": "legacy",
        "model": model,
        "created_at": "2026-01-27T00:00:00",  # 旧数据统一时间
        "schema_version": "v2",
    }
```

### 3.3 新数据转换函数

```python
def convert_generated_to_v2(row: dict) -> dict:
    """将generated格式转换为v2 schema"""
    import hashlib

    text = row['text']

    return {
        "text_id": row.get('text_id', hashlib.md5(text.encode()).hexdigest()[:12]),
        "text": text,
        "label": 1,  # 生成数据都是AI
        "scenario_id": row['scenario_id'],
        "scenario": row['scenario'],
        "style": row['style_plan'],
        "length": len(text),
        "length_bucket": row['length_bucket'],
        "source": f"generated_{row['model']}",
        "source_type": "generated",
        "model": row['model'],
        "created_at": row.get('created_at', ''),
        "schema_version": "v2",
    }
```

---

## 四、合并策略

### 4.1 数据流

```
┌─────────────────┐     ┌─────────────────┐
│   core_v1       │     │   generated     │
│   (46,849)      │     │   (8,367+)      │
│   旧schema      │     │   新schema      │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│ convert_legacy  │     │ convert_generated│
│   _to_v2()      │     │   _to_v2()       │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └──────────┬────────────┘
                    ▼
         ┌─────────────────┐
         │   去重检查       │
         │   (text_id)     │
         └────────┬────────┘
                  ▼
         ┌─────────────────┐
         │  标签平衡检查    │
         │  (50:50 目标)   │
         └────────┬────────┘
                  ▼
         ┌─────────────────┐
         │   core_v2       │
         │   (统一schema)  │
         └─────────────────┘
```

### 4.2 合并优先级

1. **新生成数据优先** - 元数据更准确
2. **去重时保留新数据** - 如果text相同
3. **Human数据不丢弃** - 平衡需要

### 4.3 平衡策略

```python
def balance_dataset(df, target_ratio=0.5):
    """平衡AI/Human比例"""
    ai_df = df[df['label'] == 1]
    human_df = df[df['label'] == 0]

    target_count = min(len(ai_df), len(human_df))

    # 随机采样到平衡
    ai_balanced = ai_df.sample(n=target_count, random_state=42)
    human_balanced = human_df.sample(n=target_count, random_state=42)

    return pd.concat([ai_balanced, human_balanced]).sample(frac=1, random_state=42)
```

---

## 五、执行计划

### 阶段1: Claude Code 准备 (现在)
- [x] 设计schema v2
- [x] 设计映射规则
- [ ] 编写转换脚本

### 阶段2: Codex 执行
- [ ] 等待生成任务完成
- [ ] 执行数据清洗
- [ ] 运行转换脚本
- [ ] 验证合并结果

### 阶段3: 验证
- [ ] 检查场景分布
- [ ] 检查标签平衡
- [ ] 检查去重效果
- [ ] 生成质量报告

---

## 六、风险与缓解

| 风险 | 缓解措施 |
|-----|---------|
| Source推断scenario不准确 | 保留原始source字段，可追溯 |
| Style推断有误差 | 使用保守规则，宁可归为explanation |
| 合并后重复 | 基于text_id去重 |
| 标签不平衡 | 最后一步做平衡采样 |

---

*创建时间: 2026-01-28*
*状态: 方案设计完成，待执行*
