# 多样化AI文本生成任务

## 📋 任务概述

基于Gemini的研究方向，生成多样化的AI文本样本，解决当前数据集的问题。

### 任务目标
- **总样本数**: 3000个
- **预计时间**: 4-5小时
- **输出格式**: JSONL (每行一个JSON对象)

## 🎯 生成任务

### 1. 技术文档风格 (1000样本)
**目标**: 解决模型对技术文档式AI识别差的问题

**包含类型**:
- API文档说明
- 技术规范文档
- 算法原理说明
- 系统架构描述
- 配置文件说明

**特点**:
- 使用专业术语
- 包含参数列表、步骤说明
- 格式规范，条目清晰
- 避免对话式、口语化表达

### 2. 学术论文风格 (500样本)
**目标**: 增加学术风格AI样本

**包含类型**:
- 论文摘要
- 方法描述
- 文献综述

**特点**:
- 学术语言规范
- 客观陈述
- 使用被动语态
- 逻辑严谨

### 3. 列表式内容 (800样本)
**目标**: 解决列表式AI识别率低（3%）的问题

**包含类型**:
- 操作步骤说明
- 要点总结
- 检查清单

**特点**:
- 使用序号、项目符号
- 包含冒号、分号
- 信息密集
- 格式规范

### 4. 对抗样本 (300样本)
**目标**: 提高模型鲁棒性

**包含类型**:
- 带拼写错误的AI文本
- 混合正式和非正式语言
- 略显口语化但信息准确

**特点**:
- 模拟人类写作瑕疵
- 不太明显的错误
- 增加识别难度

### 5. 领域特定文本 (400样本)
**目标**: 覆盖特定领域

**包含类型**:
- 代码注释
- 产品说明
- 技术博客

**特点**:
- 领域专业性强
- 实用性高
- 格式多样

## 🚀 使用方法

### 前置条件
1. 反代API服务运行在 `localhost:8317`
2. Python虚拟环境已激活
3. 有足够的磁盘空间（约50MB）

### 启动任务
```bash
cd /mnt/c/datacollection
./start_generation.sh
```

### 查看进度
```bash
# 实时查看日志
tail -f datasets/logs/augmented_v2/generation_log_*.txt

# 查看已生成的文件
ls -lh datasets/logs/augmented_v2/

# 统计已生成样本数
wc -l datasets/logs/augmented_v2/*.jsonl
```

### 停止任务
```bash
# 查找进程ID
ps aux | grep generate_diverse_samples

# 停止任务
kill <PID>
```

## 📊 输出格式

每个样本的JSON格式：
```json
{
  "text": "样本文本内容...",
  "label": 1,
  "style": "technical_doc",
  "source": "glm-4.7",
  "timestamp": "2026-01-26T21:00:00"
}
```

### 字段说明
- `text`: 生成的文本内容
- `label`: 标签 (1=AI, 0=Human)
- `style`: 样本风格类型
- `source`: 生成模型
- `timestamp`: 生成时间

## 📁 输出文件

```
datasets/logs/augmented_v2/
├── technical_docs.jsonl       # 技术文档 (1000)
├── academic_texts.jsonl       # 学术论文 (500)
├── list_contents.jsonl        # 列表式 (800)
├── adversarial_samples.jsonl  # 对抗样本 (300)
├── domain_specific.jsonl      # 领域特定 (400)
├── generation_log_*.txt       # 生成日志
└── nohup.out                  # 后台运行输出
```

## ⏱️ 时间估算

| 任务 | 样本数 | 预计时间 | 说明 |
|------|--------|---------|------|
| 技术文档 | 1000 | 1小时 | 每样本2秒 + API延迟 |
| 学术论文 | 500 | 40分钟 | 每样本2秒 + API延迟 |
| 列表式 | 800 | 1小时 | 每样本2秒 + API延迟 |
| 对抗样本 | 300 | 30分钟 | 每样本2秒 + API延迟 |
| 领域特定 | 400 | 40分钟 | 每样本2秒 + API延迟 |
| **总计** | **3000** | **4-5小时** | 包含错误重试 |

## 🔧 配置调整

### 修改生成数量
编辑 `scripts/generation/generate_diverse_samples.py`:
```python
tasks = [
    ("技术文档", generate_technical_docs, 1000),  # 修改这里
    ("学术论文", generate_academic_texts, 500),
    # ...
]
```

### 修改API配置
```python
API_BASE = "http://localhost:8317/v1/chat/completions"
API_KEY = "cliproxyapi-test-key-2026"
MODEL = "glm-4.7"  # 可改为其他模型
```

### 调整生成参数
```python
temperature=0.7,  # 创造性 (0.1-1.0)
max_tokens=600,   # 最大长度
```

## 📈 后续处理

### 1. 合并数据
```bash
cd datasets/logs/augmented_v2
cat *.jsonl > all_augmented.jsonl
```

### 2. 转换为CSV
```python
import pandas as pd
import json

data = []
with open('all_augmented.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)
df.to_csv('augmented_samples.csv', index=False)
```

### 3. 与现有数据集合并
```python
# 加载现有数据
old_df = pd.read_csv('datasets/active/core_v1/train.csv')

# 加载新数据
new_df = pd.read_csv('datasets/logs/augmented_v2/augmented_samples.csv')

# 合并
combined_df = pd.concat([old_df, new_df], ignore_index=True)

# 打乱
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# 保存
combined_df.to_csv('datasets/final_v3/train.csv', index=False)
```

## ⚠️ 注意事项

1. **API限流**: 脚本已设置2秒延迟，避免触发限流
2. **中断恢复**: 中间结果每50个样本保存一次
3. **错误处理**: API失败会自动跳过并记录日志
4. **磁盘空间**: 确保至少有100MB可用空间
5. **网络稳定**: 需要稳定的网络连接

## 🐛 故障排查

### 问题1: API连接失败
```bash
# 检查反代服务
curl http://localhost:8317/v1/models

# 重启反代服务
# (根据你的反代服务启动方式)
```

### 问题2: 生成速度慢
- 检查API响应时间
- 考虑降低 `time.sleep(2)` 的值
- 使用更快的模型

### 问题3: 内存不足
- 减少单次生成数量
- 分批运行任务

## 📝 日志示例

```
[2026-01-26 21:00:00] 多样化AI文本生成任务启动
[2026-01-26 21:00:00] API: http://localhost:8317/v1/chat/completions
[2026-01-26 21:00:00] 模型: glm-4.7
[2026-01-26 21:00:05] 开始任务: 技术文档
[2026-01-26 21:00:05] 开始生成技术文档样本，目标: 1000个
[2026-01-26 21:02:30] 技术文档: 50/1000 完成
[2026-01-26 21:05:00] 技术文档: 100/1000 完成
...
```

---

**创建时间**: 2026-01-26 21:00
**预计完成**: 2026-01-27 01:00
**负责人**: 自动化脚本
