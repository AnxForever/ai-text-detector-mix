# Agent快速使用指南

## 🤖 已创建的6个专用Agent

### 1️⃣ data-collector（数据收集）
**用途**：收集真实人类文本数据
```bash
# 示例任务
"帮我收集HC3数据集"
"验证数据真实性"
"检查是否有AI生成的假人类数据"
```

### 2️⃣ data-processor（数据处理）
**用途**：数据清洗、平衡、去偏
```bash
# 示例任务
"处理长度偏差问题"
"应用格式去偏"
"重新划分数据集"
```

### 3️⃣ feature-engineer（特征工程）
**用途**：提取统计和图特征
```bash
# 示例任务
"提取统计特征"
"构建文本图"
"批量提取图特征"
```

### 4️⃣ model-trainer（模型训练）
**用途**：训练各种模型
```bash
# 示例任务
"训练长度平衡模型"
"训练混合特征模型"
"从已有模型继续训练"
```

### 5️⃣ experiment-evaluator（实验评估）
**用途**：全面评估模型
```bash
# 示例任务
"评估模型在各长度区间的性能"
"进行对抗鲁棒性测试"
"对比不同模型"
```

### 6️⃣ paper-assistant（论文助手）
**用途**：辅助论文撰写
```bash
# 示例任务
"生成实验结果表格"
"整理关键数据"
"提供写作建议"
```

---

## 🚀 使用方法

### 方法1：直接对话
```
你：帮我处理数据集的长度偏差
我：[自动调用 data-processor Agent]
   运行 balance_length_distribution.py...
```

### 方法2：指定Agent
```
你：@data-processor 处理长度偏差
你：@model-trainer 训练新模型
你：@experiment-evaluator 评估性能
```

### 方法3：工作流
```
你：执行完整训练流程

我会自动执行：
1. data-collector: 验证数据
2. data-processor: 平衡+去偏
3. feature-engineer: 提取特征
4. model-trainer: 训练模型
5. experiment-evaluator: 评估
6. paper-assistant: 生成报告
```

---

## 📋 常用工作流

### 工作流1：数据准备
```
1. @data-collector 验证数据真实性
2. @data-processor 长度平衡
3. @data-processor 格式去偏
4. @data-processor 划分数据集
```

### 工作流2：模型训练
```
1. @data-processor 准备训练数据
2. @feature-engineer 提取特征（可选）
3. @model-trainer 训练模型
4. @experiment-evaluator 评估性能
```

### 工作流3：论文撰写
```
1. @experiment-evaluator 整理所有实验结果
2. @paper-assistant 生成表格和图表
3. @paper-assistant 提供写作建议
```

---

## 💡 智能调度

Agent调度器会根据你的任务描述自动选择合适的Agent：

| 关键词 | 调用的Agent |
|--------|------------|
| 收集、数据集、验证 | data-collector |
| 清洗、平衡、长度、格式 | data-processor |
| 特征、提取、统计、图 | feature-engineer |
| 训练、模型、BERT | model-trainer |
| 评估、测试、准确率 | experiment-evaluator |
| 论文、写作、表格 | paper-assistant |

---

## 🎯 当前项目状态

### ✅ 已完成
- 数据真实性验证（THUCNews真实人类文本）
- 长度偏差处理（78% → 34%）
- 数据集重新划分（5730/1229/1231）
- 所有工具脚本就绪

### ⏳ 待完成
- 安装PyTorch环境
- 训练长度平衡模型
- 实现混合特征模型
- 全面评估实验

---

## 📞 使用示例

### 示例1：处理当前任务
```
你：现在数据已经平衡了，下一步做什么？

我：[调用 model-trainer Agent]
建议：训练长度平衡模型
命令：python3 scripts/training/continue_training.py ...
```

### 示例2：准备论文
```
你：帮我整理实验结果用于论文

我：[调用 paper-assistant Agent]
1. 读取 EXPERIMENT_FINAL_RESULTS.md
2. 生成表格
3. 提供写作建议
```

### 示例3：快速实验
```
你：我想对比长度平衡前后的效果

我：[调用 experiment-evaluator Agent]
1. 评估原始模型
2. 评估平衡模型
3. 生成对比报告
```

---

## 🔧 配置文件位置

```
.kiro/
├── agents/
│   ├── data-collector.json
│   ├── data-processor.json
│   ├── feature-engineer.json
│   ├── model-trainer.json
│   ├── experiment-evaluator.json
│   └── paper-assistant.json
├── agent_dispatcher.py
└── AGENTS_README.md
```

---

**现在你可以直接说任务，我会自动调用合适的Agent来完成！** 🚀
