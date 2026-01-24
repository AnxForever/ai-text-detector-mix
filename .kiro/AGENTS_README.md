# AI文本检测项目 - 专用Agent配置

本项目的专用Agent，用于提高毕业设计工作效率。

## Agent列表

### 1. 数据收集Agent (data-collector)
**职责**：收集和验证真实人类文本数据
- 从HC3、THUCNews等标准数据集收集
- 验证数据真实性（排除AI生成的"人类风格"文本）
- 统计数据质量指标

**使用场景**：
- 需要扩充数据集时
- 验证数据来源时
- 检查数据质量时

---

### 2. 数据处理Agent (data-processor)
**职责**：数据清洗、平衡、去偏
- 长度分布平衡
- 格式去偏处理
- 数据集划分（train/val/test）

**使用场景**：
- 发现数据偏差时
- 需要重新划分数据集时
- 应用格式去偏时

---

### 3. 特征工程Agent (feature-engineer)
**职责**：提取和分析特征
- 统计特征提取（困惑度、TTR、词性分布）
- 图特征提取（实体关系图）
- 特征重要性分析

**使用场景**：
- 实现混合特征模型时
- 分析特征有效性时
- 进行消融实验时

---

### 4. 模型训练Agent (model-trainer)
**职责**：模型训练和优化
- 配置训练参数
- 监控训练过程
- 保存最佳模型

**使用场景**：
- 训练新模型时
- 继续训练时
- 调整超参数时

---

### 5. 实验评估Agent (experiment-evaluator)
**职责**：全面评估模型性能
- 标准指标评估（准确率、F1等）
- 长度感知评估
- 跨模型泛化测试
- 对抗鲁棒性测试

**使用场景**：
- 评估模型性能时
- 进行消融实验时
- 对比不同模型时

---

### 6. 论文助手Agent (paper-assistant)
**职责**：辅助论文撰写
- 生成实验结果表格
- 绘制性能对比图
- 整理关键数据
- 提供写作建议

**使用场景**：
- 撰写论文时
- 准备答辩材料时
- 整理实验结果时

---

## 使用方法

### 方式1：直接调用Agent
```bash
# 示例：调用数据处理Agent
kiro agent data-processor "处理长度偏差问题"
```

### 方式2：在对话中指定Agent
```
@data-collector 帮我收集HC3数据集
@feature-engineer 提取图特征
@experiment-evaluator 评估模型在各长度区间的性能
```

### 方式3：Agent协作
```
让data-processor和model-trainer协作：
1. 先平衡数据长度
2. 然后训练新模型
```

---

## Agent工作流示例

### 完整训练流程
```
1. @data-collector: 验证数据真实性
2. @data-processor: 长度平衡 + 格式去偏
3. @feature-engineer: 提取统计特征
4. @model-trainer: 训练混合特征模型
5. @experiment-evaluator: 全面评估
6. @paper-assistant: 生成结果表格
```

### 快速实验流程
```
1. @data-processor: 准备实验数据
2. @model-trainer: 训练对比模型
3. @experiment-evaluator: 对比评估
```

---

## Agent配置文件

每个Agent都有独立的配置文件，位于：
```
.kiro/agents/
├── data-collector.json
├── data-processor.json
├── feature-engineer.json
├── model-trainer.json
├── experiment-evaluator.json
└── paper-assistant.json
```

---

## 注意事项

1. **数据真实性**：data-collector只收集真实人类文本，不生成
2. **资源管理**：model-trainer会检查GPU/内存资源
3. **版本控制**：所有Agent操作都会记录日志
4. **协作模式**：多个Agent可以串行或并行工作

---

## 下一步

创建具体的Agent配置文件...
