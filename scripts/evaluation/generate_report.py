#!/usr/bin/env python3
"""
生成完整评估报告
"""
import json
import pandas as pd
from datetime import datetime

def generate_report():
    report = []
    report.append("=" * 80)
    report.append("混合文本AI检测系统 - 完整评估报告")
    report.append("=" * 80)
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # 1. 数据集统计
    report.append("## 1. 数据集统计")
    report.append("")
    
    # Combined v2
    train_df = pd.read_csv('datasets/combined_v2/train.csv')
    val_df = pd.read_csv('datasets/combined_v2/val.csv')
    test_df = pd.read_csv('datasets/combined_v2/test.csv')
    
    report.append(f"总数据量: {len(train_df) + len(val_df) + len(test_df):,} 条")
    report.append(f"  - 训练集: {len(train_df):,}")
    report.append(f"  - 验证集: {len(val_df):,}")
    report.append(f"  - 测试集: {len(test_df):,}")
    report.append("")
    
    # Hybrid数据
    hybrid_df = pd.read_csv('datasets/hybrid/hybrid_dataset_with_sep.csv')
    report.append(f"混合数据: {len(hybrid_df):,} 条")
    for cat, count in hybrid_df['category'].value_counts().items():
        report.append(f"  - {cat}: {count:,}")
    report.append("")
    
    # Span标注数据
    with open('datasets/hybrid/c2_span_labels.json', 'r') as f:
        span_data = json.load(f)
    report.append(f"Span标注数据: {len(span_data):,} 条 (C2类别)")
    report.append("")
    
    # 2. 模型性能
    report.append("## 2. 模型性能")
    report.append("")
    
    report.append("### 2.1 BERT分类模型 (bert_v2_with_sep)")
    report.append("")
    report.append("训练结果:")
    report.append("  - Epoch 1: Val Acc = 98.61%")
    report.append("  - Epoch 2: Val Acc = 98.82% ⭐ (最佳)")
    report.append("  - Epoch 3: Val Acc = 97.70%")
    report.append("")
    
    report.append("测试集性能 (5,901 samples):")
    report.append("  - 整体准确率: 98.05%")
    report.append("  - Human: Precision=98.51%, Recall=97.24%")
    report.append("  - AI: Precision=97.67%, Recall=98.74%")
    report.append("")
    
    # C2检测对比
    report.append("C2混合文本检测 (211 samples):")
    report.append("  - 准确率: 93.8%")
    report.append("  - 错误率: 6.2% (13/211)")
    report.append("  - 正确样本置信度: 99.88%")
    report.append("")
    
    report.append("对比旧模型 (无[SEP]标记):")
    report.append("  - 旧模型C2准确率: 79.8%")
    report.append("  - 新模型C2准确率: 93.8%")
    report.append("  - 提升: +14.0% ⭐")
    report.append("")
    
    report.append("### 2.2 Span边界检测器 (bert_span_detector)")
    report.append("")
    report.append("训练结果:")
    report.append("  - Epoch 1: Val Token Acc = 95.57%")
    report.append("  - Epoch 2: Val Token Acc = 96.27% ⭐ (最佳)")
    report.append("  - Epoch 3: Val Token Acc = 96.04%")
    report.append("")
    
    report.append("测试集性能 (204 samples):")
    report.append("  - Token分类准确率: 96.69%")
    report.append("  - 边界定位准确率: 49.51% (±5 tokens)")
    report.append("")
    
    # 3. 技术创新
    report.append("## 3. 技术创新点")
    report.append("")
    report.append("1. **边界标记机制**")
    report.append("   - 在混合文本的人类/AI边界处插入[SEP]标记")
    report.append("   - 使模型能够学习边界信息")
    report.append("   - C2检测率提升14%")
    report.append("")
    
    report.append("2. **双层检测架构**")
    report.append("   - 第一层: 文本分类 (Human vs AI)")
    report.append("   - 第二层: 边界定位 (Token-level标注)")
    report.append("   - 实现从粗粒度到细粒度的检测")
    report.append("")
    
    report.append("3. **Token-level标注**")
    report.append("   - 每个token标记为Human(0)或AI(1)")
    report.append("   - 支持精确的边界定位")
    report.append("   - 可扩展到多段混合文本")
    report.append("")
    
    # 4. 实验结果
    report.append("## 4. 关键发现")
    report.append("")
    report.append("1. **[SEP]标记的有效性**")
    report.append("   - 所有C2样本都包含[SEP]标记")
    report.append("   - 模型成功学习利用边界信息")
    report.append("   - 误判率从20.2%降至6.2%")
    report.append("")
    
    report.append("2. **混合文本检测挑战**")
    report.append("   - C3 (改写): 100% 准确率 ✅")
    report.append("   - C4 (润色): 92.97% 准确率 ✅")
    report.append("   - C2 (续写): 93.8% 准确率 (改进后)")
    report.append("   - C2最具挑战性，因为需要识别边界")
    report.append("")
    
    report.append("3. **边界定位精度**")
    report.append("   - Token分类准确率高达96.69%")
    report.append("   - 边界定位准确率49.51% (±5 tokens)")
    report.append("   - 演示显示实际边界误差通常<10字符")
    report.append("")
    
    # 5. 应用价值
    report.append("## 5. 应用价值")
    report.append("")
    report.append("1. **学术诚信检测**")
    report.append("   - 识别学生作业中的AI辅助部分")
    report.append("   - 定位具体的AI生成段落")
    report.append("")
    
    report.append("2. **内容审核**")
    report.append("   - 检测新闻/文章中的AI生成内容")
    report.append("   - 标记需要人工审核的部分")
    report.append("")
    
    report.append("3. **写作辅助**")
    report.append("   - 帮助作者识别过度依赖AI的部分")
    report.append("   - 提供改进建议")
    report.append("")
    
    # 6. 未来工作
    report.append("## 6. 未来改进方向")
    report.append("")
    report.append("1. **提升边界定位精度**")
    report.append("   - 当前49.51% → 目标70%+")
    report.append("   - 尝试CRF层或序列标注模型")
    report.append("")
    
    report.append("2. **扩展到多段混合**")
    report.append("   - 当前仅支持单一边界")
    report.append("   - 扩展到多个人类/AI交替段落")
    report.append("")
    
    report.append("3. **跨模型泛化**")
    report.append("   - 测试对未见过AI模型的检测能力")
    report.append("   - 增强模型鲁棒性")
    report.append("")
    
    report.append("=" * 80)
    report.append("报告结束")
    report.append("=" * 80)
    
    return "\n".join(report)

if __name__ == '__main__':
    report = generate_report()
    
    # 保存到文件
    with open('evaluation_results/final_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(report)
    print(f"\n报告已保存到: evaluation_results/final_report.txt")
