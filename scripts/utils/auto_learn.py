#!/usr/bin/env python3
"""
自动学习脚本 - 从工作会话中提取经验并更新技能文件

功能：
1. 扫描训练日志、评估结果、错误记录
2. 提取可复用的经验模式
3. 更新 SKILL.md 经验章节
4. 更新 model_iteration_log.md
"""
import os
import json
from datetime import datetime


def read_latest_training_log():
    """读取最新的训练日志."""
    log_patterns = [
        '/tmp/train_balanced3.log',
        '/tmp/train_balanced.log',
        '/tmp/train_overnight.log'
    ]

    for log_path in log_patterns:
        if os.path.exists(log_path):
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                return ''.join(lines[-200:])  # 最后200行
    return None


def read_comparison_results():
    """读取模型对比结果."""
    result_path = 'models/bert_v2_balanced/comparison_results.json'
    if os.path.exists(result_path):
        with open(result_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def read_iteration_log():
    """读取迭代日志."""
    log_path = 'docs/plans/model_iteration_log.md'
    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""


def extract_insights(train_log, comparison, iteration_log):
    """从日志中提取经验."""
    insights = []
    today = datetime.now().strftime('%Y-%m-%d')

    # 1. 训练参数经验
    if train_log:
        # 提取成功的参数配置
        if 'batch_size=2' in train_log and 'accumulation_steps=8' in train_log:
            if 'CUDA out of memory' not in train_log:
                insights.append({
                    'category': '训练参数',
                    'title': '8GB显存下的稳定配置',
                    'phenomenon': 'RTX 5060 8GB显存训练BERT模型',
                    'cause': 'batch_size=2, accumulation_steps=8, max_len=256',
                    'solution': '等效batch_size=16，训练稳定无OOM',
                    'confidence': '高',
                    'date': today
                })

        # 检查是否有错误模式
        if 'torch.optim.utils' in train_log and 'AttributeError' in train_log:
            insights.append({
                'category': '常见错误',
                'title': 'PyTorch梯度裁剪模块位置',
                'phenomenon': "AttributeError: module 'torch.optim' has no attribute 'utils'",
                'cause': '梯度裁剪在torch.nn.utils而非torch.optim.utils',
                'solution': '使用 torch.nn.utils.clip_grad_norm_',
                'confidence': '高',
                'date': today
            })

    # 2. 评估发现
    if comparison:
        # 检查是否有显著改进
        for scenario, result in comparison.items():
            old_acc = result.get('old_model', {}).get('accuracy', 0)
            new_acc = result.get('new_model', {}).get('accuracy', 0)

            if new_acc - old_acc > 0.05:  # 5%以上的改进
                insights.append({
                    'category': '模型改进',
                    'title': f'{scenario}场景显著改进',
                    'phenomenon': f'准确率从{old_acc:.2%}提升到{new_acc:.2%}',
                    'cause': '分析对比结果判断原因',
                    'solution': '记录改进措施并复用',
                    'confidence': '高',
                    'date': today
                })

    # 3. 数据质量经验
    if iteration_log:
        # 检查是否提到长度偏差问题
        if '长度偏差' in iteration_log or 'length bias' in iteration_log.lower():
            insights.append({
                'category': '数据质量',
                'title': '长度偏差是最关键的数据问题',
                'phenomenon': 'Human平均438字符 vs AI平均910字符时，模型学会捷径',
                'cause': '训练数据中Human和AI的文本长度分布不对齐',
                'solution': '长度分层采样，确保Human/AI平均长度差<50字符',
                'confidence': '高',
                'date': today
            })

    return insights


def format_insight(insight):
    """格式化经验条目."""
    return f"""
### {insight['category']}

**{insight['title']}** ({insight['date']})
- 现象: {insight['phenomenon']}
- 原因: {insight['cause']}
- 解决: {insight['solution']}
- 置信度: {insight['confidence']}
"""


def append_to_skill_file(insights):
    """追加经验到 SKILL.md."""
    skill_path = '/home/anx4758/.claude/skills/ai-text-detector/SKILL.md'

    if not os.path.exists(skill_path):
        print(f"[警告] SKILL.md 不存在: {skill_path}")
        return

    with open(skill_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 查找 "## 经验教训" 章节
    if '## 经验教训' not in content:
        # 如果章节不存在，在文件末尾添加
        content += "\n\n## 经验教训（自动学习）\n"

    # 去重：检查是否已存在类似标题
    new_insights = []
    for insight in insights:
        # 简单的标题匹配去重
        if insight['title'] not in content:
            new_insights.append(insight)

    if not new_insights:
        print("[信息] 没有新的经验需要添加（已存在类似条目）")
        return

    # 追加新经验
    for insight in new_insights:
        content += format_insight(insight)

    # 写回文件
    with open(skill_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"[成功] 已添加 {len(new_insights)} 条新经验到 SKILL.md")
    for insight in new_insights:
        print(f"  - {insight['category']}: {insight['title']}")


def main():
    print("=" * 70)
    print("自动学习 - 提取经验并更新技能文件")
    print("=" * 70)

    # 1. 收集数据
    print("\n[1/3] 收集会话数据...")
    train_log = read_latest_training_log()
    comparison = read_comparison_results()
    iteration_log = read_iteration_log()

    if train_log:
        print("  ✓ 找到训练日志")
    if comparison:
        print("  ✓ 找到对比结果")
    if iteration_log:
        print("  ✓ 找到迭代日志")

    if not any([train_log, comparison, iteration_log]):
        print("[警告] 未找到任何数据源，无法提取经验")
        return

    # 2. 提取经验
    print("\n[2/3] 提取可复用经验...")
    insights = extract_insights(train_log, comparison, iteration_log)

    if not insights:
        print("[信息] 未提取到新经验（可能已记录或无显著发现）")
        return

    print(f"  提取到 {len(insights)} 条经验:")
    for i, insight in enumerate(insights, 1):
        print(f"  {i}. [{insight['category']}] {insight['title']}")

    # 3. 更新技能文件
    print("\n[3/3] 更新 SKILL.md...")
    append_to_skill_file(insights)

    print("\n" + "=" * 70)
    print("自动学习完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()
