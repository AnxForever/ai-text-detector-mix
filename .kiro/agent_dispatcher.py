#!/usr/bin/env python3
"""
AI文本检测项目 - Agent调度器
根据任务自动选择和调用合适的Agent
"""

import json
from pathlib import Path

class AgentDispatcher:
    """Agent调度器"""
    
    def __init__(self, agents_dir=".kiro/agents"):
        self.agents_dir = Path(agents_dir)
        self.agents = self._load_agents()
    
    def _load_agents(self):
        """加载所有Agent配置"""
        agents = {}
        for config_file in self.agents_dir.glob("*.json"):
            with open(config_file, 'r', encoding='utf-8') as f:
                agent = json.load(f)
                agents[agent['name']] = agent
        return agents
    
    def dispatch(self, task_description):
        """根据任务描述调度Agent"""
        task_lower = task_description.lower()
        
        # 关键词匹配
        keywords = {
            'data-collector': ['收集', '数据集', 'hc3', 'thucnews', '验证', '真实性'],
            'data-processor': ['清洗', '平衡', '长度', '格式', '去偏', '划分'],
            'feature-engineer': ['特征', '提取', '统计', '图', 'gcn', '困惑度'],
            'model-trainer': ['训练', '模型', 'bert', '混合', '多任务'],
            'experiment-evaluator': ['评估', '测试', '准确率', 'f1', '鲁棒性', '对抗'],
            'paper-assistant': ['论文', '写作', '表格', '图表', '答辩']
        }
        
        # 计算匹配分数
        scores = {}
        for agent_name, agent_keywords in keywords.items():
            score = sum(1 for kw in agent_keywords if kw in task_lower)
            if score > 0:
                scores[agent_name] = score
        
        if not scores:
            return None
        
        # 返回最匹配的Agent
        best_agent = max(scores, key=scores.get)
        return self.agents[best_agent]
    
    def suggest_workflow(self, goal):
        """根据目标建议工作流"""
        workflows = {
            '完整训练': [
                ('data-collector', '验证数据真实性'),
                ('data-processor', '长度平衡 + 格式去偏'),
                ('feature-engineer', '提取统计特征'),
                ('model-trainer', '训练混合特征模型'),
                ('experiment-evaluator', '全面评估'),
                ('paper-assistant', '生成结果表格')
            ],
            '快速实验': [
                ('data-processor', '准备实验数据'),
                ('model-trainer', '训练对比模型'),
                ('experiment-evaluator', '对比评估')
            ],
            '论文撰写': [
                ('experiment-evaluator', '整理实验结果'),
                ('paper-assistant', '生成表格和图表'),
                ('paper-assistant', '提供写作建议')
            ]
        }
        
        return workflows.get(goal, [])
    
    def list_agents(self):
        """列出所有可用Agent"""
        print("可用的Agent:")
        print("="*60)
        for name, agent in self.agents.items():
            print(f"\n{name}:")
            print(f"  描述: {agent['description']}")
            print(f"  能力: {', '.join(agent['capabilities'][:3])}...")
        print("\n" + "="*60)


def main():
    """示例使用"""
    dispatcher = AgentDispatcher()
    
    print("AI文本检测项目 - Agent调度器")
    print("="*60)
    
    # 示例1：任务调度
    print("\n示例1：任务调度")
    task = "帮我处理数据集的长度偏差问题"
    agent = dispatcher.dispatch(task)
    if agent:
        print(f"任务: {task}")
        print(f"推荐Agent: {agent['name']}")
        print(f"工具: {agent['tools'][0]}")
    
    # 示例2：工作流建议
    print("\n示例2：工作流建议")
    workflow = dispatcher.suggest_workflow('完整训练')
    print("完整训练流程:")
    for i, (agent_name, task) in enumerate(workflow, 1):
        print(f"  {i}. [{agent_name}] {task}")
    
    # 示例3：列出所有Agent
    print("\n示例3：所有可用Agent")
    dispatcher.list_agents()


if __name__ == "__main__":
    main()
