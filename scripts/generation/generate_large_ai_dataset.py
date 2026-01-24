#!/usr/bin/env python3
"""
大规模AI文本生成工具
目标：生成5万条多模型AI文本
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict
import time

class LargeScaleAIGenerator:
    def __init__(self, output_dir="datasets/ai_large"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 多样化的提示词模板
        self.prompt_templates = self._load_prompts()
        
    def _load_prompts(self) -> List[Dict]:
        """加载多样化的提示词"""
        return [
            # 科技类
            {"domain": "科技", "prompt": "请用400字介绍{topic}的发展现状和未来趋势。"},
            {"domain": "科技", "prompt": "分析{topic}技术的优势和挑战。"},
            
            # 教育类
            {"domain": "教育", "prompt": "讨论{topic}在现代教育中的重要性。"},
            {"domain": "教育", "prompt": "如何改进{topic}相关的教学方法？"},
            
            # 社会类
            {"domain": "社会", "prompt": "评价{topic}对社会的影响。"},
            {"domain": "社会", "prompt": "探讨{topic}现象的成因和解决方案。"},
            
            # 文化类
            {"domain": "文化", "prompt": "介绍{topic}的文化内涵和传承价值。"},
            {"domain": "文化", "prompt": "分析{topic}在当代的发展和创新。"},
            
            # 经济类
            {"domain": "经济", "prompt": "分析{topic}的经济影响和市场前景。"},
            {"domain": "经济", "prompt": "讨论{topic}的商业模式和盈利方式。"},
        ]
    
    def generate_plan(self, total_count=50000, models_per_prompt=3):
        """生成任务计划"""
        print(f"生成任务计划: {total_count}条文本")
        
        topics = [
            # 科技
            "人工智能", "区块链", "量子计算", "5G通信", "云计算",
            "物联网", "大数据", "机器学习", "自动驾驶", "虚拟现实",
            
            # 教育
            "在线教育", "素质教育", "职业教育", "终身学习", "教育公平",
            
            # 社会
            "老龄化", "城市化", "环境保护", "公共卫生", "社会保障",
            
            # 文化
            "传统文化", "文化创意", "非遗保护", "文化交流", "文化自信",
            
            # 经济
            "数字经济", "共享经济", "绿色经济", "平台经济", "消费升级",
        ]
        
        plan = []
        task_id = 0
        
        for template in self.prompt_templates:
            for topic in topics:
                prompt = template["prompt"].format(topic=topic)
                plan.append({
                    "id": task_id,
                    "domain": template["domain"],
                    "topic": topic,
                    "prompt": prompt,
                    "models": ["gpt-4", "claude-3", "gemini-pro"][:models_per_prompt]
                })
                task_id += 1
                
                if task_id >= total_count:
                    break
            if task_id >= total_count:
                break
        
        plan_file = self.output_dir / "generation_plan.json"
        with open(plan_file, 'w', encoding='utf-8') as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 计划已生成: {len(plan)} 个任务")
        print(f"  保存到: {plan_file}")
        return plan
    
    def generate_adversarial_samples(self, ai_texts_file, target=10000):
        """生成对抗样本（人工润色的AI文本）"""
        print(f"\n生成对抗样本: {target}条")
        print("方法: AI生成 -> 改写模型润色 -> 标注为AI")
        
        # 这里需要实现：
        # 1. 读取AI文本
        # 2. 使用另一个模型改写（如Claude改写GPT的输出）
        # 3. 保持语义但改变表达方式
        
        print("⚠️  对抗样本生成需要调用API，请参考 parallel_generation.py 实现")
        print("   建议策略:")
        print("   - 同义词替换")
        print("   - 句式重组")
        print("   - 段落重排")
        print("   - 添加/删除连接词")


def main():
    print("="*60)
    print("大规模AI文本生成计划")
    print("="*60)
    
    generator = LargeScaleAIGenerator()
    
    print("\n选项:")
    print("  1. 生成任务计划（50,000条）")
    print("  2. 生成对抗样本计划（10,000条）")
    print("  3. 全部生成")
    
    choice = input("\n请选择 (1-3): ").strip()
    
    if choice == '1':
        generator.generate_plan(total_count=50000)
    elif choice == '2':
        ai_file = input("请输入AI文本文件路径: ").strip()
        generator.generate_adversarial_samples(ai_file, target=10000)
    elif choice == '3':
        generator.generate_plan(total_count=50000)
        print("\n⚠️  生成AI文本需要调用API，请使用 parallel_generation.py")
        print("   生成完成后再运行对抗样本生成")
    
    print("\n下一步:")
    print("  1. 使用 parallel_generation.py 执行生成任务")
    print("  2. 检查生成质量")
    print("  3. 生成对抗样本")


if __name__ == "__main__":
    main()
