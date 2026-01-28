#!/usr/bin/env python3
"""
多模型数据生成框架
支持: DeepSeek, GPT, Gemini, Claude, Qwen, GLM
"""
import os
import sys
import json
import random
import time
from datetime import datetime

from openai import OpenAI

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from scripts.utils.api_config import get_proxy_config

# API配置
remote = get_proxy_config("remote_proxy", "REMOTE_PROXY", "https://api.hotaruapi.top/v1")
APIS = {
    'remote': {
        'base_url': remote['url'],
        'key': remote['key'],
        'models': ['deepseek-ai/deepseek-v3.1', 'gemini-2.5-flash', 'gpt-4.1-mini']
    }
}

# 生成类型配置
GEN_TYPES = {
    'C2_continue': {  # 续写
        'system': '你是一位专业的中文写作助手。',
        'prompt': '请自然地续写以下文本，保持风格一致，续写约150-300字：\n\n{text}'
    },
    'C3_rewrite': {  # 改写
        'system': '你是一位专业的中文改写专家。',
        'prompt': '请用不同的表达方式改写以下文本，保持原意但改变句式和用词：\n\n{text}'
    },
    'C4_polish': {  # 润色
        'system': '你是一位专业的中文编辑。',
        'prompt': '请润色以下文本，使其更加流畅、专业，修正语法错误：\n\n{text}'
    },
    'T1_tech_doc': {  # 技术文档
        'system': '你是一位资深技术文档工程师。直接输出技术内容，不要任何开场白。',
        'prompt': '为以下函数编写API文档，包含参数说明、返回值、示例代码：\n\n函数名: {func_name}\n功能: {func_desc}'
    },
    'T2_table': {  # 表格生成
        'system': '你是一位技术文档专家。直接输出Markdown表格，不要解释。',
        'prompt': '生成一个关于"{topic}"的Markdown参数配置表，包含参数名、类型、默认值、说明四列，至少8行。'
    }
}

class MultiModelGenerator:
    def __init__(self):
        self.clients = {}
        for name, cfg in APIS.items():
            self.clients[name] = OpenAI(base_url=cfg['base_url'], api_key=cfg['key'])
    
    def generate(self, model: str, gen_type: str, **kwargs) -> dict:
        """生成单条数据"""
        cfg = GEN_TYPES[gen_type]
        prompt = cfg['prompt'].format(**kwargs)
        
        # 选择API
        api_name = 'remote'
        client = self.clients[api_name]
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {'role': 'system', 'content': cfg['system']},
                    {'role': 'user', 'content': prompt}
                ],
                max_tokens=1024,
                temperature=0.7
            )
            content = response.choices[0].message.content
            return {
                'success': True,
                'content': content,
                'model': model,
                'gen_type': gen_type,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {'success': False, 'error': str(e), 'model': model}
    
    def batch_generate_c2(self, human_texts: list, models: list, output_file: str):
        """批量生成C2续写数据"""
        results = []
        
        for i, text in enumerate(human_texts):
            # 随机截断位置 (20%-80%)
            cut_ratio = random.uniform(0.2, 0.8)
            cut_pos = int(len(text) * cut_ratio)
            human_part = text[:cut_pos]
            
            # 随机选择模型
            model = random.choice(models)
            
            result = self.generate(model, 'C2_continue', text=human_part)
            
            if result['success']:
                results.append({
                    'human_prefix': human_part,
                    'ai_continuation': result['content'],
                    'full_text': human_part + result['content'],
                    'boundary': len(human_part),
                    'model': model,
                    'label': 1,
                    'category': 'C2'
                })
                print(f"[{i+1}/{len(human_texts)}] {model} ✓")
            else:
                print(f"[{i+1}/{len(human_texts)}] {model} ✗ {result['error']}")
            
            time.sleep(0.5)  # 避免限流
        
        # 保存
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n生成完成: {len(results)} 条, 保存到 {output_file}")
        return results


def test_api():
    """测试API连接"""
    gen = MultiModelGenerator()
    
    print("测试多模型API连接...\n")
    
    for model in ['deepseek-ai/deepseek-v3.1', 'gemini-2.5-flash']:
        result = gen.generate(
            model=model,
            gen_type='C2_continue',
            text='人工智能正在改变我们的生活方式'
        )
        
        if result['success']:
            print(f"✓ {model}")
            print(f"  输出: {result['content'][:100]}...")
        else:
            print(f"✗ {model}: {result['error']}")
        print()


if __name__ == '__main__':
    test_api()
