#!/usr/bin/env python3
"""构建困难测试集：通过数据增强模拟人类化AI文本"""
import re, random
import pandas as pd

# AI常用词/短语（去掉这些让AI文本更像人类）
AI_PATTERNS = [
    (r'首先[，,]?', ''), (r'其次[，,]?', ''), (r'最后[，,]?', ''),
    (r'总之[，,]?', ''), (r'综上所述[，,]?', ''), (r'总的来说[，,]?', ''),
    (r'值得注意的是[，,]?', ''), (r'需要指出的是[，,]?', ''),
    (r'一方面.*另一方面', ''), (r'不仅.*而且', ''),
    (r'此外[，,]?', ''), (r'另外[，,]?', ''), (r'同时[，,]?', ''),
    (r'\d+\.\s*', ''),  # 去掉编号
    (r'[一二三四五六七八九十][、.]\s*', ''),  # 去掉中文编号
]

# 口语化替换
CASUAL_REPLACE = [
    ('因此', '所以'), ('然而', '但是'), ('尽管', '虽然'),
    ('此外', '还有'), ('例如', '比如'), ('即', '就是'),
    ('较为', '比较'), ('非常', '很'), ('十分', '挺'),
    ('能够', '能'), ('可以', '能'), ('应该', '得'),
]

def humanize_ai_text(text):
    """让AI文本更像人类写的"""
    # 去掉AI特征词
    for pattern, repl in AI_PATTERNS:
        text = re.sub(pattern, repl, text)
    
    # 口语化替换
    for formal, casual in CASUAL_REPLACE:
        if random.random() > 0.5:
            text = text.replace(formal, casual)
    
    # 随机加语气词
    if random.random() > 0.7:
        particles = ['啊', '呢', '吧', '嘛', '哦', '呀']
        sentences = text.split('。')
        if len(sentences) > 2:
            idx = random.randint(0, len(sentences)-2)
            sentences[idx] += random.choice(particles)
            text = '。'.join(sentences)
    
    return text.strip()

def main():
    # 读取AI文本
    ai_df = pd.read_csv('datasets/final_clean/all_ai.csv')
    
    # 随机采样500条进行人类化处理
    samples = ai_df.sample(min(500, len(ai_df)))
    
    hard_samples = []
    for _, row in samples.iterrows():
        original = row['text']
        humanized = humanize_ai_text(original)
        
        # 只保留变化明显的
        if len(humanized) > 50 and humanized != original:
            hard_samples.append({
                'text': humanized,
                'label': 1,  # 仍然是AI
                'type': 'humanized_ai'
            })
    
    # 同时采样一些人类文本作为对照
    human_df = pd.read_csv('datasets/final_clean/all_human.csv')
    human_samples = human_df.sample(min(500, len(human_df)))
    
    for _, row in human_samples.iterrows():
        hard_samples.append({
            'text': row['text'],
            'label': 0,
            'type': 'human'
        })
    
    # 保存
    df = pd.DataFrame(hard_samples)
    df = df.sample(frac=1).reset_index(drop=True)  # 打乱
    df.to_csv('datasets/hard_test/hard_test_set.csv', index=False)
    
    print(f"困难测试集: {len(df)} 条")
    print(f"  - 人类化AI: {len(df[df['type']=='humanized_ai'])}")
    print(f"  - 人类文本: {len(df[df['type']=='human'])}")

if __name__ == '__main__':
    import os
    os.makedirs('datasets/hard_test', exist_ok=True)
    main()
