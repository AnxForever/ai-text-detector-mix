#!/usr/bin/env python3
"""
统计特征提取模块
提取困惑度、词汇丰富度、词性分布等特征
"""

import numpy as np
import pandas as pd
import jieba
import jieba.posseg as pseg
from typing import List, Dict
from collections import Counter

class StatisticalFeatureExtractor:
    def __init__(self):
        self.feature_names = [
            'perplexity', 'ttr', 'noun_ratio', 'verb_ratio', 
            'adj_ratio', 'adv_ratio', 'avg_word_len', 
            'unique_word_ratio', 'stopword_ratio', 'punct_ratio'
        ]
    
    def extract_features(self, text: str) -> np.ndarray:
        """提取单个文本的统计特征"""
        features = []
        
        # 分词
        words = list(jieba.cut(text))
        words_clean = [w for w in words if w.strip()]
        
        # 1. 困惑度（简化版：使用字符级n-gram熵）
        ppl = self._compute_perplexity(text)
        features.append(ppl)
        
        # 2. 词汇丰富度 (TTR)
        ttr = len(set(words_clean)) / len(words_clean) if words_clean else 0
        features.append(ttr)
        
        # 3-6. 词性分布
        pos_counts = self._get_pos_distribution(text)
        total_words = sum(pos_counts.values()) or 1
        features.append(pos_counts.get('n', 0) / total_words)  # 名词
        features.append(pos_counts.get('v', 0) / total_words)  # 动词
        features.append(pos_counts.get('a', 0) / total_words)  # 形容词
        features.append(pos_counts.get('d', 0) / total_words)  # 副词
        
        # 7. 平均词长
        avg_word_len = np.mean([len(w) for w in words_clean]) if words_clean else 0
        features.append(avg_word_len)
        
        # 8. 独特词比例
        word_counts = Counter(words_clean)
        unique_ratio = sum(1 for c in word_counts.values() if c == 1) / len(words_clean) if words_clean else 0
        features.append(unique_ratio)
        
        # 9. 停用词比例
        stopwords = self._load_stopwords()
        stopword_ratio = sum(1 for w in words_clean if w in stopwords) / len(words_clean) if words_clean else 0
        features.append(stopword_ratio)
        
        # 10. 标点符号比例
        punct_count = sum(1 for c in text if c in '，。！？；：、""''（）【】《》')
        punct_ratio = punct_count / len(text) if text else 0
        features.append(punct_ratio)
        
        return np.array(features, dtype=np.float32)
    
    def extract_batch(self, texts: List[str]) -> np.ndarray:
        """批量提取特征"""
        features = [self.extract_features(text) for text in texts]
        return np.array(features)
    
    def _compute_perplexity(self, text: str) -> float:
        """计算简化困惑度（字符级2-gram熵）"""
        if len(text) < 2:
            return 0.0
        
        bigrams = [text[i:i+2] for i in range(len(text)-1)]
        counts = Counter(bigrams)
        total = len(bigrams)
        
        entropy = -sum((c/total) * np.log2(c/total) for c in counts.values())
        perplexity = 2 ** entropy
        
        return perplexity
    
    def _get_pos_distribution(self, text: str) -> Dict[str, int]:
        """获取词性分布"""
        words = pseg.cut(text)
        pos_counts = Counter(flag for word, flag in words)
        return pos_counts
    
    def _load_stopwords(self) -> set:
        """加载停用词表"""
        # 简化版停用词
        return {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', 
                '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去'}
    
    def save_features(self, texts: List[str], labels: List[int], output_file: str):
        """提取并保存特征"""
        print(f"提取 {len(texts)} 条文本的统计特征...")
        
        features = self.extract_batch(texts)
        
        df = pd.DataFrame(features, columns=self.feature_names)
        df['label'] = labels
        df['text'] = texts
        
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"✓ 特征已保存到: {output_file}")
        
        # 统计信息
        print(f"\n特征统计:")
        print(df[self.feature_names].describe())
        
        return df


def main():
    """测试统计特征提取"""
    extractor = StatisticalFeatureExtractor()
    
    # 测试文本
    test_texts = [
        "人工智能技术在近年来取得了突破性进展。深度学习算法的发展使得计算机视觉和自然语言处理能力大幅提升。",
        "今天天气很好，我去公园散步了。看到很多人在锻炼身体，感觉很开心。"
    ]
    
    for i, text in enumerate(test_texts):
        features = extractor.extract_features(text)
        print(f"\n文本 {i+1}:")
        print(f"  {text[:50]}...")
        print(f"  特征: {features}")
        for name, val in zip(extractor.feature_names, features):
            print(f"    {name}: {val:.4f}")


if __name__ == "__main__":
    main()
