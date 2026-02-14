#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据质量分析脚本：v6数据分布 + 训练集平衡性 + 采样策略设计

三大任务：
1. V6数据质量分析 - 分析现有5,305条数据的特征分布
2. 训练集平衡性检查 - 审视train_balanced_v2.csv是否确实平衡
3. 采样策略设计 - 制定2,000条口语AI、1,000条缺陷AI的生成方案
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import re
from datetime import datetime

class V6DataAnalyzer:
    """V6数据质量分析器"""

    def __init__(self):
        self.v6_data_dir = Path('/mnt/c/datacollection/datasets/generated/scenario_fill/2026-01-27_10h_multi_proxies')
        self.train_csv = Path('/mnt/c/datacollection/datasets/bert_v2_overnight/train_balanced_v2.csv')

    def load_v6_cleaned_data(self):
        """加载V6清洗后的所有数据文件"""
        v6_files = list(self.v6_data_dir.glob('ai_scenario_fill_20260128_123122_part*_cleaned.jsonl'))
        # 正确的排序：提取part后的数字
        v6_files = sorted(v6_files, key=lambda x: int(x.stem.split('part')[1].split('_')[0]))

        all_data = []
        for file_path in v6_files:
            print(f"加载: {file_path.name}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            all_data.append(json.loads(line))
            except Exception as e:
                print(f"  错误: {e}")

        return all_data

    def analyze_text_features(self, texts):
        """分析文本特征"""
        features = {
            'lengths': [],
            'sentence_counts': [],
            'word_counts': [],
            'has_formula': [],
            'has_markdown': [],
            'has_code': [],
            'puctuation_ratio': [],
            'english_ratio': [],
            'chinese_ratio': [],
            'formal_markers': [],  # 学术、正式词汇
            'informal_markers': [],  # 口语词汇
        }

        formal_keywords = ['因此', '鉴于', '综上', '诚然', '然而', '相应地', '相比之下', '故', '具体而言', '概而言之']
        informal_keywords = ['呃', '嗯', '然后', '真的', '特别', '特么', '咋', '啥', '简直', '也是醉了']

        for text in texts:
            # 长度统计
            features['lengths'].append(len(text))
            features['sentence_counts'].append(text.count('。') + text.count('！') + text.count('？'))
            features['word_counts'].append(len(text.split()))

            # 特殊内容检测
            features['has_formula'].append(1 if ('$' in text or '∑' in text or '∆' in text) else 0)
            features['has_markdown'].append(1 if ('#' in text or '**' in text or '```' in text) else 0)
            features['has_code'].append(1 if ('def ' in text or 'class ' in text or 'import ' in text) else 0)

            # 字符比例
            punc_count = len([c for c in text if c in '，。；：""''（）【】《》、！？'])
            features['puctuation_ratio'].append(punc_count / len(text) if len(text) > 0 else 0)

            english_count = len([c for c in text if ord(c) < 128 and c.isalpha()])
            chinese_count = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
            total = len(text)

            features['english_ratio'].append(english_count / total if total > 0 else 0)
            features['chinese_ratio'].append(chinese_count / total if total > 0 else 0)

            # 正式/口语标记
            formal_count = sum(1 for keyword in formal_keywords if keyword in text)
            informal_count = sum(1 for keyword in informal_keywords if keyword in text)

            features['formal_markers'].append(formal_count)
            features['informal_markers'].append(informal_count)

        return features

    def print_v6_analysis(self):
        """打印V6数据分析"""
        print("\n" + "="*80)
        print("📊 V6数据质量分析 (V6-ACD优先版本)")
        print("="*80)

        # 加载数据
        v6_data = self.load_v6_cleaned_data()
        print(f"\n✅ 加载完成：{len(v6_data)}条数据")

        # 基本统计
        print(f"\n【基本统计】")
        print(f"  总样本数: {len(v6_data)}")

        # 场景分布
        scenarios = Counter([d['scenario'] for d in v6_data])
        print(f"\n【场景分布】")
        for scenario, count in scenarios.most_common():
            print(f"  {scenario}: {count} ({100*count/len(v6_data):.1f}%)")

        # 模型分布
        models = Counter([d.get('model', 'unknown') for d in v6_data])
        print(f"\n【模型分布】(Top 10)")
        for model, count in models.most_common(10):
            print(f"  {model}: {count} ({100*count/len(v6_data):.1f}%)")

        # 温度分布
        temps = Counter([d.get('temperature', 'unknown') for d in v6_data])
        print(f"\n【温度(Diversity)分布】")
        for temp, count in sorted(temps.items()):
            print(f"  temperature={temp}: {count} ({100*count/len(v6_data):.1f}%)")

        # 清洗标记
        cleaned_forbidden = sum(1 for d in v6_data if d.get('cleaned_forbidden', 0))
        cleaned_residue = sum(1 for d in v6_data if d.get('cleaned_residue', 0))
        print(f"\n【清洗标记】")
        print(f"  需清理禁用词: {cleaned_forbidden} ({100*cleaned_forbidden/len(v6_data):.1f}%)")
        print(f"  需清理提示词残留: {cleaned_residue} ({100*cleaned_residue/len(v6_data):.1f}%)")

        # 文本特征分析
        texts = [d['text'] for d in v6_data]
        features = self.analyze_text_features(texts)

        print(f"\n【文本长度分布】")
        lengths = features['lengths']
        print(f"  Mean: {np.mean(lengths):.0f} | Median: {np.median(lengths):.0f} | Std: {np.std(lengths):.0f}")
        print(f"  Min: {np.min(lengths):.0f} | Max: {np.max(lengths):.0f}")
        print(f"  P25: {np.percentile(lengths, 25):.0f} | P75: {np.percentile(lengths, 75):.0f}")

        print(f"\n【文本特征】")
        print(f"  含公式(%): {100*np.mean(features['has_formula']):.1f}%")
        print(f"  含Markdown(%): {100*np.mean(features['has_markdown']):.1f}%")
        print(f"  含代码(%): {100*np.mean(features['has_code']):.1f}%")
        print(f"  英文比例: {100*np.mean(features['english_ratio']):.1f}%")
        print(f"  中文比例: {100*np.mean(features['chinese_ratio']):.1f}%")
        print(f"  正式标记(平均数/文本): {np.mean(features['formal_markers']):.2f}")
        print(f"  口语标记(平均数/文本): {np.mean(features['informal_markers']):.2f}")

        # 样本示例
        print(f"\n【样本示例】(3个典型V6数据)")
        for i in [0, len(v6_data)//2, len(v6_data)-1]:
            d = v6_data[i]
            print(f"\n  [{i}] Model: {d.get('model')}, Scenario: {d['scenario']}, Temp: {d.get('temperature')}")
            text_preview = d['text'][:150].replace('\n', ' ')
            print(f"      Text: {text_preview}...")

    def analyze_train_balanced_v2(self):
        """分析train_balanced_v2.csv的平衡性"""
        print("\n" + "="*80)
        print("📊 训练集平衡性检查 (train_balanced_v2.csv)")
        print("="*80)

        # 加载数据
        try:
            df = pd.read_csv(self.train_csv, encoding='utf-8-sig', nrows=None)
            print(f"\n✅ 加载完成：{len(df)}行数据")
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            return

        # 基本统计
        print(f"\n【基本信息】")
        print(f"  总行数: {len(df)}")
        print(f"  列数: {len(df.columns)}")
        print(f"  列名: {list(df.columns)}")

        # 标签分布
        if 'label' in df.columns:
            label_dist = df['label'].value_counts().sort_index()
            print(f"\n【标签分布】")
            for label, count in label_dist.items():
                pct = 100 * count / len(df)
                print(f"  Label {label}: {count:7d} ({pct:5.2f}%)")

            # 平衡度评估
            if len(label_dist) >= 2:
                ratio = label_dist.max() / label_dist.min()
                print(f"\n【平衡度评估】")
                print(f"  多数类/少数类比: {ratio:.3f}")
                if ratio < 1.05:
                    print(f"  📊 非常平衡 (接近1:1)")
                elif ratio < 1.1:
                    print(f"  📊 很平衡 (偏差<10%)")
                elif ratio < 1.2:
                    print(f"  ⚠️  比较平衡 (偏差10-20%)")
                else:
                    print(f"  ❌ 不平衡 (偏差>20%)")

        # 来源分布
        if 'source' in df.columns:
            source_dist = df['source'].value_counts()
            print(f"\n【来源分布】(Top 15)")
            for source, count in source_dist.head(15).items():
                print(f"  {source}: {count:7d} ({100*count/len(df):5.2f}%)")

        # 类别分布
        if 'category' in df.columns:
            cat_dist = df['category'].value_counts()
            print(f"\n【类别分布】")
            for cat, count in cat_dist.items():
                print(f"  {cat}: {count:7d} ({100*count/len(df):5.2f}%)")

        # 文本长度分布
        if 'text' in df.columns:
            df['text_length'] = df['text'].fillna('').apply(len)
            print(f"\n【文本长度分布】")
            print(f"  Mean: {df['text_length'].mean():.0f}")
            print(f"  Median: {df['text_length'].median():.0f}")
            print(f"  Std: {df['text_length'].std():.0f}")
            print(f"  Min: {df['text_length'].min():.0f}")
            print(f"  Max: {df['text_length'].max():.0f}")
            print(f"  P25: {df['text_length'].quantile(0.25):.0f}")
            print(f"  P75: {df['text_length'].quantile(0.75):.0f}")

        # 按标签检查文本特征
        print(f"\n【按标签的文本特征】")
        if 'label' in df.columns and 'text' in df.columns:
            for label in sorted(df['label'].unique()):
                subset = df[df['label'] == label]['text'].fillna('')
                avg_len = subset.apply(len).mean()
                print(f"  Label {label}: avg_length={avg_len:.0f}")

        # 数据类型检查
        print(f"\n【数据类型】")
        print(df.dtypes)

    def design_sampling_strategy(self):
        """设计采样策略"""
        print("\n" + "="*80)
        print("🎯 采样策略设计：口语/缺陷AI + 正式Human")
        print("="*80)

        print("""
【问题现状】
  ❌ 口语化/有缺陷AI被误判为Human (99%+)
  ❌ 正式Human被误判为AI (100%)
  ✅ 原因: 模型学了表面缺陷特征，而非真实生成模式

【采样目标】
  📌 目标1: 生成2,000条口语AI样本
     - 特征: 不完美、有语法错误、口语化、随意
     - 用途: 打破"缺陷=Human"的假设
     - 模型选择: GLM-4(diverse), DeepSeek(creative)

  📌 目标2: 生成1,000条缺陷AI样本
     - 特征: 重复、不连贯、逻辑混乱、不完整
     - 用途: 让模型学习AI也会"犯错"
     - 模型选择: 低温度下截断、故意减少token

  📌 目标3: 补充2,000条正式Human样本
     - 特征: 正式、流畅、学术化、完美
     - 用途: 打破"完美=AI"的假设
     - 数据源: 知乎高赞回答、学术文章、新闻评论

【详细策略】

## 1️⃣ 口语AI样本生成 (2,000条)

生成配置参数:
  - 模型: gpt-4-turbo (temperature=0.9) + GLM-4 (temperature=0.8)
  - 提示词: 添加"用口语、不正式、有错误的语气回答"
  - token限制: max_tokens=150-250 (短于正常)
  - 后处理:
    * 随机插入语法错误 (10-20%)
    * 随机删除部分句子 (5-10%)
    * 随机替换为口语词汇 (20-30%)

样本特征描述:
  | 特征 | 口语AI | 正式AI | 差异 |
  |----|----|----|-----|
  | 长度 | 120-200字 | 250-350字 | 短 |
  | 句长 | 不规则 | 规则 | 跳跃 |
  | 连接词 | 然后、其实、就是 | 因此、故而、综上 | 口语多 |
  | 语法错误 | 5-10处/篇 | 0处 | 有 |
  | 完整度 | 70-80% | 95-100% | 低 |

生成任务配置JSON:
  scenario_fill_v7_colloquial_ai_2026-01-29.json
  {
    "target_count": 2000,
    "models": ["gpt-4-turbo", "glm-4", "deepseek-v3"],
    "temperature_profile": {
      "gpt-4-turbo": 0.9,
      "glm-4": 0.8,
      "deepseek-v3": 0.85
    },
    "prompt_template": "用很口语的、不正式的、有点错误的语气回答：{topic}",
    "max_tokens": 200,
    "post_processing": {
      "inject_typos": "10-20%",
      "remove_sentences": "5-10%",
      "colloquial_replacement": "20-30%"
    }
  }

## 2️⃣ 缺陷AI样本生成 (1,000条)

生成配置参数:
  - 模型: gpt-4-turbo (temperature=0.1) + GLM-4 (temperature=0.2)
  - 策略A: 早期截断 (在40-50%位置硬停)
  - 策略B: 低质量提示词 (诱发重复/混淆)
  - 策略C: 极端参数 (top_p=0.3, frequency_penalty=2.0)

样本特征描述:
  | 特征 | 缺陷AI | 正常AI | 差异 |
  |----|----|----|-----|
  | 长度 | 80-150字 | 250-350字 | 极短 |
  | 完整性 | 50-70% | >95% | 不完整 |
  | 重复 | 20-40% | <5% | 频繁 |
  | 连贯性 | 低 | 高 | 破碎 |
  | 逻辑 | 混乱 | 清晰 | 无序 |

生成任务配置JSON:
  scenario_fill_v8_defective_ai_2026-01-29.json
  {
    "target_count": 1000,
    "models": ["gpt-4-turbo", "glm-4"],
    "generation_strategy": [
      {
        "name": "early_truncation",
        "stop_at_ratio": "0.4-0.5",
        "count": 400
      },
      {
        "name": "low_quality_prompt",
        "prompt": "重复说出关键词，生成混乱文本",
        "count": 300
      },
      {
        "name": "extreme_params",
        "top_p": 0.3,
        "frequency_penalty": 2.0,
        "count": 300
      }
    ]
  }

## 3️⃣ 正式Human样本补充 (2,000条)

数据源优先级:
  1️⃣ 知乎高赞回答 (800条)
     - 筛选: 赞数>100, 字数>300
     - 特征: 专业、严谨、有参考资料

  2️⃣ 学术文章摘要 (600条)
     - 来源: arXiv, 知网, 维普
     - 特征: 正式语言、学术术语、严密逻辑

  3️⃣ 新闻/评论 (400条)
     - 媒体: 人民日报、新华社、财经类
     - 特征: 权威、完整、流畅

  4️⃣ 官方文档 (200条)
     - 源: 政府公告、产品说明、学术指南
     - 特征: 规范、系统、完美

采集任务配置JSON:
  scenario_fill_v9_formal_human_2026-01-29.json
  {
    "target_count": 2000,
    "sources": [
      {
        "name": "zhihu_top_answers",
        "count": 800,
        "filter": "likes>100 AND length>300"
      },
      {
        "name": "academic_papers",
        "count": 600,
        "sources": ["arxiv", "cnki", "vpcs"]
      },
      {
        "name": "news_articles",
        "count": 400,
        "media": ["people.com.cn", "xinhuanet.com", "caixin.com"]
      },
      {
        "name": "official_documents",
        "count": 200,
        "types": ["government", "product", "guide"]
      }
    ]
  }

【总体执行流程】

  步骤1: 生成配置文件
    └─ scenario_fill_v7_colloquial_ai_2026-01-29.json (2,000条)
    └─ scenario_fill_v8_defective_ai_2026-01-29.json (1,000条)
    └─ scenario_fill_v9_formal_human_2026-01-29.json (2,000条)

  步骤2: 批量生成数据
    └─ python scenario_fill_generate.py --config scenario_fill_v7_*.json
    └─ python scenario_fill_generate.py --config scenario_fill_v8_*.json
    └─ 并行采集human样本

  步骤3: 数据合并与清洗
    └─ 合并到train_balanced_v3.csv (总计66,501 + 5,000 = 71,501条)
    └─ 标记数据类型 (colloquial_ai, defective_ai, formal_human)

  步骤4: 数据质量验证
    └─ 特征分布检查 (确保多样性)
    └─ 去重检查 (避免数据污染)

  步骤5: 重新训练
    └─ train_bert_v2_balanced_v3.py --epochs 5
    └─ 评估新模型的误判率改善

【预期改进效果】

  当前问题:
    - 口语AI被识为Human: 99%+ ❌
    - 正式Human被识为AI: 100% ❌
    - 总体准确率: 98.71% ✅ (虚假高分)

  改进后预期:
    - 口语AI被识为Human: <30% 📈 (大幅下降)
    - 正式Human被识为AI: <20% 📈 (大幅下降)
    - 总体准确率: 95-97% 📉 (短期下降，长期提升)
    - 真实泛化能力: 显著提升

【数据验证清单】

  ☐ 口语AI样本 (2,000)
    ☐ 样本数 >= 1,900
    ☐ 平均长度 120-200字
    ☐ 语法错误率 5-10%
    ☐ 去重率 >95%

  ☐ 缺陷AI样本 (1,000)
    ☐ 样本数 >= 900
    ☐ 平均长度 80-150字
    ☐ 完整性 50-70%
    ☐ 重复度 20-40%

  ☐ 正式Human样本 (2,000)
    ☐ 样本数 >= 1,900
    ☐ 平均长度 >300字
    ☐ 来源多样性 >=4类
    ☐ 质量评分 >=7/10

【关键成功指标 (KSI)】

  指标1: 口语AI识别率
    现状: 误判为Human 99%+
    目标: 误判为Human <30%
    验证: 在v9+human_formal集上评估

  指标2: 正式Human识别率
    现状: 误判为AI 100%
    目标: 误判为AI <20%
    验证: 在v9_formal_human集上评估

  指标3: 模型稳健性
    指标: 在多域测试集上的平均准确率
    目标: >85% (在所有域上)

  指标4: 数据多样性
    指标: 特征分布的熵
    目标: 每个类型的entropy >5.0 bits
""")

def main():
    analyzer = V6DataAnalyzer()

    # 任务1: V6数据质量分析
    analyzer.print_v6_analysis()

    # 任务2: 训练集平衡性检查
    analyzer.analyze_train_balanced_v2()

    # 任务3: 采样策略设计
    analyzer.design_sampling_strategy()

    print("\n" + "="*80)
    print("✅ 分析完成！")
    print("="*80)

if __name__ == '__main__':
    main()
