"""
数据质量分析脚本
分析生成的文本数据集质量
"""
import sys
import io
import os

# UTF-8 编码
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except:
        pass

import pandas as pd
import json
from collections import Counter
import re

# 常见AI模板词汇
AI_TEMPLATE_PHRASES = [
    "作为一个AI", "作为AI助手", "作为人工智能", "很抱歉", "我是一个",
    "我无法", "我不能", "需要注意的是", "值得注意的是", "首先", "其次", "最后",
    "综上所述", "总的来说", "总之", "需要强调的是", "重要的是"
]

def load_data():
    """加载数据"""
    print("=" * 70)
    print("数据质量分析报告".center(70))
    print("=" * 70)
    print()

    df = pd.read_csv('new_plan_datasets/parallel_dataset.csv', encoding='utf-8-sig')
    print(f"✅ 数据加载成功: {len(df)} 条记录")
    print()
    return df

def basic_stats(df):
    """基本统计"""
    print("=" * 70)
    print("【1】基本统计信息")
    print("=" * 70)
    print(f"总记录数: {len(df)}")
    print(f"平均质量: {df['generation_quality'].mean():.3f}")
    print(f"平均长度: {df['length'].mean():.0f} 字符")
    print(f"最短文本: {df['length'].min()} 字符")
    print(f"最长文本: {df['length'].max()} 字符")
    print(f"中位数长度: {df['length'].median():.0f} 字符")
    print()

def quality_distribution(df):
    """质量分布"""
    print("=" * 70)
    print("【2】质量分布")
    print("=" * 70)
    quality_counts = df['generation_quality'].value_counts().sort_index(ascending=False)
    for quality, count in quality_counts.items():
        percentage = count / len(df) * 100
        bar = "█" * int(percentage / 2)
        print(f"{quality:.2f}: {count:>5} 条 ({percentage:>5.1f}%) {bar}")

    # 质量等级分类
    excellent = len(df[df['generation_quality'] >= 0.9])
    good = len(df[(df['generation_quality'] >= 0.7) & (df['generation_quality'] < 0.9)])
    fair = len(df[df['generation_quality'] < 0.7])
    print()
    print(f"优秀 (≥0.9): {excellent} 条 ({excellent/len(df)*100:.1f}%)")
    print(f"良好 (0.7-0.9): {good} 条 ({good/len(df)*100:.1f}%)")
    print(f"一般 (<0.7): {fair} 条 ({fair/len(df)*100:.1f}%)")
    print()

def length_analysis(df):
    """长度分析"""
    print("=" * 70)
    print("【3】长度分布分析")
    print("=" * 70)

    length_ranges = [
        (0, 500, "超短"),
        (500, 1000, "短"),
        (1000, 2000, "中等"),
        (2000, 3000, "长"),
        (3000, 10000, "超长")
    ]

    for min_len, max_len, label in length_ranges:
        count = len(df[(df['length'] >= min_len) & (df['length'] < max_len)])
        percentage = count / len(df) * 100
        bar = "█" * int(percentage / 2)
        print(f"{label:>4} ({min_len:>4}-{max_len:<4}): {count:>5} 条 ({percentage:>5.1f}%) {bar}")
    print()

def attribute_distribution(df):
    """属性分布"""
    print("=" * 70)
    print("【4】六维属性分布")
    print("=" * 70)

    dimensions = ['attribute', 'genre', 'role', 'style', 'constraint']

    for dim in dimensions:
        if dim in df.columns:
            print(f"\n{dim.upper()}:")
            counts = df[dim].value_counts().head(10)
            for val, count in counts.items():
                percentage = count / len(df) * 100
                print(f"  {val[:20]:<20}: {count:>4} ({percentage:>5.1f}%)")

def ai_template_detection(df):
    """AI模板检测"""
    print("\n" + "=" * 70)
    print("【5】AI 模板词汇检测")
    print("=" * 70)

    template_counts = {phrase: 0 for phrase in AI_TEMPLATE_PHRASES}

    for text in df['text_content']:
        for phrase in AI_TEMPLATE_PHRASES:
            if phrase in text:
                template_counts[phrase] += 1

    # 按出现次数排序
    sorted_templates = sorted(template_counts.items(), key=lambda x: x[1], reverse=True)

    total_detected = sum(1 for _, count in sorted_templates if count > 0)
    total_occurrences = sum(count for _, count in sorted_templates)

    print(f"检测到 {total_detected} 种模板词汇，共出现 {total_occurrences} 次")
    print(f"受影响文本: {len(df[df['text_content'].str.contains('|'.join(AI_TEMPLATE_PHRASES), regex=True)])} 条 "
          f"({len(df[df['text_content'].str.contains('|'.join(AI_TEMPLATE_PHRASES), regex=True)])/len(df)*100:.1f}%)")
    print()

    print("高频模板词汇 (前10):")
    for phrase, count in sorted_templates[:10]:
        if count > 0:
            print(f"  '{phrase}': {count} 次")
    print()

def show_samples(df):
    """展示样本"""
    print("=" * 70)
    print("【6】数据样本展示")
    print("=" * 70)

    # 随机抽取5个样本
    samples = df.sample(n=min(5, len(df)))

    for idx, row in samples.iterrows():
        print(f"\n样本 {idx}:")
        print(f"属性: {row['attribute']} | 质量: {row['generation_quality']:.2f} | 长度: {row['length']} 字符")
        print(f"主题: {row['topic'][:50]}...")
        print(f"内容预览:")
        preview = row['text_content'][:200] + "..." if len(row['text_content']) > 200 else row['text_content']
        print(f"  {preview}")
        print("-" * 70)

def api_performance(df):
    """API性能分析"""
    print("\n" + "=" * 70)
    print("【7】API 性能分析")
    print("=" * 70)

    api_stats = df.groupby('source_api').agg({
        'generation_quality': ['mean', 'count'],
        'length': 'mean'
    }).round(2)

    api_stats.columns = ['平均质量', '生成数量', '平均长度']
    api_stats = api_stats.sort_values('生成数量', ascending=False)

    print(api_stats.to_string())
    print()

def save_report(df):
    """保存分析报告"""
    report_file = 'new_plan_datasets/quality_analysis_report.json'

    report = {
        "总记录数": len(df),
        "平均质量": float(df['generation_quality'].mean()),
        "平均长度": float(df['length'].mean()),
        "质量分布": df['generation_quality'].value_counts().to_dict(),
        "属性分布": df['attribute'].value_counts().to_dict(),
        "API使用": df['source_api'].value_counts().to_dict(),
        "长度统计": {
            "最小": int(df['length'].min()),
            "最大": int(df['length'].max()),
            "中位数": float(df['length'].median())
        }
    }

    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"✅ 分析报告已保存: {report_file}")

def main():
    df = load_data()
    basic_stats(df)
    quality_distribution(df)
    length_analysis(df)
    attribute_distribution(df)
    ai_template_detection(df)
    api_performance(df)
    show_samples(df)
    save_report(df)

    print("\n" + "=" * 70)
    print("分析完成！".center(70))
    print("=" * 70)

if __name__ == "__main__":
    main()
