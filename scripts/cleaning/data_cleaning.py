"""
数据清洗与优化脚本
- 删除AI模板词汇
- 筛除过短文本
- 生成高质量清洗版数据集
"""
import sys
import io
import os

os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except:
        pass

import pandas as pd
import re

# AI模板词替换规则
TEMPLATE_REPLACEMENTS = {
    # 连接词替换
    "首先，": "",
    "首先": "",
    "其次，": "另外，",
    "其次": "另外",
    "最后，": "此外，",
    "最后": "此外",
    "综上所述，": "",
    "综上所述": "",
    "总的来说，": "",
    "总的来说": "",
    "总之，": "",
    "总之": "",

    # AI特征词删除
    "作为一个AI助手，": "",
    "作为一个AI，": "",
    "作为AI，": "",
    "我是一个AI": "",
    "我是一个人工智能": "",
    "作为人工智能，": "",
    "很抱歉，": "",
    "我无法": "无法",
    "我不能": "不能",

    # 模板短语删除
    "需要注意的是，": "",
    "需要注意的是": "",
    "值得注意的是，": "",
    "值得注意的是": "",
    "需要强调的是，": "",
    "需要强调的是": "",
    "重要的是，": "",
    "重要的是": "",
}

def clean_text(text):
    """清洗单个文本"""
    if pd.isna(text):
        return text

    # 替换模板词汇
    for template, replacement in TEMPLATE_REPLACEMENTS.items():
        text = text.replace(template, replacement)

    # 清理多余空格和换行
    text = re.sub(r'\n\s*\n+', '\n\n', text)  # 多个空行变两个
    text = re.sub(r' +', ' ', text)  # 多个空格变一个
    text = text.strip()

    return text

def main():
    print("=" * 70)
    print("数据清洗与优化".center(70))
    print("=" * 70)
    print()

    # 加载数据
    print("[1/5] 加载原始数据...")
    df = pd.read_csv('new_plan_datasets/parallel_dataset.csv', encoding='utf-8-sig')
    original_count = len(df)
    print(f"      原始数据: {original_count} 条")
    print()

    # 清洗文本
    print("[2/5] 清洗AI模板词汇...")
    df['text_content'] = df['text_content'].apply(clean_text)

    # 重新计算长度
    df['length'] = df['text_content'].str.len()

    # 统计模板词使用
    template_phrases = ['作为', '我是', '很抱歉', '我无法', '我不能']
    df['has_template'] = df['text_content'].apply(
        lambda x: any(phrase in str(x) for phrase in template_phrases)
    )
    template_count = df['has_template'].sum()
    print(f"      含模板词的文本: {template_count} 条 ({template_count/original_count*100:.1f}%)")
    print()

    # 筛选条件
    print("[3/5] 应用质量筛选...")

    # 条件1: 长度 >= 300字
    length_filter = df['length'] >= 300
    print(f"      长度筛选 (≥300字): {length_filter.sum()} 条保留, {(~length_filter).sum()} 条删除")

    # 条件2: 质量 >= 0.7
    quality_filter = df['generation_quality'] >= 0.7
    print(f"      质量筛选 (≥0.7): {quality_filter.sum()} 条保留, {(~quality_filter).sum()} 条删除")

    # 条件3: 不含过多AI特征（可选，暂不启用）
    # template_filter = ~df['has_template']

    # 综合筛选
    final_filter = length_filter & quality_filter
    df_cleaned = df[final_filter].copy()

    removed_count = original_count - len(df_cleaned)
    print(f"      总计删除: {removed_count} 条 ({removed_count/original_count*100:.1f}%)")
    print(f"      保留数据: {len(df_cleaned)} 条 ({len(df_cleaned)/original_count*100:.1f}%)")
    print()

    # 删除辅助列
    if 'has_template' in df_cleaned.columns:
        df_cleaned = df_cleaned.drop('has_template', axis=1)

    # 保存清洗后数据
    print("[4/5] 保存清洗后数据...")
    output_file = 'new_plan_datasets/parallel_dataset_cleaned.csv'
    df_cleaned.to_csv(output_file, index=False, encoding='utf-8-sig')

    file_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"      文件保存: {output_file}")
    print(f"      文件大小: {file_size:.1f} MB")
    print()

    # 生成清洗报告
    print("[5/5] 生成清洗报告...")

    report = {
        "原始数据": original_count,
        "清洗后数据": len(df_cleaned),
        "删除数量": removed_count,
        "保留率": f"{len(df_cleaned)/original_count*100:.1f}%",
        "平均质量": {
            "原始": float(df['generation_quality'].mean()),
            "清洗后": float(df_cleaned['generation_quality'].mean())
        },
        "平均长度": {
            "原始": float(df['length'].mean()),
            "清洗后": float(df_cleaned['length'].mean())
        },
        "长度分布": {
            "300-1000": int(len(df_cleaned[(df_cleaned['length'] >= 300) & (df_cleaned['length'] < 1000)])),
            "1000-2000": int(len(df_cleaned[(df_cleaned['length'] >= 1000) & (df_cleaned['length'] < 2000)])),
            "2000-3000": int(len(df_cleaned[(df_cleaned['length'] >= 2000) & (df_cleaned['length'] < 3000)])),
            "3000+": int(len(df_cleaned[df_cleaned['length'] >= 3000]))
        }
    }

    print("      清洗统计:")
    print(f"        原始数据: {report['原始数据']} 条")
    print(f"        清洗后: {report['清洗后数据']} 条")
    print(f"        保留率: {report['保留率']}")
    print(f"        平均质量提升: {report['平均质量']['原始']:.3f} → {report['平均质量']['清洗后']:.3f}")
    print(f"        平均长度变化: {report['平均长度']['原始']:.0f} → {report['平均长度']['清洗后']:.0f} 字符")
    print()

    # 保存报告
    import json
    report_file = 'new_plan_datasets/cleaning_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print("✅ 数据清洗完成！".center(70))
    print("=" * 70)
    print()
    print(f"清洗后数据: {output_file}")
    print(f"清洗报告: {report_file}")

if __name__ == "__main__":
    main()
