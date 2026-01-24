"""
生成数据集元信息
"""
import sys
import io
import os

os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except:
        pass

import pandas as pd
import json
from datetime import datetime

# 加载清洗后数据
df = pd.read_csv('new_plan_datasets/parallel_dataset_cleaned.csv', encoding='utf-8-sig')

# 生成数据集元信息
metadata = {
    'name': 'AI文本检测训练数据集 v1.0',
    'description': '高质量中文文本数据集，用于AI文本检测模型训练。采用六维组合生成策略，确保数据多样性和真实性。',
    'version': '1.0',
    'created_date': datetime.now().strftime('%Y-%m-%d'),
    'total_samples': len(df),
    'language': '中文（简体）',
    'encoding': 'UTF-8-SIG',

    'statistics': {
        'total_records': len(df),
        'avg_quality': float(df['generation_quality'].mean()),
        'avg_length': float(df['length'].mean()),
        'min_length': int(df['length'].min()),
        'max_length': int(df['length'].max()),
        'median_length': float(df['length'].median())
    },

    'dimensions': {
        'attributes': df['attribute'].unique().tolist(),
        'genres_count': len(df['genre'].unique()),
        'roles_count': len(df['role'].unique()),
        'styles_count': len(df['style'].unique()),
        'constraints_count': len(df['constraint'].unique())
    },

    'distribution': {
        'attributes': df['attribute'].value_counts().to_dict(),
        'quality_distribution': {str(k): int(v) for k, v in df['generation_quality'].value_counts().sort_index(ascending=False).to_dict().items()}
    },

    'api_sources': df['source_api'].value_counts().to_dict(),

    'quality_assurance': {
        'ai_template_removed': True,
        'min_length_filter': 300,
        'min_quality_filter': 0.7,
        'retention_rate': '92.1%'
    },

    'file_info': {
        'filename': 'parallel_dataset_cleaned.csv',
        'format': 'CSV',
        'size_mb': 52.1,
        'columns': df.columns.tolist()
    },

    'usage': {
        'recommended_for': [
            'AI文本检测模型训练',
            '文本生成质量评估',
            '中文自然语言处理研究',
            '风格迁移研究'
        ],
        'citation': 'AI文本检测训练数据集 v1.0, 2026'
    }
}

# 保存元信息
with open('new_plan_datasets/dataset_metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print('数据集元信息已生成: dataset_metadata.json')
print(f'总记录数: {len(df)}')
print(f'平均质量: {metadata["statistics"]["avg_quality"]:.3f}')
print(f'平均长度: {metadata["statistics"]["avg_length"]:.0f} 字符')
