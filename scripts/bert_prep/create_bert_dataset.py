"""
BERT格式转换和PyTorch Dataset创建
用于AI文本检测的二分类任务
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
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from datetime import datetime
import json

class AIDetectionDataset(Dataset):
    """AI文本检测数据集（PyTorch格式）- 支持动态Padding"""

    def __init__(self, texts, labels, tokenizer, max_length=512, use_dynamic_padding=True):
        """
        Args:
            texts: 文本列表
            labels: 标签列表 (0=人类, 1=AI)
            tokenizer: BERT tokenizer
            max_length: 最大序列长度
            use_dynamic_padding: 是否使用动态padding（推荐True以减少长度偏差）
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_dynamic_padding = use_dynamic_padding

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # BERT tokenization
        if self.use_dynamic_padding:
            # 动态padding：不在这里padding，在batch级别padding
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding=False,  # 不padding，由collate_fn处理
                truncation=True,  # 截断超长文本
                return_tensors='pt'  # 返回PyTorch张量
            )

            return {
                'input_ids': encoding['input_ids'].flatten(),  # 变长
                'attention_mask': encoding['attention_mask'].flatten(),  # 变长
                'token_type_ids': encoding['token_type_ids'].flatten(),  # 变长
                'labels': torch.tensor(label, dtype=torch.long)  # 标量
            }
        else:
            # 固定padding：填充到max_length（旧方法）
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',  # 填充到max_length
                truncation=True,  # 截断超长文本
                return_tensors='pt'  # 返回PyTorch张量
            )

            return {
                'input_ids': encoding['input_ids'].flatten(),  # [max_length]
                'attention_mask': encoding['attention_mask'].flatten(),  # [max_length]
                'token_type_ids': encoding['token_type_ids'].flatten(),  # [max_length]
                'labels': torch.tensor(label, dtype=torch.long)  # 标量
            }


def dynamic_collate_fn(batch):
    """
    动态padding collate函数
    在batch级别padding到batch内最长序列，而不是固定的max_length

    优势：
    1. 减少padding token数量，提高训练效率
    2. 避免模型过度依赖padding模式
    3. 减少长度信号泄露

    Args:
        batch: 数据样本列表

    Returns:
        批次数据字典
    """
    # 找到batch中最长的序列长度
    max_len = max(len(item['input_ids']) for item in batch)

    # 初始化批次数据
    padded_batch = {
        'input_ids': [],
        'attention_mask': [],
        'token_type_ids': [],
        'labels': []
    }

    # 对每个样本进行padding
    for item in batch:
        current_len = len(item['input_ids'])
        pad_len = max_len - current_len

        # Padding到batch最大长度
        padded_batch['input_ids'].append(
            torch.cat([item['input_ids'], torch.zeros(pad_len, dtype=torch.long)])
        )
        padded_batch['attention_mask'].append(
            torch.cat([item['attention_mask'], torch.zeros(pad_len, dtype=torch.long)])
        )
        padded_batch['token_type_ids'].append(
            torch.cat([item['token_type_ids'], torch.zeros(pad_len, dtype=torch.long)])
        )
        padded_batch['labels'].append(item['labels'])

    # 堆叠成tensor
    return {
        'input_ids': torch.stack(padded_batch['input_ids']),
        'attention_mask': torch.stack(padded_batch['attention_mask']),
        'token_type_ids': torch.stack(padded_batch['token_type_ids']),
        'labels': torch.stack(padded_batch['labels'])
    }


def create_bert_datasets(max_length=512, batch_size=16, use_dynamic_padding=True):
    """创建BERT数据集和DataLoader"""

    print("=" * 60, flush=True)
    print("BERT数据集和DataLoader创建程序", flush=True)
    print("=" * 60, flush=True)
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    # 1. 加载BERT tokenizer
    print("\n[1/5] 加载BERT tokenizer...", flush=True)
    print("  模型: bert-base-chinese", flush=True)

    try:
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        print(f"  ✓ Tokenizer加载成功", flush=True)
        print(f"  词汇表大小: {tokenizer.vocab_size:,}", flush=True)
    except Exception as e:
        print(f"  ✗ 加载失败: {str(e)}", flush=True)
        print("  请先下载模型或检查网络连接", flush=True)
        return None

    # 2. 加载数据集
    print("\n[2/5] 加载数据集...", flush=True)

    train_df = pd.read_csv('datasets/bert/train.csv', encoding='utf-8-sig')
    val_df = pd.read_csv('datasets/bert/val.csv', encoding='utf-8-sig')
    test_df = pd.read_csv('datasets/bert/test.csv', encoding='utf-8-sig')

    print(f"  ✓ 训练集: {len(train_df):,} 条", flush=True)
    print(f"  ✓ 验证集: {len(val_df):,} 条", flush=True)
    print(f"  ✓ 测试集: {len(test_df):,} 条", flush=True)

    # 3. 创建PyTorch Dataset
    print("\n[3/5] 创建PyTorch Dataset...", flush=True)
    print(f"  参数: max_length={max_length}", flush=True)
    print(f"  动态Padding: {'启用' if use_dynamic_padding else '禁用'}", flush=True)

    if use_dynamic_padding:
        print("  ✓ 使用动态Padding策略（batch级别自适应）", flush=True)
        print("    - 减少padding token数量", flush=True)
        print("    - 避免模型过度依赖padding模式", flush=True)
        print("    - 减少长度信号泄露", flush=True)

    train_dataset = AIDetectionDataset(
        train_df['text'].tolist(),
        train_df['label'].tolist(),
        tokenizer,
        max_length,
        use_dynamic_padding
    )

    val_dataset = AIDetectionDataset(
        val_df['text'].tolist(),
        val_df['label'].tolist(),
        tokenizer,
        max_length,
        use_dynamic_padding
    )

    test_dataset = AIDetectionDataset(
        test_df['text'].tolist(),
        test_df['label'].tolist(),
        tokenizer,
        max_length,
        use_dynamic_padding
    )

    print(f"  ✓ 训练Dataset: {len(train_dataset)} 样本", flush=True)
    print(f"  ✓ 验证Dataset: {len(val_dataset)} 样本", flush=True)
    print(f"  ✓ 测试Dataset: {len(test_dataset)} 样本", flush=True)

    # 4. 创建DataLoader
    print("\n[4/5] 创建DataLoader...", flush=True)
    print(f"  批次大小: {batch_size}", flush=True)

    # 选择collate函数
    collate_fn = dynamic_collate_fn if use_dynamic_padding else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 训练集打乱
        num_workers=0,  # Windows环境使用0
        collate_fn=collate_fn  # 动态padding
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # 验证集不打乱
        num_workers=0,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # 测试集不打乱
        num_workers=0,
        collate_fn=collate_fn
    )

    print(f"  ✓ 训练DataLoader: {len(train_loader)} batches", flush=True)
    print(f"  ✓ 验证DataLoader: {len(val_loader)} batches", flush=True)
    print(f"  ✓ 测试DataLoader: {len(test_loader)} batches", flush=True)

    # 5. 测试数据加载
    print("\n[5/5] 测试数据加载...", flush=True)

    try:
        # 获取一个batch测试
        batch = next(iter(train_loader))

        print(f"  ✓ 测试batch加载成功", flush=True)
        print(f"    input_ids shape: {batch['input_ids'].shape}", flush=True)
        print(f"    attention_mask shape: {batch['attention_mask'].shape}", flush=True)
        print(f"    token_type_ids shape: {batch['token_type_ids'].shape}", flush=True)
        print(f"    labels shape: {batch['labels'].shape}", flush=True)

        # 显示一个样例
        print(f"\n  样例数据（batch中的第一个样本）:", flush=True)
        print(f"    文本长度（token）: {batch['attention_mask'][0].sum().item()}", flush=True)
        print(f"    标签: {batch['labels'][0].item()} ({'AI' if batch['labels'][0].item()==1 else '人类'})", flush=True)

    except Exception as e:
        print(f"  ✗ 测试失败: {str(e)}", flush=True)
        return None

    # 保存配置信息
    config = {
        'created_at': datetime.now().isoformat(),
        'tokenizer': 'bert-base-chinese',
        'max_length': max_length,
        'batch_size': batch_size,
        'use_dynamic_padding': use_dynamic_padding,
        'vocab_size': tokenizer.vocab_size,
        'datasets': {
            'train': len(train_dataset),
            'val': len(val_dataset),
            'test': len(test_dataset)
        },
        'dataloaders': {
            'train_batches': len(train_loader),
            'val_batches': len(val_loader),
            'test_batches': len(test_loader)
        }
    }

    config_file = 'datasets/bert/dataset_config.json'
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 配置信息已保存: {config_file}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("BERT数据集准备完成！", flush=True)
    print("=" * 60, flush=True)

    print("\n训练准备就绪！", flush=True)
    print("\n下一步 - 模型训练:", flush=True)
    print("  1. 加载BERT模型 (BertForSequenceClassification)", flush=True)
    print("  2. 配置优化器和学习率调度器", flush=True)
    print("  3. 定义训练循环", flush=True)
    print("  4. 开始训练和验证", flush=True)
    print("  5. 在测试集上评估最终性能", flush=True)

    print("\n推荐训练参数:", flush=True)
    print("  - 学习率: 2e-5", flush=True)
    print("  - Epochs: 3-5", flush=True)
    print("  - Warmup steps: 500", flush=True)
    print("  - Weight decay: 0.01", flush=True)

    return {
        'tokenizer': tokenizer,
        'datasets': {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        },
        'dataloaders': {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }
    }


def analyze_text_lengths(tokenizer, max_length=512):
    """分析文本长度分布，帮助选择合适的max_length"""

    print("\n" + "=" * 60, flush=True)
    print("文本长度分析", flush=True)
    print("=" * 60, flush=True)

    # 加载数据
    train_df = pd.read_csv('datasets/bert/train.csv', encoding='utf-8-sig')

    print(f"分析样本数: {len(train_df)}", flush=True)

    # 随机采样1000条进行分析
    sample_df = train_df.sample(min(1000, len(train_df)), random_state=42)

    token_lengths = []
    for text in sample_df['text']:
        tokens = tokenizer.encode(text, truncation=False)
        token_lengths.append(len(tokens))

    token_lengths_series = pd.Series(token_lengths)

    print(f"\nToken长度统计:", flush=True)
    print(f"  平均值: {token_lengths_series.mean():.0f}", flush=True)
    print(f"  中位数: {token_lengths_series.median():.0f}", flush=True)
    print(f"  最小值: {token_lengths_series.min()}", flush=True)
    print(f"  最大值: {token_lengths_series.max()}", flush=True)
    print(f"  25分位: {token_lengths_series.quantile(0.25):.0f}", flush=True)
    print(f"  75分位: {token_lengths_series.quantile(0.75):.0f}", flush=True)
    print(f"  95分位: {token_lengths_series.quantile(0.95):.0f}", flush=True)
    print(f"  99分位: {token_lengths_series.quantile(0.99):.0f}", flush=True)

    # 覆盖率分析
    print(f"\n不同max_length的覆盖率:", flush=True)
    for length in [128, 256, 384, 512, 768, 1024]:
        coverage = (token_lengths_series <= length).sum() / len(token_lengths_series) * 100
        print(f"  max_length={length}: {coverage:.1f}% 的文本完全保留", flush=True)

    print("\n推荐:")
    if token_lengths_series.quantile(0.95) <= 512:
        print("  ✓ max_length=512 可以覆盖95%以上的文本", flush=True)
    elif token_lengths_series.quantile(0.95) <= 768:
        print("  建议使用 max_length=768", flush=True)
    else:
        print("  建议使用 max_length=1024 或考虑文本截断策略", flush=True)


if __name__ == "__main__":
    # 先进行文本长度分析
    print("=" * 60, flush=True)
    print("步骤1: 文本长度分析", flush=True)
    print("=" * 60, flush=True)

    try:
        tokenizer_temp = BertTokenizer.from_pretrained('bert-base-chinese')
        analyze_text_lengths(tokenizer_temp, max_length=512)
    except Exception as e:
        print(f"长度分析失败: {str(e)}", flush=True)
        print("跳过分析，继续创建数据集...", flush=True)

    # 创建BERT数据集
    print("\n" + "=" * 60, flush=True)
    print("步骤2: 创建BERT数据集", flush=True)
    print("=" * 60, flush=True)

    # 启用动态padding以减少长度偏差
    result = create_bert_datasets(max_length=512, batch_size=16, use_dynamic_padding=True)

    if result:
        print("\n✅ 所有准备工作完成！数据集已就绪，可以开始训练BERT模型。", flush=True)
    else:
        print("\n❌ 数据集创建失败，请检查错误信息。", flush=True)
