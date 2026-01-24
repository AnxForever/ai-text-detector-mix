#!/usr/bin/env python3
"""
继续训练脚本 - 从已有模型继续训练
支持长度平衡数据集
"""

import sys
import os
import argparse
from pathlib import Path

def continue_training():
    """继续训练配置"""
    
    parser = argparse.ArgumentParser(description='继续训练BERT模型')
    parser.add_argument('--base_model', type=str, 
                       default='models/bert_improved/best_model',
                       help='基础模型路径')
    parser.add_argument('--output_dir', type=str,
                       default='models/bert_improved_v2',
                       help='新模型输出目录')
    parser.add_argument('--train_data', type=str,
                       default='datasets/bert/train.csv',
                       help='训练数据')
    parser.add_argument('--val_data', type=str,
                       default='datasets/bert/val.csv',
                       help='验证数据')
    parser.add_argument('--test_data', type=str,
                       default='datasets/bert/test.csv',
                       help='测试数据')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='批次大小（8GB显存建议用8）')
    parser.add_argument('--epochs', type=int, default=3,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='学习率（继续训练建议更小）')
    
    args = parser.parse_args()
    
    # 检查基础模型
    if not Path(args.base_model).exists():
        print(f"✗ 基础模型不存在: {args.base_model}")
        print("  请先训练基础模型或指定正确路径")
        return
    
    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("继续训练配置")
    print("="*60)
    print(f"基础模型: {args.base_model}")
    print(f"输出目录: {args.output_dir}")
    print(f"训练数据: {args.train_data}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"训练轮数: {args.epochs}")
    print("="*60)
    
    # 调用训练脚本
    cmd = f"""python3 scripts/training/train_bert_improved.py \
        --train_csv {args.train_data} \
        --val_csv {args.val_data} \
        --test_csv {args.test_data} \
        --model_dir {args.output_dir} \
        --batch_size {args.batch_size} \
        --epochs {args.epochs} \
        --lr {args.lr} \
        --load_from {args.base_model}"""
    
    print(f"\n执行命令:")
    print(cmd)
    print()
    
    os.system(cmd)


if __name__ == "__main__":
    continue_training()
