#!/usr/bin/env python3
"""
完整的数据处理和训练流程
解决长度偏差问题
"""

import subprocess
import sys
from pathlib import Path

def run_step(step_name, command):
    """运行单个步骤"""
    print("\n" + "="*60)
    print(f"步骤: {step_name}")
    print("="*60)
    print(f"命令: {command}\n")
    
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"\n✗ 步骤失败: {step_name}")
        sys.exit(1)
    print(f"\n✓ 步骤完成: {step_name}")


def main():
    print("="*60)
    print("完整训练流程 - 解决长度偏差")
    print("="*60)
    
    # 步骤1: 长度平衡
    run_step(
        "1. 长度分布平衡",
        "python scripts/data_cleaning/balance_length_distribution.py"
    )
    
    # 步骤2: 重新划分数据集
    run_step(
        "2. 划分训练/验证/测试集",
        """python scripts/bert_prep/split_dataset.py \
            --input datasets/bert/full_dataset_length_balanced.csv \
            --output_dir datasets/bert_balanced"""
    )
    
    # 步骤3: 格式去偏（如果需要）
    print("\n" + "="*60)
    print("步骤: 3. 格式去偏（可选）")
    print("="*60)
    choice = input("是否需要格式去偏？(y/n): ").strip().lower()
    
    if choice == 'y':
        run_step(
            "3. 格式去偏",
            """python scripts/data_cleaning/remove_format_bias.py \
                --input datasets/bert_balanced/train.csv \
                --output datasets/bert_balanced_debiased"""
        )
        train_dir = "datasets/bert_balanced_debiased"
    else:
        train_dir = "datasets/bert_balanced"
    
    # 步骤4: 训练模型
    print("\n" + "="*60)
    print("步骤: 4. 训练模型")
    print("="*60)
    print("选项:")
    print("  1. 从头训练")
    print("  2. 从已有模型继续训练")
    
    choice = input("请选择 (1/2): ").strip()
    
    if choice == '2':
        base_model = input("基础模型路径 [models/bert_improved/best_model]: ").strip()
        if not base_model:
            base_model = "models/bert_improved/best_model"
        
        output_dir = input("输出目录 [models/bert_improved_v2]: ").strip()
        if not output_dir:
            output_dir = "models/bert_improved_v2"
        
        run_step(
            "4. 继续训练",
            f"""python scripts/training/continue_training.py \
                --base_model {base_model} \
                --output_dir {output_dir} \
                --train_data {train_dir}/train.csv \
                --val_data {train_dir}/val.csv \
                --test_data {train_dir}/test.csv \
                --batch_size 8 \
                --epochs 3 \
                --lr 1e-5"""
        )
    else:
        run_step(
            "4. 从头训练",
            f"""python scripts/training/train_bert_improved.py \
                --train_csv {train_dir}/train.csv \
                --val_csv {train_dir}/val.csv \
                --test_csv {train_dir}/test.csv \
                --model_dir models/bert_length_balanced \
                --batch_size 8 \
                --epochs 5"""
        )
    
    print("\n" + "="*60)
    print("✓ 所有步骤完成！")
    print("="*60)
    print("\n下一步:")
    print("  1. 检查模型性能")
    print("  2. 运行长度感知评估")
    print("  3. 对比长度平衡前后的效果")


if __name__ == "__main__":
    main()
