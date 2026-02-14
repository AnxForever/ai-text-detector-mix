#!/bin/bash
# 稳妥过夜训练 - 使用简单版脚本

cd /mnt/c/datacollection
export PYTHONPATH="/mnt/c/datacollection:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

LOG_FILE="logs/training/train_overnight_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs/training

echo "========================================"
echo "稳妥过夜训练 - $(date)"
echo "========================================"

/mnt/c/datacollection/.venv/bin/python3 \
    scripts/training/train_overnight_simple.py \
    2>&1 | tee "$LOG_FILE"

echo "========================================"
echo "完成: $(date)"
echo "========================================"
