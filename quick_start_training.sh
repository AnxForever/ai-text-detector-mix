#!/bin/bash
# 快速启动训练 - 基于Codex建议的最小可行路线

echo "=========================================="
echo "AI文本检测 - 长度平衡模型训练"
echo "=========================================="
echo ""

# 检查环境
echo "1. 检查环境..."
if ! python3 -c "import torch" 2>/dev/null; then
    echo "✗ PyTorch未安装"
    echo "请先运行: bash install_pytorch.sh"
    exit 1
fi
echo "✓ PyTorch已安装"

if ! python3 -c "import transformers" 2>/dev/null; then
    echo "✗ Transformers未安装"
    echo "请先运行: bash install_pytorch.sh"
    exit 1
fi
echo "✓ Transformers已安装"

# 检查数据
echo ""
echo "2. 检查数据集..."
if [ ! -f "datasets/bert_v2_balanced/train.csv" ]; then
    echo "✗ 训练数据不存在"
    exit 1
fi
echo "✓ 数据集就绪"
echo "  训练集: datasets/bert_v2_balanced/train.csv"
echo "  验证集: datasets/bert_v2_balanced/val.csv"
echo "  测试集: datasets/bert_v2_balanced/test.csv"

# 检查基础模型
echo ""
echo "3. 检查基础模型..."
if [ ! -d "models/bert_improved/best_model" ]; then
    echo "⚠️  基础模型不存在，将从头训练"
    BASE_MODEL_ARG=""
else
    echo "✓ 基础模型存在"
    BASE_MODEL_ARG="--load_from models/bert_improved/best_model"
fi

# 开始训练
echo ""
echo "=========================================="
echo "开始训练长度平衡模型"
echo "=========================================="
echo ""
echo "配置:"
echo "  输出目录: models/bert_length_balanced"
echo "  批次大小: 8 (适配8GB显存)"
echo "  训练轮数: 3"
echo "  学习率: 1e-5"
echo ""
echo "预计时间: 2-4小时"
echo ""

read -p "按Enter开始训练，或Ctrl+C取消..."

python3 scripts/training/train_bert_improved.py \
    --train_csv datasets/bert_v2_balanced/train.csv \
    --val_csv datasets/bert_v2_balanced/val.csv \
    --test_csv datasets/bert_v2_balanced/test.csv \
    --model_dir models/bert_length_balanced \
    --batch_size 8 \
    --epochs 3 \
    --lr 1e-5 \
    $BASE_MODEL_ARG

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo ""
echo "下一步:"
echo "  1. 查看结果: cat models/bert_length_balanced/training_history.json"
echo "  2. 长度分段评估: python3 scripts/evaluation/length_aware_evaluation.py"
echo "  3. 查看任务清单: cat WEEKLY_TASKS.md"
