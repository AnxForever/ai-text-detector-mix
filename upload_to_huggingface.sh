#!/bin/bash
# 上传数据集和模型到Hugging Face

set -e

echo "=================================="
echo "上传到Hugging Face"
echo "=================================="

# 激活环境
source .venv/bin/activate

# 检查是否已登录
if ! huggingface-cli whoami &>/dev/null; then
    echo "请先登录Hugging Face:"
    echo "huggingface-cli login"
    exit 1
fi

echo "当前用户: $(huggingface-cli whoami)"
echo ""

# 创建仓库
echo "步骤1: 创建仓库..."
huggingface-cli repo create chinese-ai-detector-bert --type model || echo "仓库已存在"
huggingface-cli repo create chinese-ai-detector-span --type model || echo "仓库已存在"
huggingface-cli repo create chinese-ai-detection-dataset --type dataset || echo "仓库已存在"
echo "✓ 仓库创建完成"
echo ""

# 上传模型
echo "步骤2: 上传BERT分类器 (390MB)..."
huggingface-cli upload $(huggingface-cli whoami)/chinese-ai-detector-bert \
    models/bert_v2_with_sep/ \
    --repo-type model
echo "✓ BERT分类器上传完成"
echo ""

echo "步骤3: 上传Span检测器 (389MB)..."
huggingface-cli upload $(huggingface-cli whoami)/chinese-ai-detector-span \
    models/bert_span_detector/ \
    --repo-type model
echo "✓ Span检测器上传完成"
echo ""

# 上传数据集
echo "步骤4: 上传数据集 (106MB)..."
huggingface-cli upload $(huggingface-cli whoami)/chinese-ai-detection-dataset \
    datasets/combined_v2/ \
    --repo-type dataset
echo "✓ 数据集上传完成"
echo ""

echo "=================================="
echo "✓ 全部上传完成！"
echo "=================================="
echo ""
echo "访问链接:"
echo "- 模型1: https://huggingface.co/$(huggingface-cli whoami)/chinese-ai-detector-bert"
echo "- 模型2: https://huggingface.co/$(huggingface-cli whoami)/chinese-ai-detector-span"
echo "- 数据集: https://huggingface.co/datasets/$(huggingface-cli whoami)/chinese-ai-detection-dataset"
