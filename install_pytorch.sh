#!/bin/bash
# PyTorch环境安装脚本
# 适配8GB显存

echo "开始安装PyTorch训练环境..."

# 安装PyTorch (CPU版本，如果有GPU会自动检测)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 安装transformers
pip3 install transformers

# 安装datasets
pip3 install datasets

# 安装其他依赖
pip3 install scikit-learn
pip3 install matplotlib
pip3 install seaborn
pip3 install jieba

echo "✓ 环境安装完成"
echo ""
echo "验证安装:"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python3 -c "import datasets; print(f'Datasets: {datasets.__version__}')"
