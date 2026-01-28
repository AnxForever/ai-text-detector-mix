#!/bin/bash

echo "🚀 启动API检测服务（使用远程模型）"
echo "========================================"

cd /mnt/c/datacollection

# 激活虚拟环境
source .venv/bin/activate

# 检查依赖
pip list | grep -q "requests" || pip install requests

echo ""
echo "✅ 配置信息:"
echo "   API: https://api.hotaruapi.top/v1"
echo "   模型: deepseek-ai/deepseek-v3.1"
echo "   端口: http://localhost:8000"
echo ""
echo "🔧 启动服务..."
echo ""

# 启动API服务
python api.py

