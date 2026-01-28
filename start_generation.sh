#!/bin/bash

echo "🚀 启动多样化AI文本生成任务"
echo "========================================"
echo ""
echo "📋 任务计划:"
echo "  1. 技术文档风格    2000样本  (~2小时)"
echo "  2. 学术论文风格    1500样本  (~1.5小时)"
echo "  3. 列表式内容      2000样本  (~2小时)"
echo "  4. 对抗样本        1000样本  (~1小时)"
echo "  5. 领域特定文本    1500样本  (~1.5小时)"
echo ""
echo "  总计: 8000样本"
echo "  预计耗时: 8-10小时"
echo ""
echo "📁 输出目录: datasets/logs/augmented_v2/"
echo "📝 日志文件: datasets/logs/augmented_v2/generation_log_*.txt"
echo ""
echo "⚠️  注意事项:"
echo "  - 确保反代API服务运行在 localhost:8317"
echo "  - 任务会持续数小时，建议使用 screen 或 tmux"
echo "  - 可以随时按 Ctrl+C 中断任务"
echo "  - 中间结果会自动保存"
echo ""

read -p "是否开始任务? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "任务已取消"
    exit 1
fi

cd /mnt/c/datacollection
source .venv/bin/activate

echo ""
echo "✅ 任务启动中..."
echo ""

# 使用 nohup 在后台运行
nohup python scripts/generation/generate_diverse_samples.py > datasets/logs/augmented_v2/nohup.out 2>&1 &

PID=$!
echo "✅ 任务已在后台启动"
echo "   进程ID: $PID"
echo ""
echo "📊 查看进度:"
echo "   tail -f datasets/logs/augmented_v2/generation_log_*.txt"
echo ""
echo "🛑 停止任务:"
echo "   kill $PID"
echo ""
echo "📁 查看输出:"
echo "   ls -lh datasets/logs/augmented_v2/"
echo ""
