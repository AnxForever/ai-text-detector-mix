#!/bin/bash

# AI续写和润色功能测试脚本

echo "🚀 启动AI文本检测系统（含续写和润色功能）"
echo "================================================"

# 检查Python环境
if [ ! -d ".venv" ]; then
    echo "❌ 虚拟环境不存在，请先运行: python -m venv .venv"
    exit 1
fi

# 检查Node环境
if ! command -v npm &> /dev/null; then
    echo "❌ npm未安装，请先安装Node.js"
    exit 1
fi

echo ""
echo "📦 检查依赖..."

# 检查Python依赖
source .venv/bin/activate
pip list | grep -q "fastapi" || pip install fastapi uvicorn requests

# 检查前端依赖
cd frontend
if [ ! -d "node_modules" ]; then
    echo "📥 安装前端依赖..."
    npm install
fi
cd ..

echo ""
echo "✅ 依赖检查完成"
echo ""
echo "🔧 启动服务..."
echo ""

# 启动后端（后台运行）
echo "1️⃣ 启动后端API服务 (http://localhost:8000)"
source .venv/bin/activate
python api.py > logs/api.log 2>&1 &
BACKEND_PID=$!
echo "   后端PID: $BACKEND_PID"

# 等待后端启动
sleep 3

# 检查后端是否启动成功
if curl -s http://localhost:8000/docs > /dev/null; then
    echo "   ✅ 后端启动成功"
else
    echo "   ❌ 后端启动失败，查看 logs/api.log"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

echo ""
echo "2️⃣ 启动前端服务 (http://localhost:3000)"
cd frontend
npm run dev > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
echo "   前端PID: $FRONTEND_PID"
cd ..

echo ""
echo "================================================"
echo "✨ 服务启动完成！"
echo ""
echo "📍 访问地址:"
echo "   前端: http://localhost:3000"
echo "   后端: http://localhost:8000"
echo "   API文档: http://localhost:8000/docs"
echo ""
echo "🎯 功能测试:"
echo "   1. 访问 http://localhost:3000/demo"
echo "   2. 输入文本"
echo "   3. 点击 '✨ AI润色' 或 '✍️ AI续写'"
echo ""
echo "📝 日志文件:"
echo "   后端: logs/api.log"
echo "   前端: logs/frontend.log"
echo ""
echo "🛑 停止服务:"
echo "   kill $BACKEND_PID $FRONTEND_PID"
echo ""
echo "或者按 Ctrl+C 然后运行:"
echo "   pkill -f 'python api.py'"
echo "   pkill -f 'next dev'"
echo ""
echo "================================================"

# 保存PID到文件
echo $BACKEND_PID > .backend.pid
echo $FRONTEND_PID > .frontend.pid

echo ""
echo "按 Ctrl+C 停止服务..."
echo ""

# 等待用户中断
trap "echo ''; echo '🛑 停止服务...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; rm -f .backend.pid .frontend.pid; echo '✅ 服务已停止'; exit 0" INT

# 保持脚本运行
wait
