"""
AI文本检测模型 - 启动器
直接运行: python start.py
"""
import os
import sys
import subprocess
import platform

def main():
    print("=" * 60)
    print("AI文本检测模型 - 启动中...")
    print("=" * 60)
    print()

    # 设置离线模式
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    print("✓ 离线模式已设置")

    # 检查模型是否存在
    model_path = "models/bert_improved/best_model"
    if not os.path.exists(model_path):
        print(f"✗ 错误: 模型文件不存在")
        print(f"  路径: {model_path}")
        return
    print(f"✓ 模型文件已找到")

    # 检查测试脚本
    test_script = "scripts/evaluation/test_single_text.py"
    if not os.path.exists(test_script):
        print(f"✗ 错误: 测试脚本不存在")
        print(f"  路径: {test_script}")
        return
    print(f"✓ 测试脚本已找到")

    # 检查并使用虚拟环境的Python
    if os.name == 'nt':  # Windows
        venv_python = os.path.join('.venv', 'Scripts', 'python.exe')
    else:  # Linux/macOS
        venv_python = os.path.join('.venv', 'bin', 'python')

    if os.path.exists(venv_python):
        python_exe = venv_python
        print(f"✓ 使用虚拟环境Python (支持GPU)")
    else:
        python_exe = sys.executable
        print("⚠️ 虚拟环境未找到，使用系统Python")

    print()
    print("=" * 60)
    print("启动测试工具...")
    print("=" * 60)
    print()

    # 运行测试工具
    try:
        subprocess.run([python_exe, test_script, "--interactive"], check=True)
    except KeyboardInterrupt:
        print("\n\n程序已退出")
    except Exception as e:
        print(f"\n✗ 运行出错: {e}")

if __name__ == "__main__":
    main()
