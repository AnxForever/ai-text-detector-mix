#!/bin/bash
# 模型保护脚本 - 防止意外删除

echo "🔒 正在保护模型文件..."

# 设置模型文件为只读
chmod -R 444 models/bert_improved/best_model/*.safetensors 2>/dev/null
chmod -R 444 models/bert_improved/best_model/config.json 2>/dev/null
chmod -R 444 models/bert_improved/final_model/*.safetensors 2>/dev/null
chmod -R 444 models/bert_improved/final_model/config.json 2>/dev/null

# 设置备份为只读
chmod -R 444 backup_models/ 2>/dev/null

# 设置目录为只读（但保留执行权限）
chmod 555 models/bert_improved/best_model/ 2>/dev/null
chmod 555 models/bert_improved/final_model/ 2>/dev/null
chmod 555 backup_models/ 2>/dev/null

echo "✅ 模型文件已设置为只读保护"
echo ""
echo "受保护的文件："
echo "  - models/bert_improved/best_model/"
echo "  - models/bert_improved/final_model/"
echo "  - backup_models/"
echo ""
echo "⚠️  如需修改，请先运行: chmod +w <文件路径>"
