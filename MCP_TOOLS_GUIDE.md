# 🛠️ AI文本检测项目 - 推荐工具配置

## 📦 已配置的 MCP 服务器

### 核心工具（已启用）

#### 1. 📁 Filesystem
**用途**: 文件管理
- 读写项目文件
- 管理数据集
- 备份模型文件

#### 2. 🔀 Git  
**用途**: 版本控制
- 提交代码变更
- 查看历史记录
- 管理分支

#### 3. 🧠 Memory
**用途**: 持久化记忆
- 记住实验参数
- 保存训练配置
- 跨会话记忆项目细节

#### 4. 🤔 Sequential Thinking
**用途**: 结构化思考
- 分析模型性能问题
- 优化训练流程
- 论文写作规划

#### 5. 🌐 Fetch
**用途**: HTTP 请求
- 下载公开数据集
- 查询 API
- 获取论文参考

#### 6. 💾 SQLite
**用途**: 数据库管理
- 记录实验结果
- 管理训练日志
- 性能指标追踪

#### 7. 🐙 GitHub
**用途**: 代码托管
- 推送代码
- 管理 Issue
- 协作开发

---

## 🎯 针对你项目的使用场景

### 📊 数据管理
```
"用 filesystem 帮我整理数据集目录"
"用 sqlite 创建实验结果数据库"
```

### 🤖 模型训练
```
"用 memory 记住最佳训练参数"
"用 sequential-thinking 分析为什么准确率下降"
```

### 📝 论文写作
```
"用 fetch 下载相关论文"
"用 memory 记住论文的关键数据"
```

### 🔄 版本控制
```
"用 git 提交今天的代码改动"
"用 github 创建项目仓库"
```

---

## 🚀 启用方法

### 1. 重启 Kiro CLI
```bash
# 退出当前会话
/quit

# 重新启动
kiro-cli chat
```

### 2. 查看已加载的 MCP
```bash
/mcp
```

### 3. 开始使用
直接在对话中说：
```
"用 memory 记住：最佳学习率是 2e-5"
"用 git 查看最近的提交记录"
"用 sequential-thinking 帮我分析模型过拟合问题"
```

---

## 🔧 可选配置

### GitHub Token（如需使用 GitHub 功能）
1. 访问: https://github.com/settings/tokens
2. 创建 Personal Access Token
3. 编辑配置文件:
```bash
nano ~/.kiro/mcp-servers.json
```
4. 替换 `your_token_here` 为你的 token

---

## 💡 实用技巧

### 实验管理
```
"用 memory 记住这次实验：
- 模型: BERT
- 准确率: 100%
- 训练时间: 2小时"
```

### 数据分析
```
"用 sqlite 创建数据库记录所有实验结果"
"用 sequential-thinking 分析为什么去偏后性能提升"
```

### 论文写作
```
"用 fetch 下载 BERT 论文"
"用 memory 记住论文的关键贡献点"
```

---

## 📋 配置文件位置

- **MCP 配置**: `~/.kiro/mcp-servers.json`
- **项目配置**: `/mnt/c/datacollection/.kiro/`

---

## ✅ 下一步

1. 重启 Kiro CLI 加载配置
2. 运行 `/mcp` 查看状态
3. 开始使用：
   ```
   "用 memory 记住我的项目信息"
   "用 sequential-thinking 帮我规划论文结构"
   ```

---

**配置已完成！重启后即可使用这些工具！** 🎉
