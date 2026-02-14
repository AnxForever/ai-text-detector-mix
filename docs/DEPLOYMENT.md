# 云服务器部署指南

本文档介绍如何将 AI 文本检测系统（前端 + 后端 + 模型）部署到云服务器。

## 架构

```
浏览器 → nginx:80 ─┬─ /         → frontend:3000 (Next.js)
                    └─ /api/*    → backend:8000  (FastAPI + BERT)
                                      ↓
                                 models/ (volume mount, ~780MB)
```

## 前置要求

### 服务器配置

| 项目 | 最低 | 推荐 |
|------|------|------|
| CPU | 2 核 | 4 核+ |
| 内存 | 4 GB | 8 GB+ |
| 磁盘 | 5 GB | 10 GB+ |
| GPU | 不需要（CPU 推理可用） | NVIDIA GPU + CUDA |
| OS | Ubuntu 22.04 / 24.04 | 同左 |

> CPU 推理延迟约 200-500ms/条，GPU 约 20-50ms/条。毕设演示 CPU 足够。

### 软件

- Docker 24+
- Docker Compose v2+
- Git

## 部署步骤

### 1. 安装 Docker

```bash
# Ubuntu
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# 重新登录使 docker 组生效
```

### 2. 上传项目代码

```bash
# 方法 A: Git clone（不含模型文件）
git clone <your-repo-url> datacollection
cd datacollection

# 方法 B: 直接上传
scp -r /mnt/c/datacollection user@server:/home/user/datacollection
```

### 3. 上传模型文件

模型文件约 780MB，不包含在 Git 仓库中，需要单独传输：

```bash
# 从本地上传到服务器
scp -r models/bert_v11c_boundary_fix user@server:~/datacollection/models/
scp -r models/bert_span_detector user@server:~/datacollection/models/
```

验证模型目录结构：

```
models/
├── bert_v11c_boundary_fix/
│   ├── config.json
│   ├── model.safetensors    (~390MB)
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── vocab.txt
└── bert_span_detector/
    ├── config.json
    ├── model.safetensors    (~388MB)
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── vocab.txt
```

### 4. 配置环境变量

```bash
cd datacollection
cp .env.deploy.example .env.deploy
```

编辑 `.env.deploy`：

```bash
# 必改项
CORS_ORIGINS=http://your-server-ip,http://your-domain.com

# 可选：如果需要聊天/续写功能
OPENAI_API_KEY=sk-your-key-here
OPENAI_BASE_URL=https://your-api-proxy.com/v1
```

### 5. 构建并启动

```bash
docker compose build
docker compose up -d
```

首次构建需要下载基础镜像和安装依赖，后续启动很快。

### 6. 验证部署

```bash
# 检查容器状态
docker compose ps

# 检查后端健康
curl http://localhost/api/health

# 测试检测 API
curl -X POST http://localhost/api/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "这是一段测试文本"}'
```

浏览器访问 `http://your-server-ip` 即可看到前端页面。

## GPU 支持（可选）

如果服务器有 NVIDIA GPU：

### 1. 安装 NVIDIA Container Toolkit

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 2. 启用 GPU 配置

编辑 `docker-compose.yml`，取消 backend 服务中 `deploy` 部分的注释：

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

### 3. 使用 GPU 版 PyTorch

编辑 `api/requirements_api.txt`，将 `torch>=2.0.0` 改为指定 CUDA 版本：

```
--index-url https://download.pytorch.org/whl/cu121
torch>=2.0.0
```

## HTTPS / 域名（可选）

### 使用 Let's Encrypt 免费 SSL

```bash
# 安装 certbot
sudo apt install certbot

# 获取证书（先确保域名 DNS 已指向服务器 IP）
sudo certbot certonly --standalone -d your-domain.com

# 证书路径
# /etc/letsencrypt/live/your-domain.com/fullchain.pem
# /etc/letsencrypt/live/your-domain.com/privkey.pem
```

在 `docker-compose.yml` 中挂载证书并在 `nginx.conf` 中启用 HTTPS 配置块。

## 常用运维命令

```bash
# 查看日志
docker compose logs -f backend     # 后端日志
docker compose logs -f frontend    # 前端日志
docker compose logs -f nginx       # nginx 日志

# 重启单个服务
docker compose restart backend

# 更新代码后重新构建
docker compose build --no-cache
docker compose up -d

# 停止所有服务
docker compose down

# 清理未使用的镜像
docker system prune -f
```

## 安全组 / 防火墙

确保云服务器安全组放行以下端口：

| 端口 | 协议 | 用途 |
|------|------|------|
| 80 | TCP | HTTP |
| 443 | TCP | HTTPS（如配置） |
| 22 | TCP | SSH |

**不要**开放 3000 和 8000 端口——这些通过 nginx 代理访问。

## 故障排除

| 问题 | 排查方法 |
|------|---------|
| 前端显示 502 | `docker compose logs backend` 检查后端是否启动成功 |
| 模型加载失败 | 检查 `models/` 目录是否正确挂载，文件是否完整 |
| CUDA 不可用 | 后端会自动回退 CPU，检查 `nvidia-smi` 和 Container Toolkit |
| 检测响应慢 | CPU 推理正常 200-500ms，如需更快则配置 GPU |
| 聊天功能报错 | 检查 `.env.deploy` 中 `OPENAI_API_KEY` 是否配置 |
