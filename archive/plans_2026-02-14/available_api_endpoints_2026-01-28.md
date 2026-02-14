# 可用 API 列表（用于构建）

> 更新日期: 2026-01-28
> 说明: 仅记录可用 API 与配置前缀，不包含密钥。

---

## 已验证可用

1) 本地代理
- base_url: http://192.168.60.105:8317/v1
- env prefix: LOCAL_PROXY
- 备注: 当前用于 glm-4.7 / gpt-4 / deepseek-v3.2

2) hybgzs
- base_url: https://ai.hybgzs.com/v1
- env prefix: HYBGZS_PROXY
- 备注: 自动拉取 gemini-3 系列模型

3) NVIDIA
- base_url: https://integrate.api.nvidia.com/v1
- env prefix: NVIDIA
- 备注: 固定模型 meta/llama-3.1-405b-instruct

4) hotaru
- base_url: https://api.hotaruapi.top/v1
- env prefix: HOTARU_PROXY
- 备注: 固定使用 gpt-5（稳定优先）

5) china.184772
- base_url: https://china.184772.xyz/v1
- env prefix: CHINA184_PROXY
- 备注: 固定使用 gpt-oss-120b

6) x666
- base_url: https://x666.me/v1
- env prefix: X666_PROXY
- 备注: 固定使用 gpt-5.2

7) crisxie
- base_url: https://api.crisxie.top/v1
- env prefix: CRISXIE_PROXY
- 备注: 固定使用 gpt-5.2

---

## 验证失败（不纳入构建）

- agentrouter: https://agentrouter.org
  - /v1/models 返回 401（unauthorized）
  - /v1beta/models 返回 404
  - /models 返回 HTML（非模型接口）
