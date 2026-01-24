# AGENTS.md

此文件包含智能编码助手在此仓库中工作时所需的指南。

## 构建和测试命令

```bash
# 运行主程序（需先配置 API 密钥）
python "new data collection.py"

# 测试单个 API 连接
python test_new_api.py

# 测试所有模型连接
python test_all_models.py

# 查看生成进度（一次性）
python monitor_progress.py

# 持续监控模式（每 5 分钟刷新）
python monitor_progress.py --watch
```

**注意**：此项目不使用正式的测试框架。测试通过运行独立的脚本执行。

## 代码风格指南

### 导入顺序

1. 标准库导入
2. 第三方库导入（openai, pandas, requests）
3. 本地模块导入

```python
import openai
import pandas as pd
import time
from datetime import datetime
from typing import List, Dict, Optional
import logging
```

### 文件编码

所有 Python 文件必须以 UTF-8 编码声明开头：
```python
# -*- coding: utf-8 -*-
```

### 缩进和格式

- 使用 **4 空格**缩进（不允许制表符）
- 文件名中允许空格（如 "new data collection.py"）
- 运行 Python 脚本时文件名需加引号

### 类型提示

使用 `typing` 模块提供类型注解：
```python
def _call_api(self, prompt: str, max_tokens: int = 2000) -> Optional[str]:
    """Direct API call using requests"""
    pass
```

### 命名约定

- **类名**：`PascalCase`（如 `AutoDimensionGenerator`）
- **函数/方法**：`snake_case`（如 `generate_topics`）
- **常量**：`UPPER_SNAKE_CASE`（如 `API_KEY`, `MODEL_CONFIGS`）
- **变量**：`snake_case`（如 `data_records`, `plan_item`）
- **私有方法**：前缀下划线（如 `_call_api`, `_get_default_topics`）

### 文档字符串

使用简洁的文档字符串描述函数/方法功能：
```python
def _call_api(self, prompt: str, max_tokens: int = 2000) -> Optional[str]:
    """Direct API call using requests"""
    pass
```

### 错误处理

- 使用 `try-except` 捕获异常
- 记录错误日志而非打印
- 提供默认值或备用方案：

```python
try:
    content = self._call_api(prompt)
    if not content:
        raise Exception("API call failed")
    # 处理逻辑
except Exception as e:
    logger.error(f"生成话题失败: {e}")
    return self._get_default_topics(num_topics)
```

### 日志记录

使用 `logging` 模块统一日志管理：
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("从缓存加载: {cache_key}")
logger.error(f"生成话题失败: {e}")
```

### 文件 I/O

- CSV 文件使用 `utf-8-sig` 编码（Excel 兼容）
- JSON 文件使用 `ensure_ascii=False` 支持中文：

```python
df.to_csv(output_file, index=False, encoding='utf-8-sig')

with open(file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
```

### API 调用

使用 `requests` 直接调用 API（绕过 WAF）：
```python
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}
resp = requests.post(url, headers=headers, json=data, timeout=60)
```

### 缓存机制

缓存文件保存 24 小时，检查过期时间：
```python
cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
if os.path.exists(cache_file):
    cached_data = json.load(open(cache_file))
    if cached_data.get('expires', 0) > time.time():
        return cached_data['data']
```

### 字符串格式化

使用 f-string：
```python
logger.info(f"从缓存加载: {cache_key}")
f"质量:{quality:.2f}"
```

### 虚拟环境

项目使用 `.venv/` 虚拟环境，运行命令：
```bash
# Windows
.venv\Scripts\python.exe script.py

# Linux/Mac
.venv/bin/python script.py
```

## 项目特定模式

### 类组织

每个类有明确的职责范围：
- `AutoDimensionGenerator`：维度生成 + 缓存
- `IntelligentCombinationGenerator`：组合生成 + 匹配逻辑
- `TextQualityAssessor`：质量评估

### 进度保存

每处理一定数量条目（如 20-50 条）后保存进度：
```python
if (idx + 1) % 20 == 0:
    df.to_csv(output_file, mode='a', header=False, ...)
    json.dump(plan, ...)
```

### 中英文混合

代码注释和日志支持中英文混合：
```python
logger.info(f"生成并缓存: {cache_key}")
logger.error(f"生成话题失败: {e}")
```

## 未配置的工具

此项目未配置以下工具，如需添加请创建相应配置：
- **Linting**：flake8, pylint, ruff
- **Type checking**：mypy
- **Formatting**：black, autopep8
- **Testing**：pytest, unittest
