"""
API配置管理模块

功能：
- 添加/删除/修改API配置
- 测试API连接状态
- 持久化配置到JSON文件
- 支持环境变量存储API密钥
"""

import json
import os
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, asdict, field
import httpx
from rich.console import Console
from rich.table import Table

console = Console()

# 默认配置文件路径
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "apis.json"


@dataclass
class APIConfig:
    """单个API配置

    API密钥存储策略：
    - 如果 api_key_env 设置，从环境变量读取密钥（推荐）
    - 否则使用 api_key 字段（仅用于测试）
    """
    name: str
    base_url: str
    api_key: str = ""  # 直接存储的密钥（不推荐用于生产）
    api_key_env: str = ""  # 环境变量名称（推荐）
    enabled: bool = True
    rate_limit: int = 10  # 每分钟请求数
    models_cache: list[str] = field(default_factory=list)

    def get_api_key(self) -> str:
        """获取API密钥，优先使用环境变量"""
        if self.api_key_env:
            key = os.environ.get(self.api_key_env, "")
            if key:
                return key
            console.print(f"[yellow]警告: 环境变量 {self.api_key_env} 未设置[/yellow]")
        return self.api_key

    def to_dict(self) -> dict[str, Any]:
        """转换为字典，不包含直接的API密钥（如果使用环境变量）"""
        data = asdict(self)
        # 如果使用环境变量，不保存实际密钥
        if self.api_key_env:
            data["api_key"] = ""
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "APIConfig":
        return cls(**data)


class APIConfigManager:
    """API配置管理器"""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.apis: dict[str, APIConfig] = {}
        self._load_config()

    def _load_config(self) -> None:
        """从文件加载配置"""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for api_data in data.get("apis", []):
                    api = APIConfig.from_dict(api_data)
                    self.apis[api.name] = api
            except (json.JSONDecodeError, KeyError) as e:
                console.print(f"[yellow]警告: 配置文件格式错误: {e}[/yellow]")

    def _save_config(self) -> None:
        """保存配置到文件"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"apis": [api.to_dict() for api in self.apis.values()]}
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def save(self) -> None:
        """保存配置（公共接口）"""
        self._save_config()

    def add_api(
        self,
        name: str,
        base_url: str,
        api_key: str = "",
        api_key_env: str = "",
        rate_limit: int = 10,
        enabled: bool = True
    ) -> bool:
        """添加新的API配置

        Args:
            name: API名称（唯一标识）
            base_url: API基础URL
            api_key: API密钥（直接存储，不推荐）
            api_key_env: 存储API密钥的环境变量名（推荐）
            rate_limit: 每分钟请求限制
            enabled: 是否启用
        """
        if name in self.apis:
            console.print(f"[red]错误: API '{name}' 已存在[/red]")
            return False

        if not api_key and not api_key_env:
            console.print("[red]错误: 必须提供 api_key 或 api_key_env[/red]")
            return False

        # 规范化URL
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"

        api = APIConfig(
            name=name,
            base_url=base_url,
            api_key=api_key,
            api_key_env=api_key_env,
            rate_limit=rate_limit,
            enabled=enabled
        )
        self.apis[name] = api
        self._save_config()
        console.print(f"[green]成功添加API: {name}[/green]")
        return True

    def remove_api(self, name: str) -> bool:
        """删除API配置"""
        if name not in self.apis:
            console.print(f"[red]错误: API '{name}' 不存在[/red]")
            return False

        del self.apis[name]
        self._save_config()
        console.print(f"[green]成功删除API: {name}[/green]")
        return True

    def update_api(
        self,
        name: str,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        api_key_env: Optional[str] = None,
        enabled: Optional[bool] = None,
        rate_limit: Optional[int] = None
    ) -> bool:
        """更新API配置

        Args:
            name: 要更新的API名称
            base_url: 新的API基础URL
            api_key: 新的API密钥
            api_key_env: 新的环境变量名称
            enabled: 是否启用
            rate_limit: 新的请求限制
        """
        if name not in self.apis:
            console.print(f"[red]错误: API '{name}' 不存在[/red]")
            return False

        api = self.apis[name]
        if base_url is not None:
            api.base_url = base_url
        if api_key is not None:
            api.api_key = api_key
        if api_key_env is not None:
            api.api_key_env = api_key_env
        if enabled is not None:
            api.enabled = enabled
        if rate_limit is not None:
            api.rate_limit = rate_limit

        self._save_config()
        console.print(f"[green]成功更新API: {name}[/green]")
        return True

    def get_api(self, name: str) -> Optional[APIConfig]:
        """获取指定API配置"""
        return self.apis.get(name)

    def list_apis(self) -> list[APIConfig]:
        """列出所有API配置"""
        return list(self.apis.values())

    def get_enabled_apis(self) -> list[APIConfig]:
        """获取所有已启用的API"""
        return [api for api in self.apis.values() if api.enabled]

    async def test_api(self, name: str) -> tuple[bool, str]:
        """测试API连接状态"""
        api = self.apis.get(name)
        if not api:
            return False, f"API '{name}' 不存在"

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{api.base_url}/models",
                    headers={"Authorization": f"Bearer {api.get_api_key()}"}
                )
                if response.status_code == 200:
                    data = response.json()
                    model_count = len(data.get("data", []))
                    return True, f"连接成功，发现 {model_count} 个模型"
                else:
                    return False, f"HTTP {response.status_code}: {response.text[:100]}"
        except httpx.TimeoutException:
            return False, "连接超时"
        except httpx.RequestError as e:
            return False, f"连接错误: {str(e)}"

    def display_apis(self) -> None:
        """以表格形式显示所有API配置"""
        table = Table(title="API配置列表")
        table.add_column("名称", style="cyan")
        table.add_column("Base URL", style="green")
        table.add_column("状态", style="yellow")
        table.add_column("限速", style="magenta")
        table.add_column("缓存模型数", style="blue")

        for api in self.apis.values():
            status = "✅ 启用" if api.enabled else "❌ 禁用"
            table.add_row(
                api.name,
                api.base_url,
                status,
                f"{api.rate_limit}/min",
                str(len(api.models_cache))
            )

        if not self.apis:
            console.print("[yellow]暂无API配置[/yellow]")
        else:
            console.print(table)
