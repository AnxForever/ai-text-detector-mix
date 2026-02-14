"""
模型查询模块

功能：
- 查询指定API的可用模型
- 缓存模型信息
- 显示模型汇总
"""

import asyncio
import json
from typing import Optional
import httpx
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from .config import APIConfigManager, APIConfig

console = Console()


class ModelManager:
    """模型管理器"""

    def __init__(self, config_manager: APIConfigManager):
        self.config_manager = config_manager

    async def fetch_models(self, api: APIConfig, force_refresh: bool = False) -> list[str]:
        """获取指定API的可用模型列表"""
        # 如果有缓存且不强制刷新，直接返回
        if api.models_cache and not force_refresh:
            return api.models_cache

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{api.base_url}/models",
                    headers={"Authorization": f"Bearer {api.get_api_key()}"}
                )

                if response.status_code == 200:
                    data = response.json()
                    models = [m.get("id", "") for m in data.get("data", [])]
                    models = sorted([m for m in models if m])  # 过滤空值并排序

                    # 更新缓存
                    api.models_cache = models
                    self.config_manager.save()

                    return models
                else:
                    console.print(f"[red]获取模型失败: HTTP {response.status_code}[/red]")
                    return []
        except httpx.TimeoutException as e:
            console.print(f"[red]请求超时: {e}[/red]")
            return []
        except httpx.RequestError as e:
            console.print(f"[red]网络错误: {e}[/red]")
            return []
        except (json.JSONDecodeError, KeyError) as e:
            console.print(f"[red]响应解析错误: {e}[/red]")
            return []

    async def fetch_all_models(self, force_refresh: bool = False) -> dict[str, list[str]]:
        """获取所有API的可用模型"""
        results = {}
        apis = self.config_manager.get_enabled_apis()

        if not apis:
            console.print("[yellow]没有已启用的API[/yellow]")
            return results

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("正在查询模型...", total=len(apis))

            async def fetch_one(api: APIConfig):
                models = await self.fetch_models(api, force_refresh)
                progress.advance(task)
                return api.name, models

            tasks = [fetch_one(api) for api in apis]
            for coro in asyncio.as_completed(tasks):
                name, models = await coro
                results[name] = models

        return results

    def display_models(self, api_name: str, models: list[str]) -> None:
        """显示单个API的模型列表"""
        if not models:
            console.print(f"[yellow]API '{api_name}' 没有可用模型[/yellow]")
            return

        table = Table(title=f"[bold]{api_name}[/bold] 可用模型 ({len(models)}个)")
        table.add_column("序号", style="dim", width=6)
        table.add_column("模型ID", style="cyan")

        for i, model in enumerate(models, 1):
            table.add_row(str(i), model)

        console.print(table)

    def display_all_models(self, all_models: dict[str, list[str]]) -> None:
        """显示所有API的模型汇总"""
        if not all_models:
            console.print("[yellow]没有可用的模型数据[/yellow]")
            return

        # 统计信息
        total_apis = len(all_models)
        total_models = sum(len(models) for models in all_models.values())

        # 找出通用模型（多个API都有的）
        model_counts = {}
        for api_name, models in all_models.items():
            for model in models:
                if model not in model_counts:
                    model_counts[model] = []
                model_counts[model].append(api_name)

        common_models = {m: apis for m, apis in model_counts.items() if len(apis) > 1}

        # 创建汇总表格
        table = Table(title=f"模型汇总 (共{total_apis}个API, {total_models}个模型)")
        table.add_column("API名称", style="cyan")
        table.add_column("模型数量", style="green", justify="right")
        table.add_column("热门模型 (前5个)", style="yellow")

        for api_name, models in all_models.items():
            top_models = ", ".join(models[:5])
            if len(models) > 5:
                top_models += f" (+{len(models)-5})"
            table.add_row(api_name, str(len(models)), top_models)

        console.print(table)

        # 显示通用模型
        if common_models:
            console.print("\n")
            common_table = Table(title="通用模型 (多个API可用)")
            common_table.add_column("模型", style="cyan")
            common_table.add_column("可用API", style="green")

            for model, apis in sorted(common_models.items(), key=lambda x: -len(x[1])):
                common_table.add_row(model, ", ".join(apis))

            console.print(common_table)

    def get_model_for_generation(
        self,
        preferred_models: Optional[list[str]] = None
    ) -> list[tuple[APIConfig, str]]:
        """
        获取用于生成的API和模型配对

        返回: [(api_config, model_name), ...]
        """
        results = []

        # 优先选择的模型列表
        if preferred_models is None:
            preferred_models = [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo",
                "gpt-4",
                "gpt-3.5-turbo",
                "claude-3-5-sonnet",
                "claude-3-sonnet",
                "deepseek-chat",
                "qwen-plus",
            ]

        for api in self.config_manager.get_enabled_apis():
            if not api.models_cache:
                continue

            # 找到第一个匹配的优先模型
            selected_model = None
            for pref in preferred_models:
                for cached in api.models_cache:
                    if pref in cached.lower():
                        selected_model = cached
                        break
                if selected_model:
                    break

            # 如果没有优先模型，用第一个
            if not selected_model and api.models_cache:
                selected_model = api.models_cache[0]

            if selected_model:
                results.append((api, selected_model))

        return results
