"""
数据生成模块

功能：
- 基于模板生成prompt
- 多API并行生成
- 进度显示和错误重试
- 自动保存和断点续传
"""

import asyncio
import json
import csv
import time
from pathlib import Path
from typing import Optional, Callable, TypedDict
from dataclasses import dataclass, field
from datetime import datetime
import httpx
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TaskProgressColumn
from rich.table import Table

from .config import APIConfigManager, APIConfig
from .models import ModelManager

console = Console()


class APIError(Exception):
    """API调用失败异常"""
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}: {message}")


class GenerationStats(TypedDict):
    """生成统计类型"""
    total: int
    success: int
    failed: int
    pending: int


@dataclass
class GenerationTask:
    """单个生成任务"""
    id: int
    prompt: str
    api_name: str
    model: str
    status: str = "pending"  # pending, running, success, failed
    result: str = ""
    error: str = ""
    retries: int = 0
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


@dataclass
class GenerationConfig:
    """生成配置"""
    system_prompt: str = "你是一个创意写作助手，请根据要求生成自然流畅的中文文本。"
    temperature: float = 0.8
    max_tokens: int = 1024
    max_retries: int = 3
    retry_delay: float = 2.0
    save_interval: int = 10  # 每N条保存一次


class DataGenerator:
    """数据生成器"""

    def __init__(
        self,
        config_manager: APIConfigManager,
        model_manager: ModelManager
    ):
        self.config_manager = config_manager
        self.model_manager = model_manager
        self.tasks: list[GenerationTask] = []
        self.stats: GenerationStats = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "pending": 0,
        }

    def create_prompts(
        self,
        topics: list[str],
        template: str = "请以「{topic}」为主题，写一段200-300字的文章。",
        count_per_topic: int = 1
    ) -> list[str]:
        """根据主题和模板创建prompts"""
        prompts = []
        for topic in topics:
            for _ in range(count_per_topic):
                prompts.append(template.format(topic=topic))
        return prompts

    def distribute_tasks(
        self,
        prompts: list[str],
        api_models: Optional[list[tuple[APIConfig, str]]] = None
    ) -> list[GenerationTask]:
        """将prompts分配到各个API"""
        if api_models is None:
            api_models = self.model_manager.get_model_for_generation()

        if not api_models:
            raise ValueError("没有可用的API和模型配对")

        tasks = []
        for i, prompt in enumerate(prompts):
            # 轮询分配到不同API
            api, model = api_models[i % len(api_models)]
            task = GenerationTask(
                id=i,
                prompt=prompt,
                api_name=api.name,
                model=model
            )
            tasks.append(task)

        self.tasks = tasks
        self.stats["total"] = len(tasks)
        self.stats["pending"] = len(tasks)
        return tasks

    async def _call_api(
        self,
        api: APIConfig,
        model: str,
        prompt: str,
        config: GenerationConfig
    ) -> str:
        """调用单个API生成文本"""
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{api.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api.get_api_key()}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": config.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens
                }
            )

            if response.status_code != 200:
                raise APIError(response.status_code, response.text[:200])

            data = response.json()
            return data["choices"][0]["message"]["content"]

    async def _run_task(
        self,
        task: GenerationTask,
        config: GenerationConfig,
        semaphore: asyncio.Semaphore
    ) -> GenerationTask:
        """执行单个生成任务"""
        api = self.config_manager.get_api(task.api_name)
        if not api:
            task.status = "failed"
            task.error = f"API {task.api_name} 不存在"
            return task

        async with semaphore:
            task.status = "running"

            for retry in range(config.max_retries):
                try:
                    result = await self._call_api(
                        api, task.model, task.prompt, config
                    )
                    task.result = result
                    task.status = "success"
                    task.completed_at = time.time()
                    self.stats["success"] += 1
                    self.stats["pending"] -= 1
                    return task

                except httpx.TimeoutException as e:
                    task.retries = retry + 1
                    task.error = f"Timeout: {e}"
                except httpx.RequestError as e:
                    task.retries = retry + 1
                    task.error = f"Network error: {e}"
                except APIError as e:
                    task.retries = retry + 1
                    task.error = str(e)
                except (json.JSONDecodeError, KeyError) as e:
                    task.retries = retry + 1
                    task.error = f"Response parse error: {e}"

                if retry < config.max_retries - 1:
                    await asyncio.sleep(config.retry_delay * (retry + 1))

            task.status = "failed"
            task.completed_at = time.time()
            self.stats["failed"] += 1
            self.stats["pending"] -= 1
            return task

    async def run_generation(
        self,
        config: Optional[GenerationConfig] = None,
        output_path: Optional[Path] = None,
        max_concurrent: int = 5,
        on_progress: Optional[Callable[[dict], None]] = None
    ) -> list[GenerationTask]:
        """执行批量生成"""
        if config is None:
            config = GenerationConfig()

        if not self.tasks:
            raise ValueError("没有待生成的任务，请先调用 distribute_tasks")

        # 筛选pending任务（支持断点续传）
        pending_tasks = [t for t in self.tasks if t.status == "pending"]

        if not pending_tasks:
            console.print("[yellow]所有任务已完成[/yellow]")
            return self.tasks

        semaphore = asyncio.Semaphore(max_concurrent)
        completed = []

        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            main_task = progress.add_task(
                "生成进度",
                total=len(pending_tasks)
            )

            async def run_and_update(task: GenerationTask):
                result = await self._run_task(task, config, semaphore)
                progress.advance(main_task)

                if on_progress:
                    on_progress(self.stats.copy())

                # 定期保存
                if output_path and len(completed) % config.save_interval == 0:
                    self.save_results(output_path)

                completed.append(result)
                return result

            await asyncio.gather(*[run_and_update(t) for t in pending_tasks])

        # 最终保存
        if output_path:
            self.save_results(output_path)

        return self.tasks

    def save_results(self, output_path: Path) -> None:
        """保存生成结果"""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 确定文件格式
        suffix = output_path.suffix.lower()

        if suffix == ".json":
            self._save_json(output_path)
        elif suffix == ".jsonl":
            self._save_jsonl(output_path)
        else:  # 默认CSV
            self._save_csv(output_path)

    def _save_csv(self, output_path: Path) -> None:
        """保存为CSV格式"""
        with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "id", "text", "label", "category", "source",
                "prompt", "model", "api_name", "status"
            ])
            writer.writeheader()

            for task in self.tasks:
                if task.status == "success":
                    writer.writerow({
                        "id": task.id,
                        "text": task.result,
                        "label": 1,  # AI生成
                        "category": "AI",
                        "source": f"{task.api_name}/{task.model}",
                        "prompt": task.prompt,
                        "model": task.model,
                        "api_name": task.api_name,
                        "status": task.status
                    })

    def _save_json(self, output_path: Path) -> None:
        """保存为JSON格式"""
        data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "stats": self.stats
            },
            "results": [
                {
                    "id": t.id,
                    "text": t.result,
                    "label": 1,
                    "category": "AI",
                    "source": f"{t.api_name}/{t.model}",
                    "prompt": t.prompt,
                    "model": t.model,
                    "api_name": t.api_name,
                    "status": t.status,
                    "error": t.error if t.status == "failed" else None
                }
                for t in self.tasks
            ]
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _save_jsonl(self, output_path: Path) -> None:
        """保存为JSONL格式"""
        with open(output_path, "w", encoding="utf-8") as f:
            for task in self.tasks:
                if task.status == "success":
                    record = {
                        "text": task.result,
                        "label": 1,
                        "category": "AI",
                        "source": f"{task.api_name}/{task.model}",
                        "prompt": task.prompt
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def load_checkpoint(self, checkpoint_path: Path) -> bool:
        """从检查点恢复"""
        if not checkpoint_path.exists():
            return False

        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.tasks = []
            for t in data.get("results", []):
                task = GenerationTask(
                    id=t["id"],
                    prompt=t["prompt"],
                    api_name=t["api_name"],
                    model=t["model"],
                    status=t["status"],
                    result=t.get("text", ""),
                    error=t.get("error", "")
                )
                self.tasks.append(task)

            # 重新计算统计
            self.stats: GenerationStats = {
                "total": len(self.tasks),
                "success": sum(1 for t in self.tasks if t.status == "success"),
                "failed": sum(1 for t in self.tasks if t.status == "failed"),
                "pending": sum(1 for t in self.tasks if t.status == "pending"),
            }

            console.print(f"[green]已恢复 {len(self.tasks)} 个任务[/green]")
            return True

        except json.JSONDecodeError as e:
            console.print(f"[red]检查点JSON格式错误: {e}[/red]")
            return False
        except KeyError as e:
            console.print(f"[red]检查点数据结构错误: 缺少字段 {e}[/red]")
            return False
        except OSError as e:
            console.print(f"[red]读取检查点文件失败: {e}[/red]")
            return False

    def display_stats(self) -> None:
        """显示生成统计"""
        table = Table(title="生成统计")
        table.add_column("指标", style="cyan")
        table.add_column("数值", style="green", justify="right")

        table.add_row("总任务数", str(self.stats["total"]))
        table.add_row("成功", str(self.stats["success"]))
        table.add_row("失败", str(self.stats["failed"]))
        table.add_row("待处理", str(self.stats["pending"]))

        if self.stats["total"] > 0:
            success_rate = self.stats["success"] / self.stats["total"] * 100
            table.add_row("成功率", f"{success_rate:.1f}%")

        console.print(table)
