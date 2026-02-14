#!/usr/bin/env python3
"""
API管理工具命令行入口

使用方式：
    python -m tools.api_manager.cli [command] [options]

命令：
    tui             启动可视化界面（推荐）
    list-apis       列出所有API配置
    add-api         添加新API
    remove-api      删除API
    test-api        测试API连接
    list-models     查询模型列表
    generate        批量生成数据
    menu            进入交互式菜单
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.panel import Panel

from .config import APIConfigManager
from .models import ModelManager
from .generator import DataGenerator, GenerationConfig

console = Console()


def get_managers() -> tuple[APIConfigManager, ModelManager]:
    """获取管理器实例"""
    config_manager = APIConfigManager()
    model_manager = ModelManager(config_manager)
    return config_manager, model_manager


# ===== CLI Commands =====

@click.group()
def cli():
    """🤖 AI数据生成工具 - API管理与批量生成"""
    pass


@cli.command("tui")
def tui_cmd():
    """启动可视化TUI界面（推荐）"""
    from .tui import run_tui
    run_tui()


@cli.command("list-apis")
def list_apis_cmd():
    """列出所有API配置"""
    config_manager, _ = get_managers()
    config_manager.display_apis()


@cli.command("add-api")
@click.option("--name", "-n", required=True, help="API名称")
@click.option("--url", "-u", required=True, help="API基础URL")
@click.option("--key-env", "-e", help="存储API密钥的环境变量名（推荐）")
@click.option("--rate-limit", "-r", default=10, help="每分钟请求限制")
def add_api_cmd(name: str, url: str, key_env: Optional[str], rate_limit: int) -> None:
    """添加新的API配置

    安全提示：推荐使用 --key-env 指定环境变量名，而非直接传递API密钥。
    如果不提供 --key-env，将通过交互式提示输入密钥。
    """
    config_manager, _ = get_managers()

    if key_env:
        # 使用环境变量
        api_key = os.environ.get(key_env, "")
        if not api_key:
            console.print(f"[yellow]警告: 环境变量 {key_env} 未设置，将在运行时读取[/yellow]")
        config_manager.add_api(name, url, api_key_env=key_env, rate_limit=rate_limit)
    else:
        # 交互式输入密钥
        api_key = Prompt.ask("API Key", password=True)
        config_manager.add_api(name, url, api_key=api_key, rate_limit=rate_limit)


@cli.command("remove-api")
@click.argument("name")
def remove_api_cmd(name: str):
    """删除API配置"""
    config_manager, _ = get_managers()

    if Confirm.ask(f"确定要删除API '{name}' 吗?"):
        config_manager.remove_api(name)


@cli.command("test-api")
@click.argument("name", required=False)
def test_api_cmd(name: Optional[str]):
    """测试API连接状态"""
    config_manager, _ = get_managers()

    async def run_test():
        if name:
            success, msg = await config_manager.test_api(name)
            status = "[green]✓[/green]" if success else "[red]✗[/red]"
            console.print(f"{status} {name}: {msg}")
        else:
            # 测试所有API
            for api in config_manager.list_apis():
                success, msg = await config_manager.test_api(api.name)
                status = "[green]✓[/green]" if success else "[red]✗[/red]"
                console.print(f"{status} {api.name}: {msg}")

    asyncio.run(run_test())


@cli.command("list-models")
@click.argument("api_name", required=False)
@click.option("--refresh", "-r", is_flag=True, help="强制刷新缓存")
def list_models_cmd(api_name: Optional[str], refresh: bool):
    """查询可用模型列表"""
    config_manager, model_manager = get_managers()

    async def run_query():
        if api_name:
            api = config_manager.get_api(api_name)
            if not api:
                console.print(f"[red]API '{api_name}' 不存在[/red]")
                return
            models = await model_manager.fetch_models(api, refresh)
            model_manager.display_models(api_name, models)
        else:
            all_models = await model_manager.fetch_all_models(refresh)
            model_manager.display_all_models(all_models)

    asyncio.run(run_query())


@cli.command("generate")
@click.option("--api", "-a", multiple=True, help="指定使用的API（可多次指定）")
@click.option("--model", "-m", help="指定模型")
@click.option("--count", "-c", default=10, help="生成数量")
@click.option("--topic", "-t", multiple=True, help="主题（可多次指定）")
@click.option("--template", help="提示词模板")
@click.option("--output", "-o", required=True, help="输出文件路径")
@click.option("--concurrent", default=5, help="并发数")
@click.option("--resume", is_flag=True, help="从检查点恢复")
def generate_cmd(
    api: tuple,
    model: Optional[str],
    count: int,
    topic: tuple,
    template: Optional[str],
    output: str,
    concurrent: int,
    resume: bool
):
    """批量生成数据"""
    config_manager, model_manager = get_managers()
    generator = DataGenerator(config_manager, model_manager)

    output_path = Path(output)
    checkpoint_path = output_path.with_suffix(".checkpoint.json")

    async def run_generation():
        # 尝试恢复
        if resume and checkpoint_path.exists():
            if generator.load_checkpoint(checkpoint_path):
                console.print("[green]从检查点恢复成功[/green]")
            else:
                console.print("[yellow]检查点加载失败，重新开始[/yellow]")

        if not generator.tasks:
            # 创建新任务
            topics = list(topic) if topic else ["科技发展", "人工智能", "环境保护", "教育改革", "文化传承"]
            tpl = template or "请以「{topic}」为主题，写一段200-300字的文章。"

            prompts = generator.create_prompts(
                topics=topics,
                template=tpl,
                count_per_topic=max(1, count // len(topics))
            )

            # 获取API和模型配对
            if api:
                api_models = []
                for api_name in api:
                    api_config = config_manager.get_api(api_name)
                    if api_config:
                        target_model = model or (api_config.models_cache[0] if api_config.models_cache else None)
                        if target_model:
                            api_models.append((api_config, target_model))
            else:
                # 刷新模型缓存
                await model_manager.fetch_all_models()
                api_models = model_manager.get_model_for_generation()

            if not api_models:
                console.print("[red]没有可用的API，请先添加并测试API配置[/red]")
                return

            generator.distribute_tasks(prompts, api_models)
            console.print(f"[green]创建了 {len(generator.tasks)} 个生成任务[/green]")

        # 执行生成
        config = GenerationConfig()
        await generator.run_generation(
            config=config,
            output_path=output_path,
            max_concurrent=concurrent
        )

        generator.display_stats()
        console.print(f"\n[green]结果已保存到: {output_path}[/green]")

        # 删除检查点
        if checkpoint_path.exists():
            checkpoint_path.unlink()

    asyncio.run(run_generation())


@cli.command("menu")
def menu_cmd():
    """进入交互式菜单"""
    run_interactive_menu()


# ===== Interactive Menu =====

def run_interactive_menu() -> None:
    """运行交互式TUI菜单"""
    config_manager, model_manager = get_managers()
    generator = DataGenerator(config_manager, model_manager)

    while True:
        console.clear()
        console.print(Panel.fit(
            "[bold cyan]🤖 AI数据生成工具 v1.0[/bold cyan]\n\n"
            "[1] 📋 API配置管理\n"
            "[2] 🔍 查看可用模型\n"
            "[3] ⚡ 批量生成数据\n"
            "[4] 🧪 测试API连接\n"
            "[0] 退出",
            title="主菜单",
            border_style="blue"
        ))

        choice = Prompt.ask("请选择", choices=["0", "1", "2", "3", "4"], default="0")

        if choice == "0":
            console.print("[yellow]再见！[/yellow]")
            break
        elif choice == "1":
            api_config_menu(config_manager)
        elif choice == "2":
            model_query_menu(config_manager, model_manager)
        elif choice == "3":
            generation_menu(config_manager, model_manager, generator)
        elif choice == "4":
            test_api_menu(config_manager)


def api_config_menu(config_manager: APIConfigManager) -> None:
    """API配置子菜单"""
    while True:
        console.clear()
        console.print(Panel.fit(
            "[bold cyan]📋 API配置管理[/bold cyan]\n\n"
            "[1] 查看所有API\n"
            "[2] 添加新API\n"
            "[3] 删除API\n"
            "[4] 启用/禁用API\n"
            "[0] 返回",
            title="API配置",
            border_style="green"
        ))

        choice = Prompt.ask("请选择", choices=["0", "1", "2", "3", "4"], default="0")

        if choice == "0":
            break
        elif choice == "1":
            config_manager.display_apis()
            Prompt.ask("\n按回车继续")
        elif choice == "2":
            name = Prompt.ask("API名称")
            url = Prompt.ask("Base URL (如 https://api.example.com/v1)")

            # 询问使用环境变量还是直接输入
            use_env = Confirm.ask("使用环境变量存储密钥？（推荐）", default=True)
            if use_env:
                key_env = Prompt.ask("环境变量名称", default=f"{name.upper()}_API_KEY")
                config_manager.add_api(name, url, api_key_env=key_env, rate_limit=IntPrompt.ask("每分钟请求限制", default=10))
            else:
                key = Prompt.ask("API Key", password=True)
                config_manager.add_api(name, url, api_key=key, rate_limit=IntPrompt.ask("每分钟请求限制", default=10))
            Prompt.ask("\n按回车继续")
        elif choice == "3":
            config_manager.display_apis()
            name = Prompt.ask("要删除的API名称")
            if Confirm.ask(f"确定删除 '{name}'?"):
                config_manager.remove_api(name)
            Prompt.ask("\n按回车继续")
        elif choice == "4":
            config_manager.display_apis()
            name = Prompt.ask("要切换状态的API名称")
            api = config_manager.get_api(name)
            if api:
                config_manager.update_api(name, enabled=not api.enabled)
                status = "启用" if not api.enabled else "禁用"
                console.print(f"[green]已{status} {name}[/green]")
            Prompt.ask("\n按回车继续")


def model_query_menu(config_manager: APIConfigManager, model_manager: ModelManager) -> None:
    """模型查询子菜单"""
    while True:
        console.clear()
        console.print(Panel.fit(
            "[bold cyan]🔍 模型查询[/bold cyan]\n\n"
            "[1] 查看所有API的模型汇总\n"
            "[2] 查看指定API的模型\n"
            "[3] 刷新模型缓存\n"
            "[0] 返回",
            title="模型查询",
            border_style="yellow"
        ))

        choice = Prompt.ask("请选择", choices=["0", "1", "2", "3"], default="0")

        if choice == "0":
            break
        elif choice == "1":
            async def query():
                all_models = await model_manager.fetch_all_models()
                model_manager.display_all_models(all_models)
            asyncio.run(query())
            Prompt.ask("\n按回车继续")
        elif choice == "2":
            config_manager.display_apis()
            name = Prompt.ask("API名称")
            api = config_manager.get_api(name)
            if api:
                async def query():
                    models = await model_manager.fetch_models(api)
                    model_manager.display_models(name, models)
                asyncio.run(query())
            else:
                console.print(f"[red]API '{name}' 不存在[/red]")
            Prompt.ask("\n按回车继续")
        elif choice == "3":
            async def refresh():
                await model_manager.fetch_all_models(force_refresh=True)
                console.print("[green]模型缓存已刷新[/green]")
            asyncio.run(refresh())
            Prompt.ask("\n按回车继续")


def generation_menu(
    config_manager: APIConfigManager,
    model_manager: ModelManager,
    generator: DataGenerator
) -> None:
    """数据生成子菜单"""
    console.clear()
    console.print(Panel.fit(
        "[bold cyan]⚡ 批量生成数据[/bold cyan]",
        title="数据生成",
        border_style="magenta"
    ))

    # 1. 选择API
    config_manager.display_apis()
    enabled_apis = config_manager.get_enabled_apis()
    if not enabled_apis:
        console.print("[red]没有已启用的API，请先配置[/red]")
        Prompt.ask("\n按回车继续")
        return

    use_all = Confirm.ask("使用所有已启用的API?", default=True)
    if not use_all:
        api_names = Prompt.ask("输入要使用的API名称（逗号分隔）")
        selected_apis = [config_manager.get_api(n.strip()) for n in api_names.split(",")]
        selected_apis = [a for a in selected_apis if a]
    else:
        selected_apis = enabled_apis

    # 2. 刷新模型并选择
    async def prepare_models():
        for api in selected_apis:
            await model_manager.fetch_models(api, force_refresh=True)
    asyncio.run(prepare_models())

    api_models = model_manager.get_model_for_generation()
    api_models = [(a, m) for a, m in api_models if a in selected_apis]

    if not api_models:
        console.print("[red]没有可用的模型[/red]")
        Prompt.ask("\n按回车继续")
        return

    console.print("\n[cyan]将使用以下API和模型：[/cyan]")
    for api, model in api_models:
        console.print(f"  • {api.name}: {model}")

    # 3. 设置生成参数
    console.print("\n[cyan]设置生成参数：[/cyan]")
    topics_input = Prompt.ask(
        "主题（逗号分隔）",
        default="科技发展,人工智能,环境保护,教育改革,文化传承"
    )
    topics = [t.strip() for t in topics_input.split(",")]

    count = IntPrompt.ask("每个主题生成数量", default=5)
    concurrent = IntPrompt.ask("并发数", default=5)

    template = Prompt.ask(
        "提示词模板（{topic}为主题占位符）",
        default="请以「{topic}」为主题，写一段200-300字的文章。"
    )

    output = Prompt.ask("输出文件路径", default="output/generated_data.csv")
    output_path = Path(output)

    # 4. 确认并执行
    total = len(topics) * count
    console.print(f"\n[yellow]将生成 {total} 条数据，输出到 {output_path}[/yellow]")

    if not Confirm.ask("开始生成?", default=True):
        return

    # 5. 执行生成
    async def run_gen():
        prompts = generator.create_prompts(topics, template, count)
        generator.distribute_tasks(prompts, api_models)

        config = GenerationConfig()
        await generator.run_generation(
            config=config,
            output_path=output_path,
            max_concurrent=concurrent
        )

        generator.display_stats()
        console.print(f"\n[green]结果已保存到: {output_path}[/green]")

    asyncio.run(run_gen())
    Prompt.ask("\n按回车继续")


def test_api_menu(config_manager: APIConfigManager) -> None:
    """测试API连接子菜单"""
    console.clear()
    console.print(Panel.fit(
        "[bold cyan]🧪 测试API连接[/bold cyan]",
        title="连接测试",
        border_style="red"
    ))

    config_manager.display_apis()

    test_all = Confirm.ask("测试所有API?", default=True)

    async def run_tests():
        apis_to_test = config_manager.list_apis() if test_all else []

        if not test_all:
            name = Prompt.ask("输入要测试的API名称")
            api = config_manager.get_api(name)
            if api:
                apis_to_test = [api]
            else:
                console.print(f"[red]API '{name}' 不存在[/red]")
                return

        console.print("\n[cyan]测试结果：[/cyan]")
        for api in apis_to_test:
            success, msg = await config_manager.test_api(api.name)
            status = "[green]✓[/green]" if success else "[red]✗[/red]"
            console.print(f"  {status} {api.name}: {msg}")

    asyncio.run(run_tests())
    Prompt.ask("\n按回车继续")


# ===== Entry Point =====

def main():
    """主入口"""
    if len(sys.argv) == 1:
        # 无参数时尝试启动界面
        try:
            # 优先尝试TUI
            from .tui import run_tui
            run_tui()
        except Exception as e:
            # TUI失败则回退到文本菜单
            console.print(f"[yellow]TUI启动失败 ({e})，使用文本菜单[/yellow]")
            run_interactive_menu()
    else:
        cli()


if __name__ == "__main__":
    main()
