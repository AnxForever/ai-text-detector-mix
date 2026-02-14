#!/usr/bin/env python3
"""
API管理工具启动脚本 - 直接运行此文件
"""
import sys
import os

# 确保项目根目录在路径中
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, IntPrompt

console = Console()


def simple_menu():
    """简化版菜单，兼容性更好"""
    from tools.api_manager.config import APIConfigManager
    from tools.api_manager.models import ModelManager
    from tools.api_manager.generator import DataGenerator, GenerationConfig
    import asyncio
    from pathlib import Path

    config_manager = APIConfigManager()
    model_manager = ModelManager(config_manager)

    while True:
        console.print("\n")
        console.print(Panel(
            "[bold cyan]🤖 AI数据生成工具[/bold cyan]\n\n"
            "[1] 查看已配置的API\n"
            "[2] 添加新API\n"
            "[3] 删除API\n"
            "[4] 测试API连接\n"
            "[5] 查看可用模型\n"
            "[6] 生成数据\n"
            "[0] 退出",
            title="主菜单"
        ))

        choice = Prompt.ask("请选择", choices=["0", "1", "2", "3", "4", "5", "6"], default="0")

        if choice == "0":
            console.print("[yellow]再见！[/yellow]")
            break

        elif choice == "1":
            # 查看API列表
            config_manager.display_apis()

        elif choice == "2":
            # 添加API
            console.print("\n[cyan]添加新API[/cyan]")
            name = Prompt.ask("API名称")
            url = Prompt.ask("Base URL")

            use_env = Confirm.ask("使用环境变量存储密钥？", default=False)
            if use_env:
                key_env = Prompt.ask("环境变量名称")
                config_manager.add_api(name, url, api_key_env=key_env)
            else:
                key = Prompt.ask("API Key", password=True)
                config_manager.add_api(name, url, api_key=key)

        elif choice == "3":
            # 删除API
            config_manager.display_apis()
            name = Prompt.ask("要删除的API名称")
            if Confirm.ask(f"确定删除 '{name}'?"):
                config_manager.remove_api(name)

        elif choice == "4":
            # 测试连接
            async def test_all():
                for api in config_manager.list_apis():
                    console.print(f"测试 {api.name}...")
                    success, msg = await config_manager.test_api(api.name)
                    status = "[green]✓[/green]" if success else "[red]✗[/red]"
                    console.print(f"  {status} {msg}")

            if config_manager.list_apis():
                asyncio.run(test_all())
            else:
                console.print("[yellow]暂无API配置[/yellow]")

        elif choice == "5":
            # 查看模型
            async def fetch_models():
                console.print("正在获取模型列表...")
                all_models = await model_manager.fetch_all_models(force_refresh=True)
                model_manager.display_all_models(all_models)

            if config_manager.list_apis():
                asyncio.run(fetch_models())
            else:
                console.print("[yellow]请先添加API配置[/yellow]")

        elif choice == "6":
            # 生成数据
            if not config_manager.get_enabled_apis():
                console.print("[red]请先添加并启用API[/red]")
                continue

            console.print("\n[cyan]数据生成配置[/cyan]")

            # 获取模型
            async def get_models():
                return await model_manager.fetch_all_models()
            all_models = asyncio.run(get_models())

            if not all_models:
                console.print("[red]没有可用模型，请先刷新模型列表[/red]")
                continue

            # 显示可用模型
            console.print("\n可用模型:")
            model_list = []
            for api_name, models in all_models.items():
                for m in models[:5]:  # 只显示前5个
                    model_list.append((api_name, m))
                    console.print(f"  [{len(model_list)}] {m} ({api_name})")

            # 选择主题
            topics_input = Prompt.ask(
                "生成主题 (逗号分隔)",
                default="人工智能,科技发展,环境保护"
            )
            topics = [t.strip() for t in topics_input.split(",")]

            count = IntPrompt.ask("每主题生成数量", default=5)
            output = Prompt.ask("输出文件", default="output/generated.jsonl")

            # 开始生成
            if Confirm.ask("开始生成?"):
                generator = DataGenerator(config_manager, model_manager)
                api_models = model_manager.get_model_for_generation()

                if not api_models:
                    console.print("[red]没有可用的API和模型[/red]")
                    continue

                template = "请以「{topic}」为主题，写一段200-300字的文章。"
                prompts = generator.create_prompts(topics, template, count)
                generator.distribute_tasks(prompts, api_models)

                console.print(f"[green]创建了 {len(generator.tasks)} 个任务[/green]")

                async def run_gen():
                    config = GenerationConfig()
                    await generator.run_generation(
                        config=config,
                        output_path=Path(output),
                        max_concurrent=5
                    )
                    generator.display_stats()
                    console.print(f"[green]已保存到: {output}[/green]")

                asyncio.run(run_gen())

        input("\n按回车继续...")


if __name__ == "__main__":
    try:
        simple_menu()
    except KeyboardInterrupt:
        console.print("\n[yellow]已取消[/yellow]")
    except Exception as e:
        console.print(f"[red]错误: {e}[/red]")
        import traceback
        traceback.print_exc()
