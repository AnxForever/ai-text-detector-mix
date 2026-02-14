#!/usr/bin/env python3
"""
API管理工具 - 可视化TUI界面

使用 Textual 框架构建的交互式终端界面
"""

import asyncio
import re
from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.screen import Screen, ModalScreen
from textual.widgets import (
    Header, Footer, Static, Button, Input, Label,
    DataTable, TabbedContent, TabPane, Select, TextArea,
    Checkbox, ProgressBar, ListView, ListItem, Markdown,
    Rule, Switch
)
from textual.message import Message
from textual import on
from rich.text import Text
from rich.panel import Panel

from .config import APIConfigManager, APIConfig
from .models import ModelManager
from .generator import DataGenerator, GenerationConfig, GenerationTask


# ==================== 数据集类型配置 ====================

DATASET_TYPES = {
    "general": {
        "name": "通用文章",
        "template": "请以「{topic}」为主题，写一段200-300字的文章。",
        "system_prompt": "你是一个创意写作助手，请根据要求生成自然流畅的中文文本。"
    },
    "news": {
        "name": "新闻报道",
        "template": "请以新闻报道的风格，写一篇关于「{topic}」的新闻稿，200-300字。",
        "system_prompt": "你是一名专业的新闻记者，请用客观、简洁的新闻语言撰写报道。"
    },
    "academic": {
        "name": "学术论文",
        "template": "请以学术论文的风格，写一段关于「{topic}」的论述，200-300字。",
        "system_prompt": "你是一名学术研究者，请用严谨、专业的学术语言撰写内容。"
    },
    "social_media": {
        "name": "社交媒体",
        "template": "请写一条关于「{topic}」的社交媒体帖子，100-150字，风格轻松活泼。",
        "system_prompt": "你是一个社交媒体博主，请用轻松、有趣的语言撰写内容。"
    },
    "review": {
        "name": "产品评测",
        "template": "请写一段关于「{topic}」的产品评测，200-300字，包含优缺点分析。",
        "system_prompt": "你是一名产品评测师，请客观分析产品的优缺点。"
    },
    "story": {
        "name": "故事创作",
        "template": "请以「{topic}」为主题，创作一个短篇故事，300-500字。",
        "system_prompt": "你是一位创意作家，请创作引人入胜的故事。"
    },
    "tutorial": {
        "name": "教程指南",
        "template": "请写一篇关于「{topic}」的教程指南，200-300字，步骤清晰。",
        "system_prompt": "你是一位技术写作专家，请用清晰易懂的语言撰写教程。"
    },
    "custom": {
        "name": "自定义",
        "template": "{topic}",
        "system_prompt": "你是一个AI助手。"
    }
}


# ==================== 模型提供商分类 ====================

def classify_model_provider(model_id: str) -> str:
    """根据模型ID判断提供商"""
    model_lower = model_id.lower()

    if any(x in model_lower for x in ["gpt", "o1", "o3", "chatgpt", "davinci", "turbo"]):
        return "OpenAI"
    elif any(x in model_lower for x in ["claude", "anthropic"]):
        return "Anthropic"
    elif any(x in model_lower for x in ["gemini", "palm", "bard"]):
        return "Google"
    elif any(x in model_lower for x in ["deepseek"]):
        return "DeepSeek"
    elif any(x in model_lower for x in ["qwen", "tongyi"]):
        return "阿里云"
    elif any(x in model_lower for x in ["glm", "chatglm", "zhipu"]):
        return "智谱AI"
    elif any(x in model_lower for x in ["ernie", "wenxin"]):
        return "百度"
    elif any(x in model_lower for x in ["llama", "meta"]):
        return "Meta"
    elif any(x in model_lower for x in ["mistral", "mixtral"]):
        return "Mistral"
    elif any(x in model_lower for x in ["yi-"]):
        return "零一万物"
    elif any(x in model_lower for x in ["moonshot", "kimi"]):
        return "月之暗面"
    elif any(x in model_lower for x in ["spark"]):
        return "讯飞"
    else:
        return "其他"


# ==================== 弹出对话框 ====================

class AddAPIModal(ModalScreen):
    """添加API的弹出对话框"""

    BINDINGS = [
        Binding("escape", "cancel", "取消"),
    ]

    def compose(self) -> ComposeResult:
        with Container(id="add-api-dialog"):
            yield Label("添加新的API配置", id="dialog-title")
            yield Rule()

            with Vertical(id="form-container"):
                yield Label("API名称 *")
                yield Input(placeholder="例如: openai_proxy", id="api-name")

                yield Label("Base URL *")
                yield Input(placeholder="https://api.example.com/v1", id="api-url")

                yield Label("API Key *")
                yield Input(placeholder="sk-xxx", password=True, id="api-key")

                yield Label("环境变量名 (可选，推荐)")
                yield Input(placeholder="例如: OPENAI_API_KEY", id="api-key-env")

                yield Label("每分钟请求限制")
                yield Input(placeholder="10", value="10", id="api-rate-limit")

            yield Rule()
            with Horizontal(id="dialog-buttons"):
                yield Button("保存", variant="primary", id="btn-save")
                yield Button("取消", variant="default", id="btn-cancel")

    @on(Button.Pressed, "#btn-save")
    def save_api(self) -> None:
        name = self.query_one("#api-name", Input).value.strip()
        url = self.query_one("#api-url", Input).value.strip()
        key = self.query_one("#api-key", Input).value.strip()
        key_env = self.query_one("#api-key-env", Input).value.strip()
        rate_limit_str = self.query_one("#api-rate-limit", Input).value.strip()

        # 验证
        if not name or not url:
            self.notify("请填写API名称和URL", severity="error")
            return

        if not key and not key_env:
            self.notify("请填写API Key或环境变量名", severity="error")
            return

        try:
            rate_limit = int(rate_limit_str) if rate_limit_str else 10
        except ValueError:
            rate_limit = 10

        self.dismiss({
            "name": name,
            "url": url,
            "key": key,
            "key_env": key_env,
            "rate_limit": rate_limit
        })

    @on(Button.Pressed, "#btn-cancel")
    def cancel(self) -> None:
        self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class ConfirmModal(ModalScreen):
    """确认对话框"""

    def __init__(self, message: str, title: str = "确认"):
        super().__init__()
        self._message = message
        self._title = title

    def compose(self) -> ComposeResult:
        with Container(id="confirm-dialog"):
            yield Label(self._title, id="dialog-title")
            yield Rule()
            yield Label(self._message, id="confirm-message")
            yield Rule()
            with Horizontal(id="dialog-buttons"):
                yield Button("确认", variant="error", id="btn-confirm")
                yield Button("取消", variant="default", id="btn-cancel")

    @on(Button.Pressed, "#btn-confirm")
    def confirm(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#btn-cancel")
    def cancel(self) -> None:
        self.dismiss(False)


# ==================== 主应用 ====================

class APIManagerApp(App):
    """API管理工具主应用"""

    CSS = """
    /* 全局样式 */
    Screen {
        background: $surface;
    }

    /* 标签页内容 */
    TabPane {
        padding: 1;
    }

    /* 表格样式 */
    DataTable {
        height: auto;
        max-height: 15;
        margin-bottom: 1;
    }

    /* 按钮组 */
    .button-group {
        height: auto;
        margin: 1 0;
    }

    .button-group Button {
        margin-right: 1;
    }

    /* 表单样式 */
    .form-group {
        height: auto;
        margin-bottom: 1;
    }

    .form-group Label {
        margin-bottom: 0;
    }

    .form-group Input, .form-group Select, .form-group TextArea {
        margin-top: 0;
    }

    /* 对话框样式 */
    #add-api-dialog, #confirm-dialog {
        width: 60;
        height: auto;
        max-height: 80%;
        padding: 1 2;
        background: $panel;
        border: tall $primary;
    }

    #dialog-title {
        text-style: bold;
        color: $text;
        margin-bottom: 1;
    }

    #form-container {
        height: auto;
    }

    #form-container Label {
        margin-top: 1;
    }

    #dialog-buttons {
        height: auto;
        margin-top: 1;
        align: center middle;
    }

    #dialog-buttons Button {
        margin: 0 1;
    }

    /* 模型列表样式 */
    #model-tree {
        height: 100%;
        border: solid $primary;
    }

    .provider-section {
        margin-bottom: 1;
        padding: 1;
        border: solid $secondary;
    }

    .provider-title {
        text-style: bold;
        color: $primary;
    }

    /* 生成配置区域 */
    #generation-config {
        height: auto;
    }

    #topic-input {
        height: 3;
    }

    #custom-template {
        height: 5;
    }

    /* 进度区域 */
    #progress-section {
        height: auto;
        margin-top: 1;
        padding: 1;
        border: solid $secondary;
    }

    /* 状态信息 */
    #status-bar {
        height: 3;
        padding: 1;
        background: $boost;
    }

    /* 选中模型列表 */
    #selected-models {
        height: auto;
        max-height: 8;
        border: solid $accent;
        padding: 1;
        margin: 1 0;
    }

    #selected-models-title {
        text-style: bold;
        color: $accent;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "退出"),
        Binding("r", "refresh", "刷新"),
        Binding("ctrl+s", "save", "保存"),
    ]

    def __init__(self):
        super().__init__()
        self.config_manager = APIConfigManager()
        self.model_manager = ModelManager(self.config_manager)
        self.generator = DataGenerator(self.config_manager, self.model_manager)
        self.selected_models: list[tuple[str, str]] = []  # [(api_name, model_id), ...]
        self.all_models: dict[str, list[str]] = {}

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent():
            with TabPane("API配置", id="tab-api"):
                yield from self._compose_api_tab()
            with TabPane("模型选择", id="tab-models"):
                yield from self._compose_models_tab()
            with TabPane("数据生成", id="tab-generate"):
                yield from self._compose_generate_tab()
        yield Footer()

    def _compose_api_tab(self) -> ComposeResult:
        """API配置标签页"""
        with Vertical():
            yield Static("已配置的API列表", classes="section-title")
            yield DataTable(id="api-table")
            with Horizontal(classes="button-group"):
                yield Button("添加API", variant="primary", id="btn-add-api")
                yield Button("删除选中", variant="error", id="btn-delete-api")
                yield Button("测试连接", variant="success", id="btn-test-api")
                yield Button("刷新模型", variant="default", id="btn-refresh-models")

    def _compose_models_tab(self) -> ComposeResult:
        """模型选择标签页"""
        with Horizontal():
            with Vertical(id="model-list-section"):
                yield Static("可用模型 (按厂商分类)", classes="section-title")
                yield Static("请先在API配置页面添加API并刷新模型", id="model-placeholder")
                with ScrollableContainer(id="model-tree"):
                    yield Static("")  # 占位

            with Vertical(id="selected-section"):
                yield Static("已选择的模型", id="selected-models-title")
                with ScrollableContainer(id="selected-models"):
                    yield Static("暂无选择", id="no-selection")
                yield Button("清空选择", variant="warning", id="btn-clear-selection")

    def _compose_generate_tab(self) -> ComposeResult:
        """数据生成标签页"""
        with Vertical(id="generation-config"):
            # 数据集类型选择
            with Vertical(classes="form-group"):
                yield Label("数据集类型")
                yield Select(
                    [(v["name"], k) for k, v in DATASET_TYPES.items()],
                    id="dataset-type",
                    value="general"
                )

            # 主题输入
            with Vertical(classes="form-group"):
                yield Label("生成主题 (每行一个)")
                yield TextArea(
                    "人工智能\n科技发展\n环境保护",
                    id="topic-input"
                )

            # 自定义模板
            with Vertical(classes="form-group"):
                yield Label("提示词模板 (可选，{topic}为主题占位符)")
                yield TextArea(id="custom-template")

            # 生成数量
            with Horizontal(classes="form-group"):
                with Vertical():
                    yield Label("每主题数量")
                    yield Input(value="5", id="count-per-topic")
                with Vertical():
                    yield Label("并发数")
                    yield Input(value="5", id="concurrent")

            # 输出设置
            with Vertical(classes="form-group"):
                yield Label("输出文件路径")
                yield Input(value="output/generated_data.jsonl", id="output-path")

            # 生成按钮
            with Horizontal(classes="button-group"):
                yield Button("开始生成", variant="success", id="btn-start-generate")
                yield Button("停止", variant="error", id="btn-stop-generate", disabled=True)

            # 进度显示
            with Vertical(id="progress-section"):
                yield Static("生成进度", classes="section-title")
                yield ProgressBar(id="gen-progress", show_eta=True)
                yield Static("等待开始...", id="gen-status")

    async def on_mount(self) -> None:
        """应用启动时初始化"""
        self._refresh_api_table()

    def _refresh_api_table(self) -> None:
        """刷新API表格"""
        table = self.query_one("#api-table", DataTable)
        table.clear(columns=True)

        table.add_columns("名称", "Base URL", "状态", "限速", "模型数")
        table.cursor_type = "row"

        for api in self.config_manager.list_apis():
            status = "✅ 启用" if api.enabled else "❌ 禁用"
            table.add_row(
                api.name,
                api.base_url[:40] + "..." if len(api.base_url) > 40 else api.base_url,
                status,
                f"{api.rate_limit}/min",
                str(len(api.models_cache)),
                key=api.name
            )

    def _refresh_model_list(self) -> None:
        """刷新模型列表"""
        container = self.query_one("#model-tree", ScrollableContainer)
        container.remove_children()

        placeholder = self.query_one("#model-placeholder", Static)

        if not self.all_models:
            placeholder.update("请先在API配置页面添加API并点击'刷新模型'")
            placeholder.display = True
            return

        placeholder.display = False

        # 按厂商分组
        providers: dict[str, list[tuple[str, str, str]]] = {}  # provider -> [(api, model, display), ...]

        for api_name, models in self.all_models.items():
            for model in models:
                provider = classify_model_provider(model)
                if provider not in providers:
                    providers[provider] = []
                providers[provider].append((api_name, model, f"{model} ({api_name})"))

        # 按厂商名排序，创建组件并直接挂载
        for provider in sorted(providers.keys()):
            models = providers[provider]

            # 挂载厂商标题
            container.mount(Static(f"📦 {provider} ({len(models)}个模型)", classes="provider-title"))

            # 挂载该厂商的所有模型复选框
            for api_name, model_id, display in sorted(models, key=lambda x: x[1]):
                is_selected = (api_name, model_id) in self.selected_models
                cb = Checkbox(display, value=is_selected)
                cb.api_name = api_name  # type: ignore
                cb.model_id = model_id  # type: ignore
                container.mount(cb)

            # 添加分隔
            container.mount(Static(""))

    def _refresh_selected_models(self) -> None:
        """刷新已选择模型显示"""
        container = self.query_one("#selected-models", ScrollableContainer)
        container.remove_children()

        if not self.selected_models:
            container.mount(Static("暂无选择", id="no-selection"))
        else:
            for api_name, model_id in self.selected_models:
                container.mount(Static(f"• {model_id} ({api_name})"))

    @on(Button.Pressed, "#btn-add-api")
    def show_add_api_dialog(self) -> None:
        """显示添加API对话框"""
        def handle_result(result: Optional[dict]) -> None:
            if result:
                self.config_manager.add_api(
                    name=result["name"],
                    base_url=result["url"],
                    api_key=result["key"],
                    api_key_env=result["key_env"],
                    rate_limit=result["rate_limit"]
                )
                self._refresh_api_table()
                self.notify(f"已添加API: {result['name']}", severity="information")

        self.push_screen(AddAPIModal(), handle_result)

    @on(Button.Pressed, "#btn-delete-api")
    def delete_api(self) -> None:
        """删除选中的API"""
        table = self.query_one("#api-table", DataTable)
        if table.cursor_row is None:
            self.notify("请先选择要删除的API", severity="warning")
            return

        row_key = table.get_row_at(table.cursor_row)
        api_name = str(table.get_cell_at((table.cursor_row, 0)))

        def handle_confirm(confirmed: bool) -> None:
            if confirmed:
                self.config_manager.remove_api(api_name)
                self._refresh_api_table()
                self.notify(f"已删除API: {api_name}", severity="information")

        self.push_screen(
            ConfirmModal(f"确定要删除API '{api_name}' 吗？"),
            handle_confirm
        )

    @on(Button.Pressed, "#btn-test-api")
    async def test_api(self) -> None:
        """测试API连接"""
        table = self.query_one("#api-table", DataTable)
        if table.cursor_row is None:
            self.notify("请先选择要测试的API", severity="warning")
            return

        api_name = str(table.get_cell_at((table.cursor_row, 0)))
        self.notify(f"正在测试 {api_name}...", severity="information")

        success, msg = await self.config_manager.test_api(api_name)
        if success:
            self.notify(f"✅ {api_name}: {msg}", severity="information")
        else:
            self.notify(f"❌ {api_name}: {msg}", severity="error")

    @on(Button.Pressed, "#btn-refresh-models")
    async def refresh_models(self) -> None:
        """刷新所有API的模型列表"""
        self.notify("正在刷新模型列表...", severity="information")

        self.all_models = await self.model_manager.fetch_all_models(force_refresh=True)

        self._refresh_api_table()
        self._refresh_model_list()

        total = sum(len(m) for m in self.all_models.values())
        self.notify(f"已刷新 {len(self.all_models)} 个API，共 {total} 个模型", severity="information")

    @on(Checkbox.Changed)
    def on_model_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """处理模型选择变化"""
        checkbox = event.checkbox
        if not hasattr(checkbox, "api_name"):
            return

        api_name = checkbox.api_name  # type: ignore
        model_id = checkbox.model_id  # type: ignore

        if event.value:
            if (api_name, model_id) not in self.selected_models:
                self.selected_models.append((api_name, model_id))
        else:
            if (api_name, model_id) in self.selected_models:
                self.selected_models.remove((api_name, model_id))

        self._refresh_selected_models()

    @on(Button.Pressed, "#btn-clear-selection")
    def clear_model_selection(self) -> None:
        """清空模型选择"""
        self.selected_models.clear()
        self._refresh_model_list()
        self._refresh_selected_models()
        self.notify("已清空模型选择", severity="information")

    @on(Select.Changed, "#dataset-type")
    def on_dataset_type_changed(self, event: Select.Changed) -> None:
        """数据集类型变化时更新模板"""
        dataset_type = str(event.value)
        if dataset_type in DATASET_TYPES:
            template_area = self.query_one("#custom-template", TextArea)
            template_area.text = DATASET_TYPES[dataset_type]["template"]

    @on(Button.Pressed, "#btn-start-generate")
    async def start_generation(self) -> None:
        """开始数据生成"""
        # 验证模型选择
        if not self.selected_models:
            self.notify("请先在'模型选择'页面选择要使用的模型", severity="error")
            return

        # 获取配置
        dataset_type = str(self.query_one("#dataset-type", Select).value)
        topics_text = self.query_one("#topic-input", TextArea).text
        custom_template = self.query_one("#custom-template", TextArea).text
        count_str = self.query_one("#count-per-topic", Input).value
        concurrent_str = self.query_one("#concurrent", Input).value
        output_path = self.query_one("#output-path", Input).value

        # 解析主题
        topics = [t.strip() for t in topics_text.strip().split("\n") if t.strip()]
        if not topics:
            self.notify("请输入至少一个主题", severity="error")
            return

        try:
            count_per_topic = int(count_str)
            concurrent = int(concurrent_str)
        except ValueError:
            self.notify("数量和并发数必须是整数", severity="error")
            return

        # 确定模板和系统提示
        if custom_template.strip():
            template = custom_template
        else:
            template = DATASET_TYPES.get(dataset_type, DATASET_TYPES["general"])["template"]

        system_prompt = DATASET_TYPES.get(dataset_type, DATASET_TYPES["general"])["system_prompt"]

        # 准备API和模型配对
        api_models = []
        for api_name, model_id in self.selected_models:
            api = self.config_manager.get_api(api_name)
            if api:
                api_models.append((api, model_id))

        if not api_models:
            self.notify("选中的API不可用", severity="error")
            return

        # 禁用开始按钮，启用停止按钮
        self.query_one("#btn-start-generate", Button).disabled = True
        self.query_one("#btn-stop-generate", Button).disabled = False

        # 创建生成任务
        prompts = self.generator.create_prompts(topics, template, count_per_topic)
        self.generator.distribute_tasks(prompts, api_models)

        total_tasks = len(self.generator.tasks)
        progress_bar = self.query_one("#gen-progress", ProgressBar)
        status_label = self.query_one("#gen-status", Static)

        progress_bar.update(total=total_tasks, progress=0)
        status_label.update(f"正在生成 0/{total_tasks}...")

        # 执行生成
        config = GenerationConfig(
            system_prompt=system_prompt,
            save_interval=5
        )

        def on_progress(stats: dict) -> None:
            completed = stats["success"] + stats["failed"]
            progress_bar.update(progress=completed)
            status_label.update(
                f"进度: {completed}/{total_tasks} | "
                f"成功: {stats['success']} | 失败: {stats['failed']}"
            )

        try:
            await self.generator.run_generation(
                config=config,
                output_path=Path(output_path),
                max_concurrent=concurrent,
                on_progress=on_progress
            )

            stats = self.generator.stats
            self.notify(
                f"生成完成！成功: {stats['success']}, 失败: {stats['failed']}",
                severity="information"
            )
            status_label.update(
                f"✅ 完成！成功: {stats['success']}, 失败: {stats['failed']} | "
                f"已保存到: {output_path}"
            )

        except Exception as e:
            self.notify(f"生成出错: {e}", severity="error")
            status_label.update(f"❌ 出错: {e}")

        finally:
            self.query_one("#btn-start-generate", Button).disabled = False
            self.query_one("#btn-stop-generate", Button).disabled = True

    def action_refresh(self) -> None:
        """刷新操作"""
        self._refresh_api_table()
        self.notify("已刷新", severity="information")

    def action_save(self) -> None:
        """保存配置"""
        self.config_manager.save()
        self.notify("配置已保存", severity="information")


def run_tui():
    """运行TUI应用"""
    app = APIManagerApp()
    app.run()


if __name__ == "__main__":
    run_tui()
