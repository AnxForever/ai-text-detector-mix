"""
API Manager - 多API中转站管理与批量数据生成工具

功能：
1. API配置管理（增删改查）
2. 模型查询（查看各API可用模型）
3. 批量数据生成（多API并行）
4. 可视化TUI界面
"""

from .config import APIConfigManager
from .models import ModelManager
from .generator import DataGenerator

__version__ = "1.0.0"
__all__ = ["APIConfigManager", "ModelManager", "DataGenerator"]


def run_tui():
    """启动可视化TUI界面"""
    from .tui import run_tui as _run_tui
    _run_tui()
