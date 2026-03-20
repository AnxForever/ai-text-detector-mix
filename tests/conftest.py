"""Shared fixtures for the datacollection test suite."""

from __future__ import annotations

import pytest


@pytest.fixture()
def sample_human_text() -> str:
    """Short human-written Chinese text."""
    return "今天天气真不错，我和朋友去公园散步了，感觉心情特别好。"


@pytest.fixture()
def sample_ai_text() -> str:
    """Typical AI-generated Chinese text."""
    return (
        "人工智能技术的发展对社会产生了深远的影响。"
        "首先，在医疗领域，AI辅助诊断系统能够显著提高诊断准确率。"
        "其次，在教育领域，智能化的教学工具为学生提供了个性化的学习体验。"
        "最后，在交通领域，自动驾驶技术正在逐步改变人们的出行方式。"
    )


@pytest.fixture()
def sample_mixed_text() -> str:
    """Mixed text with human intro and AI continuation."""
    return (
        "我觉得这个方案还行吧，但是有几个地方需要改一下。"
        "基于上述分析，本方案从以下三个维度进行了系统性的优化。"
        "第一，在数据预处理阶段，采用了多层级清洗策略。"
        "第二，在模型架构方面，引入了注意力机制以增强特征提取能力。"
    )


@pytest.fixture()
def sample_short_text() -> str:
    """Very short text for edge-case testing."""
    return "你好"
