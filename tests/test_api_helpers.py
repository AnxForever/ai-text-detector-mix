"""Tests for API helper functions that do NOT require model loading.

These tests cover pure logic in api/api.py — sentence splitting, domain
inference, risk-flag collection, rate-limiting, auth, and IP extraction.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from api.api import (
    build_reason_analysis,
    collect_risk_flags,
    ensure_detector_loaded,
    get_client_ip,
    infer_domain_hint,
    resolve_api_key,
    run_project_agent_live_detection,
    split_sentences,
    verify_internal_token,
)


# =========================================================================
# split_sentences
# =========================================================================
class TestSplitSentences:
    """split_sentences: Chinese punctuation-based sentence segmentation."""

    def test_single_sentence_no_punct(self):
        result = split_sentences("没有标点的句子")
        assert result == ["没有标点的句子"]

    def test_multiple_sentences(self):
        result = split_sentences("第一句。第二句！第三句？")
        assert len(result) == 3
        assert result[0] == "第一句。"
        assert result[1] == "第二句！"
        assert result[2] == "第三句？"

    def test_mixed_punctuation(self):
        result = split_sentences("你好！你在干嘛？我在写代码。")
        assert len(result) == 3

    def test_empty_string(self):
        result = split_sentences("")
        assert result == []

    def test_whitespace_only(self):
        result = split_sentences("   ")
        assert result == []

    def test_english_punctuation(self):
        result = split_sentences("Hello! How are you? Fine.")
        # English ! and ? are in the split pattern
        assert len(result) >= 2

    def test_preserves_content(self):
        text = "AI技术很强大。但也有风险。"
        result = split_sentences(text)
        joined = "".join(result)
        assert joined == text


# =========================================================================
# infer_domain_hint
# =========================================================================
class TestInferDomainHint:
    """infer_domain_hint: regex-based domain classification."""

    def test_formal_text(self):
        assert infer_domain_hint("关于开展安全检查的通知") == "formal"
        assert infer_domain_hint("特此公告") == "formal"

    def test_technical_text(self):
        assert infer_domain_hint("这个神经网络模型的训练过程") == "technical"
        assert infer_domain_hint("部署API服务到生产环境") == "technical"

    def test_casual_text(self):
        assert infer_domain_hint("哈哈这也太搞笑了") == "casual"
        assert infer_domain_hint("我觉得还行吧") == "casual"

    def test_general_text(self):
        assert infer_domain_hint("今天是星期三") == "general"

    def test_empty_text(self):
        assert infer_domain_hint("") == "general"


# =========================================================================
# collect_risk_flags
# =========================================================================
class TestCollectRiskFlags:
    """collect_risk_flags: heuristic risk detection."""

    def test_short_text_flag(self):
        flags = collect_risk_flags(
            "短", confidence=90, boundary_sentence_index=None, result_type="human"
        )
        assert "short_text" in flags

    def test_long_text_flag(self):
        text = "x" * 2049
        flags = collect_risk_flags(
            text, confidence=90, boundary_sentence_index=None, result_type="human"
        )
        assert "long_text" in flags

    def test_extreme_length_flag(self):
        text = "x" * 5001
        flags = collect_risk_flags(
            text, confidence=90, boundary_sentence_index=None, result_type="human"
        )
        assert "extreme_length" in flags
        assert "long_text" in flags

    def test_low_confidence_flag(self):
        flags = collect_risk_flags(
            "一些文本", confidence=60, boundary_sentence_index=None, result_type="human"
        )
        assert "low_confidence" in flags

    def test_template_like_flag(self):
        flags = collect_risk_flags(
            "分析请求的内容", confidence=90, boundary_sentence_index=None, result_type="human"
        )
        assert "template_like" in flags

    def test_mixed_without_boundary(self):
        flags = collect_risk_flags(
            "一些文本内容", confidence=90, boundary_sentence_index=None, result_type="mixed"
        )
        assert "mixed_without_boundary" in flags

    def test_no_flags_normal_text(self):
        text = "这是一段正常长度的文本，" * 15  # >128 chars to avoid short_text flag
        flags = collect_risk_flags(
            text, confidence=90, boundary_sentence_index=0, result_type="human"
        )
        assert flags == []


# =========================================================================
# build_reason_analysis
# =========================================================================
class TestBuildReasonAnalysis:
    """build_reason_analysis: human-readable rationale generation."""

    def test_ai_summary_mentions_score_advantage(self):
        summary, signals = build_reason_analysis(
            result_type="ai",
            confidence=96,
            ai_percentage=95,
            human_percentage=5,
            boundary_sentence_index=None,
            domain_hint="general",
            risk_flags=[],
        )
        assert "AI 生成" in summary
        assert any("AI倾向明显高于人类倾向" in signal for signal in signals)
        assert any("未发现明确的混合边界" in signal for signal in signals)

    def test_mixed_summary_mentions_boundary(self):
        summary, signals = build_reason_analysis(
            result_type="mixed",
            confidence=72,
            ai_percentage=58,
            human_percentage=42,
            boundary_sentence_index=2,
            domain_hint="technical",
            risk_flags=["low_confidence"],
        )
        assert "混合文本" in summary
        assert "第 3 句附近" in summary
        assert any("风格切换" in signal for signal in signals)
        assert any("技术术语" in signal for signal in signals)
        assert any("低置信区间" in signal for signal in signals)

    def test_short_text_summary_adds_caution(self):
        summary, signals = build_reason_analysis(
            result_type="human",
            confidence=88,
            ai_percentage=12,
            human_percentage=88,
            boundary_sentence_index=None,
            domain_hint="casual",
            risk_flags=["short_text"],
        )
        assert "文本较短" in summary
        assert any("口语化表达" in signal for signal in signals)


# =========================================================================
# get_client_ip
# =========================================================================
class TestGetClientIp:
    """get_client_ip: extract IP from request."""

    def test_from_x_forwarded_for(self):
        # Policy: rightmost XFF entry is the last trusted proxy (spoof-resistant).
        request = MagicMock()
        request.headers = {"x-forwarded-for": "1.2.3.4, 5.6.7.8"}
        assert get_client_ip(request) == "5.6.7.8"

    def test_prefers_x_real_ip_over_xff(self):
        request = MagicMock()
        request.headers = {"x-real-ip": "9.9.9.9", "x-forwarded-for": "1.2.3.4, 5.6.7.8"}
        assert get_client_ip(request) == "9.9.9.9"

    def test_from_client_host(self):
        request = MagicMock()
        request.headers = {}
        request.client.host = "10.0.0.1"
        assert get_client_ip(request) == "10.0.0.1"

    def test_unknown_when_no_info(self):
        request = MagicMock()
        request.headers = {}
        request.client = None
        assert get_client_ip(request) == "unknown"


# =========================================================================
# verify_internal_token
# =========================================================================
class TestVerifyInternalToken:
    """verify_internal_token: API auth check."""

    def test_passes_when_enforcement_disabled(self, monkeypatch):
        monkeypatch.setattr("api.api.ENFORCE_INTERNAL_TOKEN", False)
        # Should not raise
        verify_internal_token(None)

    def test_raises_500_when_no_token_configured(self, monkeypatch):
        monkeypatch.setattr("api.api.ENFORCE_INTERNAL_TOKEN", True)
        monkeypatch.setattr("api.api.INTERNAL_API_TOKEN", "")
        with pytest.raises(HTTPException) as exc_info:
            verify_internal_token("some-token")
        assert exc_info.value.status_code == 500

    def test_raises_401_when_wrong_token(self, monkeypatch):
        monkeypatch.setattr("api.api.ENFORCE_INTERNAL_TOKEN", True)
        monkeypatch.setattr("api.api.INTERNAL_API_TOKEN", "correct-token")
        with pytest.raises(HTTPException) as exc_info:
            verify_internal_token("wrong-token")
        assert exc_info.value.status_code == 401

    def test_passes_with_correct_token(self, monkeypatch):
        monkeypatch.setattr("api.api.ENFORCE_INTERNAL_TOKEN", True)
        monkeypatch.setattr("api.api.INTERNAL_API_TOKEN", "correct-token")
        verify_internal_token("correct-token")

    def test_raises_401_when_no_header(self, monkeypatch):
        monkeypatch.setattr("api.api.ENFORCE_INTERNAL_TOKEN", True)
        monkeypatch.setattr("api.api.INTERNAL_API_TOKEN", "correct-token")
        with pytest.raises(HTTPException) as exc_info:
            verify_internal_token(None)
        assert exc_info.value.status_code == 401


# =========================================================================
# project agent live detector
# =========================================================================
class TestProjectAgentLiveDetector:
    """Lazy detector loading for project copilot text analysis."""

    def test_ensure_detector_loaded_returns_existing_detector(self, monkeypatch):
        sentinel = object()
        monkeypatch.setattr("api.api.detector", sentinel)
        assert ensure_detector_loaded() is sentinel

    def test_run_project_agent_live_detection_can_lazy_init(self, monkeypatch):
        class FakeDetector:
            def classify(self, text):
                return {
                    "label": "AI",
                    "confidence": 0.91,
                    "prob_ai": 0.91,
                    "prob_human": 0.09,
                }

            def detect_boundary(self, text):
                return {"boundary_char": 12}

        monkeypatch.setattr("api.api.detector", None)
        monkeypatch.setattr("api.api.HybridTextDetector", lambda: FakeDetector())
        monkeypatch.setattr("api.api.SPAN_TRIGGER_MIN_CHARS", 0)

        summary = run_project_agent_live_detection("前半段像人写的，后半段更像AI。")

        assert summary is not None
        assert "实时检测结果" in summary
        assert "边界字符位置" in summary


# =========================================================================
# resolve_api_key
# =========================================================================
class TestResolveApiKey:
    """resolve_api_key: resolve key from env or header."""

    def test_env_key_takes_priority(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        assert resolve_api_key("Bearer header-key") == "env-key"

    def test_bearer_header_fallback(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert resolve_api_key("Bearer my-key") == "my-key"

    def test_raw_header_fallback(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert resolve_api_key("raw-key") == "raw-key"

    def test_none_when_nothing(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert resolve_api_key(None) is None

    def test_bearer_only_no_token_returns_none(self, monkeypatch):
        """'Bearer ' with no actual token should return None."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert resolve_api_key("Bearer ") is None
