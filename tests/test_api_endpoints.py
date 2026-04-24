"""Tests for API endpoint functions without loading real models."""

from __future__ import annotations

import asyncio
import json
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from starlette.datastructures import UploadFile
from starlette.requests import Request

import api.api as api_module


def make_request(headers: dict[str, str] | None = None) -> Request:
    """Build a minimal Starlette request for direct endpoint calls."""
    raw_headers = [
        (key.lower().encode("utf-8"), value.encode("utf-8"))
        for key, value in (headers or {}).items()
    ]
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/",
        "headers": raw_headers,
        "client": ("127.0.0.1", 12345),
    }
    return Request(scope)


@pytest.fixture()
def mock_detector():
    """Create a mock HybridTextDetector with sensible defaults."""
    det = MagicMock()

    det.classify.return_value = {
        "label": "AI",
        "confidence": 0.95,
        "prob_human": 0.05,
        "prob_ai": 0.95,
    }

    det.detect_boundary.return_value = {
        "boundary_token": 5,
        "boundary_char": 20,
        "text": "test",
    }

    return det


@pytest.fixture()
def api_context(mock_detector):
    """Install a mock detector and disable auth for direct endpoint tests."""
    original_token_flag = api_module.ENFORCE_INTERNAL_TOKEN
    original_detector = api_module.detector
    api_module.ENFORCE_INTERNAL_TOKEN = False
    api_module.detector = mock_detector

    yield mock_detector

    api_module.detector = original_detector
    api_module.ENFORCE_INTERNAL_TOKEN = original_token_flag
    api_module.RATE_LIMIT_STATE.clear()


class TestHealthEndpoint:
    """GET /api/health"""

    def test_health_returns_ok(self, api_context):
        data = asyncio.run(api_module.health_check())
        assert data["status"] == "ok"
        assert data["detectorReady"] is True
        assert "modelVersion" in data


class TestDetectEndpoint:
    """POST /api/detect"""

    def test_detect_ai_text(self, api_context):
        mock_det = api_context
        mock_det.classify.return_value = {
            "label": "AI",
            "confidence": 0.97,
            "prob_human": 0.03,
            "prob_ai": 0.97,
        }
        mock_det.detect_boundary.return_value = {
            "boundary_token": None,
            "boundary_char": None,
            "text": "AI text",
        }

        response = api_module.detect_text(
            api_module.DetectRequest(text="AI生成的文本，具有典型的结构化特征。"),
            make_request(),
        )

        assert response.type == "ai"
        assert response.confidence > 0
        assert len(response.sentences) > 0
        assert response.processingTime >= 0
        assert response.feedbackRequired is True
        assert response.reasonSummary is not None
        assert response.reasonSignals
        assert any("AI倾向" in signal for signal in response.reasonSignals)

    def test_detect_human_text(self, api_context):
        mock_det = api_context
        mock_det.classify.return_value = {
            "label": "Human",
            "confidence": 0.92,
            "prob_human": 0.92,
            "prob_ai": 0.08,
        }

        response = api_module.detect_text(
            api_module.DetectRequest(text="我今天去超市买了点菜，晚上做个红烧肉。"),
            make_request(),
        )

        assert response.type == "human"
        assert response.humanPercentage > response.aiPercentage

    def test_detect_empty_text_rejected(self):
        with pytest.raises(ValidationError):
            api_module.DetectRequest(text="")

    def test_detect_missing_text_rejected(self):
        with pytest.raises(ValidationError):
            api_module.DetectRequest()

    def test_detect_response_has_sentences(self, api_context):
        mock_det = api_context
        mock_det.classify.return_value = {
            "label": "AI",
            "confidence": 0.90,
            "prob_human": 0.10,
            "prob_ai": 0.90,
        }
        mock_det.detect_boundary.return_value = {
            "boundary_token": 3,
            "boundary_char": 15,
            "text": "test",
        }

        response = api_module.detect_text(
            api_module.DetectRequest(text="第一句话。第二句话。第三句话。"),
            make_request(),
        )

        assert len(response.sentences) == 3

    def test_detect_mixed_with_boundary(self, api_context, monkeypatch):
        mock_det = api_context
        mock_det.classify.return_value = {
            "label": "AI",
            "confidence": 0.65,
            "prob_human": 0.35,
            "prob_ai": 0.65,
        }
        mock_det.detect_boundary.return_value = {
            "boundary_token": 5,
            "boundary_char": 10,
            "text": "test",
        }
        # New policy: span detector is gated by text length. Lower threshold for this unit test.
        monkeypatch.setattr(api_module, "SPAN_TRIGGER_MIN_CHARS", 0)

        response = api_module.detect_text(
            api_module.DetectRequest(text="我觉得还行。AI生成的内容开始了。继续生成。"),
            make_request(),
        )

        assert response.type == "mixed"
        assert response.boundary is not None
        assert "混合文本" in response.reasonSummary
        assert any("风格切换" in signal for signal in response.reasonSignals)

    def test_detect_uses_exact_feedback_override(self, api_context, monkeypatch):
        mock_det = api_context
        monkeypatch.setattr(
            api_module,
            "lookup_feedback_override",
            lambda **kwargs: {
                "confirmed_label": "human",
                "boundary": None,
                "domain_hint": "formal",
                "source": "manual_feedback_exact_match",
            },
        )

        response = api_module.detect_text(
            api_module.DetectRequest(text="这是一段曾经被误判、后来被人工确认的人类文本。"),
            make_request(),
        )

        assert response.type == "human"
        assert response.confidence == 100.0
        assert response.humanPercentage == 100
        assert response.aiPercentage == 0
        assert response.riskFlags == ["feedback_override_exact_match"]
        assert "完全相同样本" in response.reasonSummary
        assert any("exact match" in signal for signal in response.reasonSignals)
        mock_det.classify.assert_not_called()


class TestFeedbackEndpoint:
    """POST /api/feedback"""

    def test_confirm_correct_prediction(self, api_context, tmp_path, monkeypatch):
        original_persist_feedback = api_module.persist_feedback

        def persist_feedback_for_test(**kwargs):
            return original_persist_feedback(**kwargs, output_dir=tmp_path)

        monkeypatch.setattr(api_module, "persist_feedback", persist_feedback_for_test)

        response = api_module.submit_feedback(
            api_module.FeedbackRequest(
                text="这是一段人工文本。",
                predictedType="human",
                confirmedCorrect=True,
                confidence=92.0,
            ),
            make_request(),
        )

        assert response.status == "ok"
        assert response.misclassifiedSaved is False

        assert not (tmp_path / "confirmations.jsonl").exists()
        assert not (tmp_path / "misclassified_samples.jsonl").exists()

    def test_confirm_incorrect_prediction_writes_dataset(
        self,
        api_context,
        tmp_path,
        monkeypatch,
    ):
        original_persist_feedback = api_module.persist_feedback

        def persist_feedback_for_test(**kwargs):
            return original_persist_feedback(**kwargs, output_dir=tmp_path)

        monkeypatch.setattr(api_module, "persist_feedback", persist_feedback_for_test)

        response = api_module.submit_feedback(
            api_module.FeedbackRequest(
                text="关于开展检查的通知。",
                predictedType="ai",
                confirmedCorrect=False,
                confirmedLabel="human",
                tags=["formal", "notice"],
                note="正式通知被误判",
            ),
            make_request(),
        )

        assert response.misclassifiedSaved is True

        correction_rows = [
            json.loads(line)
            for line in (tmp_path / "misclassified_samples.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert len(correction_rows) == 1
        assert correction_rows[0]["confirmed_label"] == "human"
        assert "manual_feedback" in correction_rows[0]["tags"]

    def test_duplicate_incorrect_prediction_is_deduplicated(
        self,
        api_context,
        tmp_path,
        monkeypatch,
    ):
        original_persist_feedback = api_module.persist_feedback

        def persist_feedback_for_test(**kwargs):
            return original_persist_feedback(**kwargs, output_dir=tmp_path)

        monkeypatch.setattr(api_module, "persist_feedback", persist_feedback_for_test)

        first_response = api_module.submit_feedback(
            api_module.FeedbackRequest(
                text="重复提交的误判样本",
                predictedType="ai",
                confirmedCorrect=False,
                confirmedLabel="human",
            ),
            make_request(),
        )
        second_response = api_module.submit_feedback(
            api_module.FeedbackRequest(
                text="  重复提交的误判样本  ",
                predictedType="ai",
                confirmedCorrect=False,
                confirmedLabel="human",
            ),
            make_request(),
        )

        correction_rows = [
            json.loads(line)
            for line in (tmp_path / "misclassified_samples.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]

        assert first_response.misclassifiedSaved is True
        assert second_response.misclassifiedSaved is False
        assert len(correction_rows) == 1

    def test_incorrect_prediction_requires_confirmed_label(self, api_context):
        with pytest.raises(HTTPException) as exc_info:
            api_module.submit_feedback(
                api_module.FeedbackRequest(
                    text="这段文本有误判。",
                    predictedType="ai",
                    confirmedCorrect=False,
                ),
                make_request(),
            )

        assert exc_info.value.status_code == 422

    def test_feedback_storage_failure_returns_json_500(self, api_context, monkeypatch):
        def persist_feedback_for_test(**kwargs):
            raise OSError("permission denied")

        monkeypatch.setattr(api_module, "persist_feedback", persist_feedback_for_test)

        with pytest.raises(HTTPException) as exc_info:
            api_module.submit_feedback(
                api_module.FeedbackRequest(
                    text="人工确认写入失败。",
                    predictedType="human",
                    confirmedCorrect=True,
                ),
                make_request(),
            )

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Feedback storage unavailable"

    def test_feedback_unexpected_failure_returns_json_500(self, api_context, monkeypatch):
        def persist_feedback_for_test(**kwargs):
            raise RuntimeError("unexpected failure")

        monkeypatch.setattr(api_module, "persist_feedback", persist_feedback_for_test)

        with pytest.raises(HTTPException) as exc_info:
            api_module.submit_feedback(
                api_module.FeedbackRequest(
                    text="人工确认未知异常。",
                    predictedType="human",
                    confirmedCorrect=True,
                ),
                make_request(),
            )

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Feedback submission failed"


class TestChatEndpoint:
    """POST /v1/chat/completions"""

    def test_chat_no_api_key_returns_500(self, api_context, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(
                api_module.chat_completions(
                    api_module.ChatRequest(messages=[{"role": "user", "content": "hello"}]),
                    make_request(),
                    authorization=None,
                    x_internal_token=None,
                )
            )

        assert exc_info.value.status_code == 500

    def test_chat_empty_messages_rejected(self):
        with pytest.raises(ValidationError):
            api_module.ChatRequest(messages=[])


class TestProjectQAEndpoint:
    """POST /api/project-qa"""

    def test_project_qa_returns_extractive_answer(self, api_context, monkeypatch):
        fake_hits = [
            SimpleNamespace(
                chunk=SimpleNamespace(path="docs/project/DEFENSE_CURRENT_STATUS.md"),
                score=0.91,
                excerpt="三集平均准确率 98.56%，当前推荐模型为 bert_v11c_boundary_fix。",
            )
        ]

        monkeypatch.setattr(
            api_module,
            "get_project_knowledge_index",
            lambda force_refresh=False: SimpleNamespace(
                search=lambda question, top_k: fake_hits,
                source_count=18,
            ),
        )
        monkeypatch.setattr(
            api_module,
            "build_extractive_answer",
            lambda question, hits: "根据仓库资料，当前三集平均准确率为 98.56%。",
        )

        response = asyncio.run(
            api_module.project_qa(
                api_module.ProjectQARequest(question="当前三集平均准确率是多少", useLLM=False),
                make_request(),
                authorization=None,
                x_internal_token=None,
            )
        )

        assert response.mode == "extractive"
        assert response.answer == "根据仓库资料，当前三集平均准确率为 98.56%。"
        assert response.agentMode == "metrics"
        assert response.answerFrame == "指标口径答辩"
        assert response.answerLength == "standard"
        assert response.speakingStyle == "natural"
        assert response.effectiveSpeakerProfile is not None
        assert response.sourceCount == 1
        assert response.indexSourceCount == 18
        assert response.sources[0].path == "docs/project/DEFENSE_CURRENT_STATUS.md"
        assert any(item.tool == "repository_search" for item in response.toolTrace)
        assert response.suggestedQuestions

    def test_project_qa_uses_llm_when_available(self, api_context, monkeypatch):
        fake_hits = [
            SimpleNamespace(
                chunk=SimpleNamespace(path="docs/project/DEFENSE_CURRENT_STATUS.md"),
                score=0.91,
                excerpt="三集平均准确率 98.56%，当前推荐模型为 bert_v11c_boundary_fix。",
            )
        ]

        monkeypatch.setattr(
            api_module,
            "get_project_knowledge_index",
            lambda force_refresh=False: SimpleNamespace(
                search=lambda question, top_k: fake_hits,
                source_count=18,
            ),
        )
        monkeypatch.setattr(
            api_module,
            "build_extractive_answer",
            lambda question, hits: "extractive fallback",
        )
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        class FakeResponse:
            status_code = 200

            @staticmethod
            def json():
                return {
                    "model": "mock-rag-model",
                    "choices": [{"message": {"content": "这是基于仓库证据生成的答辩回答。"}}],
                }

        class FakeAsyncClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def post(self, url, headers=None, json=None):
                return FakeResponse()

        monkeypatch.setattr(api_module.httpx, "AsyncClient", FakeAsyncClient)

        response = asyncio.run(
            api_module.project_qa(
                api_module.ProjectQARequest(
                    question="当前推荐模型是什么",
                    answerLength="brief",
                    speakingStyle="confident",
                    speakerProfile="我是计算机专业本科生，正在做毕设答辩。",
                ),
                make_request(),
                authorization=None,
                x_internal_token=None,
            )
        )

        assert response.mode == "rag"
        assert response.model == "mock-rag-model"
        assert response.answer == "这是基于仓库证据生成的答辩回答。"
        assert response.answerFrame == "标准答辩口径"
        assert response.answerLength == "brief"
        assert response.speakingStyle == "confident"
        assert response.effectiveSpeakerProfile == "我是计算机专业本科生，正在做毕设答辩。"
        assert any(item.tool == "llm_synthesis" and item.status == "used" for item in response.toolTrace)

    def test_project_qa_uses_history_and_live_detector(self, api_context, monkeypatch):
        fake_hits = [
            SimpleNamespace(
                chunk=SimpleNamespace(path="docs/thesis/project_technical_deep_dive.md"),
                score=0.77,
                excerpt="双层检测架构由分类器和边界检测器组成。",
            )
        ]

        monkeypatch.setattr(
            api_module,
            "get_project_knowledge_index",
            lambda force_refresh=False: SimpleNamespace(
                search=lambda question, top_k: fake_hits,
                source_count=24,
            ),
        )
        monkeypatch.setattr(
            api_module,
            "build_extractive_answer",
            lambda question, hits: "根据现有证据，系统采用双层检测架构。",
        )

        response = asyncio.run(
            api_module.project_qa(
                api_module.ProjectQARequest(
                    question="继续上一个问题，为什么还要保留边界检测器？",
                    useLLM=False,
                    history=[
                        {"role": "user", "content": "这个项目的整体架构是什么？"},
                        {"role": "assistant", "content": "系统采用双层检测架构。"},
                    ],
                    analysisText="我先写一段，再让 AI 续写后半段。",
                ),
                make_request(),
                authorization=None,
                x_internal_token=None,
            )
        )

        assert response.agentMode == "technical"
        assert response.answerFrame == "原理解释链"
        assert response.memorySummary is not None
        assert any(item.tool == "conversation_memory" for item in response.toolTrace)
        assert any(item.tool == "live_detector" and item.status == "used" for item in response.toolTrace)


class TestProjectQAMaterialsEndpoint:
    """GET/POST /api/project-qa/materials"""

    def test_upload_and_list_project_qa_materials(self, api_context, tmp_path, monkeypatch):
        monkeypatch.setenv("DC_PROJECT_QA_UPLOAD_DIR", str(tmp_path))
        monkeypatch.setattr(api_module, "get_project_knowledge_index", lambda force_refresh=False: SimpleNamespace())

        upload = UploadFile(filename="答辩提纲.md", file=BytesIO("核心创新点\n边界检测".encode("utf-8")))

        upload_response = asyncio.run(
            api_module.upload_project_qa_materials(
                make_request(),
                files=[upload],
                x_internal_token=None,
            )
        )

        assert upload_response.status == "ok"
        assert len(upload_response.uploaded) == 1
        assert upload_response.uploaded[0].sourceType == "md"

        list_response = asyncio.run(
            api_module.list_project_qa_materials(
                make_request(),
                x_internal_token=None,
            )
        )

        assert list_response.total == 1
        assert list_response.materials[0].name.endswith(".md")

    def test_upload_project_qa_material_rejects_unsupported_file(self, api_context, tmp_path, monkeypatch):
        monkeypatch.setenv("DC_PROJECT_QA_UPLOAD_DIR", str(tmp_path))
        monkeypatch.setattr(api_module, "get_project_knowledge_index", lambda force_refresh=False: SimpleNamespace())

        upload = UploadFile(filename="notes.exe", file=BytesIO(b"binary"))

        response = asyncio.run(
            api_module.upload_project_qa_materials(
                make_request(),
                files=[upload],
                x_internal_token=None,
            )
        )

        assert response.uploaded == []
        assert response.skipped


class TestModelInfoEndpoint:
    """GET /api/model-info"""

    def test_model_info_has_core_sections(self, api_context):
        data = asyncio.run(api_module.model_info())
        assert "modelVersion" in data
        assert "runtime" in data
        assert "training" in data
        assert "evaluation" in data
        assert data["runtime"]["decisionThreshold"] == api_module.DECISION_THRESHOLD
        assert data["runtime"]["temperature"] == api_module.CLASSIFIER_TEMPERATURE

    def test_model_info_reflects_metrics_payload(self, api_context, monkeypatch):
        monkeypatch.setattr(
            api_module,
            "CLASSIFIER_METRICS",
            {
                "three_set_avg": 99.9,
                "_full": {"three_set_avg": 99.9, "independent_data": {"accuracy": 100.0}},
            },
        )
        monkeypatch.setattr(
            api_module, "CLASSIFIER_TRAINING_LOG", {"version": "v99", "train_samples": 12345}
        )

        data = asyncio.run(api_module.model_info())
        assert data["training"]["version"] == "v99"
        assert data["training"]["trainSamples"] == 12345
        assert data["evaluation"]["threeSetAvg"] == 99.9


class TestBatchDetectEndpoint:
    """POST /api/detect/batch"""

    def test_batch_detect_returns_per_item_results(self, api_context):
        mock_det = api_context
        mock_det.classify_batch.return_value = [
            {"prob_human": 0.9, "prob_ai": 0.1, "confidence": 0.9},
            {"prob_human": 0.05, "prob_ai": 0.95, "confidence": 0.95},
            {"prob_human": 0.5, "prob_ai": 0.5, "confidence": 0.5},
        ]

        response = api_module.detect_text_batch(
            api_module.BatchDetectRequest(texts=["人类文本。", "AI 文本。", "模糊文本。"]),
            make_request(),
        )

        assert response.total == 3
        assert len(response.results) == 3
        assert response.results[0].type == "human"
        assert response.results[1].type == "ai"
        assert response.results[2].type == "mixed"
        assert response.modelVersion == api_module.MODEL_VERSION

    def test_batch_detect_rejects_over_limit(self):
        oversized = ["x"] * (api_module.BATCH_MAX_ITEMS + 1)
        with pytest.raises(ValidationError):
            api_module.BatchDetectRequest(texts=oversized)

    def test_batch_detect_handles_blank_items(self, api_context):
        mock_det = api_context
        mock_det.classify_batch.return_value = [
            {"prob_human": 0.9, "prob_ai": 0.1, "confidence": 0.9},
        ]

        response = api_module.detect_text_batch(
            api_module.BatchDetectRequest(texts=["有效文本。", "   "]),
            make_request(),
        )

        assert response.results[0].type == "human"
        assert response.results[1].type == "invalid"
        assert response.results[1].error == "empty text"


class TestGetClientIp:
    """get_client_ip IP-extraction safety."""

    def test_prefers_x_real_ip(self):
        req = make_request({"x-real-ip": "198.51.100.9", "x-forwarded-for": "10.0.0.1, 10.0.0.2"})
        assert api_module.get_client_ip(req) == "198.51.100.9"

    def test_takes_rightmost_xff_when_no_real_ip(self):
        """Rightmost XFF is the last proxy — most trust-worthy without X-Real-IP."""
        req = make_request({"x-forwarded-for": "spoofed.client.ip, 10.0.0.99"})
        assert api_module.get_client_ip(req) == "10.0.0.99"

    def test_single_xff_entry(self):
        req = make_request({"x-forwarded-for": "203.0.113.5"})
        assert api_module.get_client_ip(req) == "203.0.113.5"

    def test_falls_back_to_peer_when_no_proxy_headers(self):
        req = make_request()
        assert api_module.get_client_ip(req) == "127.0.0.1"

    def test_ignores_blank_xff(self):
        req = make_request({"x-forwarded-for": "  "})
        assert api_module.get_client_ip(req) == "127.0.0.1"


class TestDetectExposesTokenSpans:
    """/api/detect now returns tokenSpans when span detector runs."""

    def test_token_spans_returned_for_long_text(self, api_context):
        mock_det = api_context
        mock_det.classify.return_value = {
            "label": "AI",
            "confidence": 0.9,
            "prob_human": 0.1,
            "prob_ai": 0.9,
        }
        mock_det.detect_boundary.return_value = {
            "boundary_token": 3,
            "boundary_char": 2,
            "text": "test",
            "tokenSpans": [
                {"token": "人", "charStart": 0, "charEnd": 1, "probAi": 0.05},
                {"token": "类", "charStart": 1, "charEnd": 2, "probAi": 0.07},
                {"token": "AI", "charStart": 2, "charEnd": 4, "probAi": 0.88},
            ],
        }

        long_text = "人类" + "AI生成的续写。" * 20
        response = api_module.detect_text(
            api_module.DetectRequest(text=long_text),
            make_request(),
        )

        assert response.tokenSpans is not None
        assert len(response.tokenSpans) == 3
        assert response.tokenSpans[0].probAi < 0.1
        assert response.tokenSpans[2].probAi > 0.8

    def test_token_spans_absent_for_short_text(self, api_context):
        mock_det = api_context
        mock_det.classify.return_value = {
            "label": "Human",
            "confidence": 0.95,
            "prob_human": 0.95,
            "prob_ai": 0.05,
        }

        response = api_module.detect_text(
            api_module.DetectRequest(text="太短了。"),
            make_request(),
        )

        assert response.tokenSpans is None
