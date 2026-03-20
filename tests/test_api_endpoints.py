"""Tests for the FastAPI endpoints using TestClient.

The BERT models are mocked by patching the global `detector` variable
directly, avoiding the need to intercept class instantiation.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import api.api as api_module


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
def client(mock_detector):
    """Create a TestClient with mocked detector — no model loading needed."""
    original_token_flag = api_module.ENFORCE_INTERNAL_TOKEN
    api_module.ENFORCE_INTERNAL_TOKEN = False

    with TestClient(api_module.app) as tc:
        # Override detector AFTER lifespan has run (lifespan loads real models)
        original_detector = api_module.detector
        api_module.detector = mock_detector
        yield tc, mock_detector
        api_module.detector = original_detector

    api_module.ENFORCE_INTERNAL_TOKEN = original_token_flag


class TestHealthEndpoint:
    """GET /api/health"""

    def test_health_returns_ok(self, client):
        tc, _ = client
        resp = tc.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["detectorReady"] is True
        assert "modelVersion" in data


class TestDetectEndpoint:
    """POST /api/detect"""

    def test_detect_ai_text(self, client):
        tc, mock_det = client
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

        resp = tc.post("/api/detect", json={"text": "AI生成的文本，具有典型的结构化特征。"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "ai"
        assert "confidence" in data
        assert "sentences" in data
        assert data["processingTime"] >= 0

    def test_detect_human_text(self, client):
        tc, mock_det = client
        mock_det.classify.return_value = {
            "label": "Human",
            "confidence": 0.92,
            "prob_human": 0.92,
            "prob_ai": 0.08,
        }

        resp = tc.post("/api/detect", json={"text": "我今天去超市买了点菜，晚上做个红烧肉。"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "human"
        assert data["humanPercentage"] > data["aiPercentage"]

    def test_detect_empty_text_rejected(self, client):
        tc, _ = client
        resp = tc.post("/api/detect", json={"text": ""})
        assert resp.status_code == 422

    def test_detect_missing_text_rejected(self, client):
        tc, _ = client
        resp = tc.post("/api/detect", json={})
        assert resp.status_code == 422

    def test_detect_response_has_sentences(self, client):
        tc, mock_det = client
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

        resp = tc.post(
            "/api/detect",
            json={"text": "第一句话。第二句话。第三句话。"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["sentences"]) == 3

    def test_detect_mixed_with_boundary(self, client):
        tc, mock_det = client
        # Neither prob exceeds DECISION_THRESHOLD (80%) → starts as "mixed"
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

        resp = tc.post(
            "/api/detect",
            json={"text": "我觉得还行。AI生成的内容开始了。继续生成。"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "mixed"
        assert data["boundary"] is not None


class TestChatEndpoint:
    """POST /v1/chat/completions"""

    def test_chat_no_api_key_returns_500(self, client, monkeypatch):
        tc, _ = client
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        resp = tc.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )
        assert resp.status_code == 500

    def test_chat_empty_messages_rejected(self, client):
        tc, _ = client
        resp = tc.post(
            "/v1/chat/completions",
            json={"messages": []},
        )
        assert resp.status_code == 422
