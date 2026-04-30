"""Tests for agent infrastructure: hybrid mode classification, session storage, QA cache."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# A3: Hybrid mode classification
# ---------------------------------------------------------------------------


def test_score_agent_mode_metrics():
    from api.api import _score_agent_mode

    mode, conf = _score_agent_mode("模型的准确率是多少？")
    assert mode == "metrics"
    assert conf >= 0.8


def test_score_agent_mode_critical():
    from api.api import _score_agent_mode

    mode, conf = _score_agent_mode("这个系统有什么局限性和风险？")
    assert mode == "critical"
    assert conf >= 0.8


def test_score_agent_mode_technical():
    from api.api import _score_agent_mode

    mode, conf = _score_agent_mode("BERT的原理是什么？为什么用SEP？")
    assert mode == "technical"
    assert conf >= 0.8


def test_score_agent_mode_defense_default():
    from api.api import _score_agent_mode

    mode, conf = _score_agent_mode("帮我介绍一下你的项目")
    assert mode == "defense"
    assert conf == 0.0


def test_score_agent_mode_ambiguous():
    from api.api import _score_agent_mode

    mode, conf = _score_agent_mode("准确率很高但是有什么风险？")
    assert conf < 0.8


def test_infer_mode_explicit_override():
    from api.api import infer_project_agent_mode

    result = asyncio.get_event_loop().run_until_complete(
        infer_project_agent_mode("随便问什么问题", "metrics")
    )
    assert result == "metrics"


def test_infer_mode_keyword_fast_path():
    from api.api import infer_project_agent_mode

    result = asyncio.get_event_loop().run_until_complete(
        infer_project_agent_mode("训练指标和ECE分别是多少", None)
    )
    assert result == "metrics"


def test_infer_mode_llm_fallback(monkeypatch):
    from api.api import infer_project_agent_mode

    async def fake_llm(question: str) -> str | None:
        return "technical"

    monkeypatch.setattr("api.api._llm_classify_agent_mode", fake_llm)
    result = asyncio.get_event_loop().run_until_complete(
        infer_project_agent_mode("帮我看看数据治理到底做了什么", None)
    )
    assert result == "technical"


def test_infer_mode_llm_failure_falls_back(monkeypatch):
    from api.api import infer_project_agent_mode

    async def fake_llm_fail(question: str) -> str | None:
        return None

    monkeypatch.setattr("api.api._llm_classify_agent_mode", fake_llm_fail)
    result = asyncio.get_event_loop().run_until_complete(
        infer_project_agent_mode("帮我看看数据治理到底做了什么", None)
    )
    assert result == "technical"


# ---------------------------------------------------------------------------
# B2: Session storage
# ---------------------------------------------------------------------------


def test_session_roundtrip(tmp_path: Path, monkeypatch):
    from api.api import load_session_history, save_session_turn

    monkeypatch.setattr("api.api.SESSION_DIR", tmp_path)
    sid = "test-session-001"
    save_session_turn(sid, "user", "你好世界")
    save_session_turn(sid, "assistant", "你好，有什么可以帮你的？")

    history = load_session_history(sid)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


def test_session_invalid_id():
    from api.api import load_session_history, save_session_turn

    assert load_session_history("../../../etc/passwd") == []
    save_session_turn("../../../etc/passwd", "user", "hack")  # should be no-op


def test_session_missing_file():
    from api.api import load_session_history

    assert load_session_history("nonexistent-session") == []


def test_session_storage_failure_is_best_effort(tmp_path: Path, monkeypatch):
    from api.api import load_session_history, save_session_turn

    blocked_path = tmp_path / "not-a-directory"
    blocked_path.write_text("occupied", encoding="utf-8")
    monkeypatch.setattr("api.api.SESSION_DIR", blocked_path)
    monkeypatch.setattr("api.api._SESSION_STORAGE_WARNING_EMITTED", False)

    save_session_turn("readonly-session-001", "user", "这次保存失败也不能影响问答")

    assert load_session_history("readonly-session-001") == []


def test_merge_session_no_sid():
    from api.api import ProjectQARequest, merge_session_history

    payload = ProjectQARequest(question="测试问题内容")
    history, sid = merge_session_history(payload)
    assert sid is None
    assert history == []


def test_merge_session_with_sid(tmp_path: Path, monkeypatch):
    from api.api import ProjectQARequest, merge_session_history, save_session_turn

    monkeypatch.setattr("api.api.SESSION_DIR", tmp_path)
    sid = "merge-test-001"
    save_session_turn(sid, "user", "之前的问题内容")
    save_session_turn(sid, "assistant", "之前的回答内容")

    payload = ProjectQARequest(question="这是一个新问题", sessionId=sid)
    history, effective_sid = merge_session_history(payload)
    assert effective_sid == sid
    assert len(history) == 2


# ---------------------------------------------------------------------------
# C1: QA response cache
# ---------------------------------------------------------------------------


def test_cache_key_deterministic():
    from api.api import ProjectQARequest, _qa_cache_key

    p1 = ProjectQARequest(question="你好世界测试")
    p2 = ProjectQARequest(question="你好世界测试")
    assert _qa_cache_key(p1) == _qa_cache_key(p2)


def test_cache_key_differs_for_different_questions():
    from api.api import ProjectQARequest, _qa_cache_key

    p1 = ProjectQARequest(question="问题内容A")
    p2 = ProjectQARequest(question="问题内容B")
    assert _qa_cache_key(p1) != _qa_cache_key(p2)


def test_cache_key_includes_agent_context_fields():
    from api.api import ProjectQARequest, _qa_cache_key

    base = ProjectQARequest(question="同一个问题测试", useLLM=False)
    with_analysis = ProjectQARequest(
        question="同一个问题测试",
        useLLM=False,
        analysisText="待检测文本 A",
    )
    with_history = ProjectQARequest(
        question="同一个问题测试",
        useLLM=False,
        history=[{"role": "user", "content": "前一个问题"}],
    )
    with_profile = ProjectQARequest(
        question="同一个问题测试",
        useLLM=False,
        speakerProfile="不同答辩身份",
    )
    with_llm = ProjectQARequest(question="同一个问题测试", useLLM=True)

    assert _qa_cache_key(base) != _qa_cache_key(with_analysis)
    assert _qa_cache_key(base) != _qa_cache_key(with_history)
    assert _qa_cache_key(base) != _qa_cache_key(with_profile)
    assert _qa_cache_key(base) != _qa_cache_key(with_llm)


def test_cache_put_and_get(monkeypatch):
    from api.api import ProjectQARequest, _qa_cache_get, _qa_cache_put

    monkeypatch.setattr("api.api._qa_cache", {})
    payload = ProjectQARequest(question="缓存测试问题")
    data = {"answer": "cached answer", "mode": "extractive"}

    assert _qa_cache_get(payload) is None
    _qa_cache_put(payload, data)
    assert _qa_cache_get(payload) == data


def test_cache_ttl_expiry(monkeypatch):
    from api.api import ProjectQARequest, _qa_cache_get, _qa_cache_put

    monkeypatch.setattr("api.api._qa_cache", {})
    monkeypatch.setattr("api.api._QA_CACHE_TTL", 0.1)
    payload = ProjectQARequest(question="过期测试问题")
    _qa_cache_put(payload, {"answer": "old"})

    time.sleep(0.15)
    assert _qa_cache_get(payload) is None


def test_cache_eviction(monkeypatch):
    from api.api import ProjectQARequest, _qa_cache, _qa_cache_put

    monkeypatch.setattr("api.api._qa_cache", {})
    monkeypatch.setattr("api.api._QA_CACHE_MAX", 3)

    for i in range(5):
        _qa_cache_put(ProjectQARequest(question=f"问题编号{i:03d}"), {"answer": f"a{i}"})

    assert len(_qa_cache) <= 3


def test_file_view_allows_nested_configured_dirs(tmp_path: Path, monkeypatch):
    from api.api import _resolve_safe_project_file

    frontend_file = tmp_path / "frontend" / "app" / "advisor" / "page.tsx"
    model_file = tmp_path / "models" / "bert_v11c_boundary_fix" / "eval_comparison.json"
    private_file = tmp_path / "private" / "secret.json"

    frontend_file.parent.mkdir(parents=True)
    model_file.parent.mkdir(parents=True)
    private_file.parent.mkdir(parents=True)
    frontend_file.write_text("export default function Page() { return null }\n", encoding="utf-8")
    model_file.write_text("{}", encoding="utf-8")
    private_file.write_text("{}", encoding="utf-8")

    monkeypatch.setattr("api.api.PROJECT_ROOT", tmp_path)

    assert _resolve_safe_project_file("frontend/app/advisor/page.tsx") == frontend_file.resolve()
    assert _resolve_safe_project_file("models/bert_v11c_boundary_fix/eval_comparison.json")
    assert _resolve_safe_project_file("private/secret.json") is None


def test_project_local_answer_generic_question_does_not_recurse():
    from api.api import build_project_local_answer

    answer = build_project_local_answer(
        "普通问法：这个项目有什么价值？",
        agent_mode="defense",
        answer_frame_title="标准说明口径",
        hits=[],
        model_snapshot=None,
    )

    assert "检索" in answer or "仓库" in answer


def test_project_qa_retry_detects_dangling_enumeration():
    from api.api import should_retry_project_qa_completion

    answer = (
        "老师好，我的毕业设计在中文AI文本检测任务上，最终模型在独立评估集上达到了"
        "98.57%的准确率，同时校准误差（ECE）仅为0.0034，这说明模型不仅预测准确，"
        "而且输出的置信度非常可靠。下面我从数据、算法、"
    )

    assert should_retry_project_qa_completion(answer, None)
    assert should_retry_project_qa_completion(answer, "length")
    assert not should_retry_project_qa_completion(answer + "工程三个方面展开。", None)
