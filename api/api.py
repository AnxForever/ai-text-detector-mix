import hashlib
import json
import logging
import os
import re
import secrets
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Literal, cast
from urllib.parse import urlparse

import httpx
import torch
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

try:
    from transformers import (
        BertForSequenceClassification,
        BertForTokenClassification,
        BertTokenizer,
    )
except ImportError:  # pragma: no cover - exercised only in lean test envs
    BertForSequenceClassification = None
    BertForTokenClassification = None
    BertTokenizer = None

from scripts.utils.paths import PATHS, PROJECT_ROOT
from scripts.utils.project_qa import (
    KnowledgeHit,
    ProjectKnowledgeIndex,
    build_extractive_answer,
    list_uploaded_project_sources,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

CLASSIFIER_MODEL_PATH = os.getenv("DETECTOR_CLASSIFIER_MODEL", "models/bert_v11c_boundary_fix")
SPAN_MODEL_PATH = os.getenv("DETECTOR_SPAN_MODEL", "models/bert_span_detector")
USE_INT8 = os.getenv("DETECTOR_USE_INT8", "0").strip().lower() in {"1", "true", "yes"}
INT8_CLASSIFIER_PATH = os.getenv("DETECTOR_INT8_CLASSIFIER", "models/bert_v11c_int8")
INT8_SPAN_PATH = os.getenv("DETECTOR_INT8_SPAN", "models/bert_span_int8")
CLASSIFIER_MAX_LENGTH = int(os.getenv("DETECTOR_MAX_LENGTH", "256"))
CLASSIFIER_TEMPERATURE = float(os.getenv("DETECTOR_TEMPERATURE", "0.8165"))
DECISION_THRESHOLD = float(os.getenv("DETECTOR_DECISION_THRESHOLD", "0.8"))
SPAN_TRIGGER_MIN_CHARS = int(os.getenv("DETECTOR_SPAN_TRIGGER_MIN_CHARS", "80"))
EXPOSE_TOKEN_PROBS = os.getenv("DETECTOR_EXPOSE_TOKEN_PROBS", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
MODEL_VERSION = os.path.basename(CLASSIFIER_MODEL_PATH.rstrip("/\\"))
DEFAULT_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "deepseek-ai/deepseek-v3.1")
DEFAULT_DEFENSE_PROFILE = os.getenv(
    "PROJECT_DEFENSE_PROFILE",
    "我是西安科技大学计算机科学与技术专业本科生包安心，正在做中文AI文本检测方向的毕业设计答辩。",
)
PROJECT_QA_MODEL_PRESETS_RAW = os.getenv("PROJECT_QA_MODEL_PRESETS", "").strip()
PROJECT_QA_DEFAULT_PRESET_ID_ENV = os.getenv("PROJECT_QA_DEFAULT_PRESET_ID", "").strip()


def _load_classifier_metrics(model_path: str) -> dict[str, Any]:
    """Load evaluation metrics from the model's eval_comparison.json (if present)."""
    metrics_path = Path(model_path) / "eval_comparison.json"
    if not metrics_path.exists():
        return {}
    try:
        with metrics_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as exc:
        logger.warning("Failed to read %s: %s", metrics_path, exc)
        return {}

    # File shape: {"<model_name>": {...metrics...}}
    if not isinstance(raw, dict) or not raw:
        return {}
    entry = next(iter(raw.values()))
    if not isinstance(entry, dict):
        return {}

    three_set_avg = entry.get("three_set_avg")
    independent = (
        entry.get("independent_data", {}) if isinstance(entry.get("independent_data"), dict) else {}
    )
    calibration = (
        entry.get("independent_data_calibration", {})
        if isinstance(entry.get("independent_data_calibration"), dict)
        else {}
    )
    return {
        "three_set_avg": three_set_avg,
        "independent_accuracy": independent.get("accuracy"),
        "independent_f1": independent.get("f1"),
        "ece_after": calibration.get("ECE_after"),
        "optimal_temperature": calibration.get("optimal_T"),
        "_full": entry,  # full payload for /api/model-info
    }


def _load_training_log(model_path: str) -> dict[str, Any]:
    """Load training log (version/config/epoch history) if present."""
    log_path = Path(model_path) / "training_log.json"
    if not log_path.exists():
        return {}
    try:
        with log_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as exc:
        logger.warning("Failed to read %s: %s", log_path, exc)
        return {}
    if not isinstance(raw, dict):
        return {}
    return raw


CLASSIFIER_METRICS: dict[str, Any] = _load_classifier_metrics(CLASSIFIER_MODEL_PATH)
CLASSIFIER_TRAINING_LOG: dict[str, Any] = _load_training_log(CLASSIFIER_MODEL_PATH)

MAX_DETECT_TEXT_CHARS = int(os.getenv("DETECTOR_MAX_TEXT_CHARS", "10000"))
CHAT_MAX_MESSAGES = int(os.getenv("OPENAI_CHAT_MAX_MESSAGES", "50"))
CHAT_MAX_TOKENS = int(os.getenv("OPENAI_CHAT_MAX_TOKENS", "2048"))
UPSTREAM_CHAT_TIMEOUT_SECONDS = int(os.getenv("OPENAI_TIMEOUT_SECONDS", "60"))

RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
DETECT_RATE_LIMIT_PER_WINDOW = int(os.getenv("DETECT_RATE_LIMIT_PER_WINDOW", "60"))
CHAT_RATE_LIMIT_PER_WINDOW = int(os.getenv("CHAT_RATE_LIMIT_PER_WINDOW", "20"))

ENFORCE_INTERNAL_TOKEN = os.getenv("ENFORCE_INTERNAL_TOKEN", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
INTERNAL_API_TOKEN = os.getenv("INTERNAL_API_TOKEN", "").strip()


def infer_provider_label(api_base: str, model_name: str, fallback_label: str | None = None) -> str:
    """Infer a short provider label from api base or model name."""
    if fallback_label:
        return fallback_label

    api_base_lower = api_base.lower()
    model_lower = model_name.lower()
    if "moonshot" in model_lower or "kimi" in model_lower:
        return "Moonshot"
    if "glm" in model_lower or "zai" in model_lower:
        return "智谱"
    if "qwen" in model_lower:
        return "通义千问"
    if "deepseek" in model_lower:
        return "DeepSeek"
    if "claude" in model_lower:
        return "Anthropic"
    if "openai" in model_lower or "gpt" in model_lower:
        return "OpenAI"

    parsed = urlparse(api_base)
    if parsed.netloc:
        return parsed.netloc
    return "默认提供方"


def load_project_qa_model_presets() -> list[dict[str, str]]:
    """Load project QA model presets from env, with a default fallback preset."""
    fallback_preset = {
        "id": "default",
        "label": "默认模型",
        "api_base": os.getenv("OPENAI_BASE_URL", "https://api.hotaruapi.top/v1").strip(),
        "api_key": os.getenv("OPENAI_API_KEY", "").strip(),
        "model": DEFAULT_CHAT_MODEL.strip(),
        "provider": infer_provider_label(
            os.getenv("OPENAI_BASE_URL", "https://api.hotaruapi.top/v1").strip(),
            DEFAULT_CHAT_MODEL.strip(),
        ),
    }

    if not PROJECT_QA_MODEL_PRESETS_RAW:
        return [fallback_preset]

    presets: list[dict[str, str]] = []
    for raw_entry in PROJECT_QA_MODEL_PRESETS_RAW.split(";"):
        entry = raw_entry.strip()
        if not entry:
            continue
        parts = [part.strip() for part in entry.split("|")]
        if len(parts) < 5:
            logger.warning("Skipping invalid PROJECT_QA_MODEL_PRESETS entry: %s", entry)
            continue

        preset_id, label, api_base, api_key, model_name, *rest = parts
        if not preset_id or not label or not api_base or not api_key or not model_name:
            logger.warning("Skipping incomplete PROJECT_QA_MODEL_PRESETS entry: %s", entry)
            continue

        provider = infer_provider_label(api_base, model_name, rest[0] if rest else None)
        presets.append(
            {
                "id": preset_id,
                "label": label,
                "api_base": api_base,
                "api_key": api_key,
                "model": model_name,
                "provider": provider,
            }
        )

    if not presets:
        return [fallback_preset]
    return presets


PROJECT_QA_MODEL_PRESETS = load_project_qa_model_presets()
PROJECT_QA_MODEL_PRESET_MAP = {preset["id"]: preset for preset in PROJECT_QA_MODEL_PRESETS}


def resolve_project_qa_default_preset_id() -> str:
    """Return the default preset id used by the project QA agent."""
    if PROJECT_QA_DEFAULT_PRESET_ID_ENV in PROJECT_QA_MODEL_PRESET_MAP:
        return PROJECT_QA_DEFAULT_PRESET_ID_ENV
    return PROJECT_QA_MODEL_PRESETS[0]["id"]


PROJECT_QA_DEFAULT_PRESET_ID = resolve_project_qa_default_preset_id()


def resolve_project_qa_model_preset(requested_preset_id: str | None) -> dict[str, str]:
    """Resolve the requested project QA model preset or fall back to default."""
    preset_id = (requested_preset_id or "").strip()
    if preset_id and preset_id in PROJECT_QA_MODEL_PRESET_MAP:
        return PROJECT_QA_MODEL_PRESET_MAP[preset_id]
    return PROJECT_QA_MODEL_PRESET_MAP[PROJECT_QA_DEFAULT_PRESET_ID]


def build_upstream_chat_headers(api_key: str) -> dict[str, str]:
    """Build stable upstream headers for OpenAI-compatible providers."""
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (compatible; DefenseCopilot/1.0; +https://www.baxfor.fun)",
    }


INCLUDE_RISK_OBSERVABILITY = os.getenv(
    "DETECTOR_INCLUDE_RISK_OBSERVABILITY", "0"
).strip().lower() in {"1", "true", "yes", "on"}

SENTENCE_SPLIT_PATTERN = re.compile(r"([。！？!?])")
FORMAL_PATTERN = re.compile(r"(通知|公告|特此|敬请|请各位|温馨提示|须知)")
TECH_PATTERN = re.compile(r"(算法|模型|神经网络|数据库|API|代码|训练|部署|实验|推理|调参)")
CASUAL_PATTERN = re.compile(r"(哈哈|hh|嗯|啊|呀|哇|我觉得|说实话|有点)")
TEMPLATE_LIKE_PATTERN = re.compile(
    r"(分析请求|逐句分析|改进思路|好的，用户|用户希望|As an AI|as an ai)",
    re.IGNORECASE,
)

RATE_LIMIT_STATE: dict[str, deque[float]] = defaultdict(deque)
RATE_LIMIT_LOCK = Lock()
FEEDBACK_WRITE_LOCK = Lock()

# --- Project QA response cache (LRU, TTL-based) ---
_QA_CACHE_MAX = 100
_QA_CACHE_TTL = 600
_qa_cache: dict[str, tuple[float, dict[str, Any]]] = {}


DEFAULT_FEEDBACK_DIR = Path(os.getenv("DETECTOR_FEEDBACK_DIR", "/app/datasets/feedback_loop"))

try:
    from scripts.utils.feedback_loop import lookup_feedback_override, persist_feedback
except Exception:

    def _normalize_feedback_label(label: str) -> str:
        normalized = label.strip().lower()
        if normalized not in {"human", "ai", "mixed"}:
            raise ValueError(f"Invalid label '{label}'")
        return normalized

    def _normalize_feedback_tags(tags: list[str] | None) -> list[str]:
        if not tags:
            return []

        result: list[str] = []
        seen: set[str] = set()
        for tag in tags:
            cleaned = tag.strip()
            if not cleaned:
                continue
            key = cleaned.casefold()
            if key in seen:
                continue
            seen.add(key)
            result.append(cleaned)
        return result

    def _derive_feedback_tags(
        predicted_label: str,
        confirmed_label: str,
        tags: list[str] | None = None,
    ) -> list[str]:
        normalized_tags = _normalize_feedback_tags(tags)
        derived = ["manual_feedback"]

        if predicted_label == "ai" and confirmed_label == "human":
            derived.append("false_positive")
        elif predicted_label == "human" and confirmed_label == "ai":
            derived.append("false_negative")
        elif predicted_label != confirmed_label:
            derived.append("label_corrected")

        return _normalize_feedback_tags(normalized_tags + derived)

    def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with FEEDBACK_WRITE_LOCK:
            with path.open("a", encoding="utf-8") as file_obj:
                json.dump(record, file_obj, ensure_ascii=False)
                file_obj.write("\n")

    def _canonicalize_feedback_text(text: str) -> str:
        return " ".join(text.split())

    def _build_feedback_dedup_key(text: str) -> str:
        canonical_text = _canonicalize_feedback_text(text)
        return hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()

    def _load_existing_dedup_keys(path: Path) -> set[str]:
        if not path.exists():
            return set()

        dedup_keys: set[str] = set()
        with path.open("r", encoding="utf-8") as file_obj:
            for raw_line in file_obj:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                dedup_key = row.get("dedup_key")
                if isinstance(dedup_key, str) and dedup_key:
                    dedup_keys.add(dedup_key)
                    continue

                text = row.get("text")
                if isinstance(text, str) and text.strip():
                    dedup_keys.add(_build_feedback_dedup_key(text))

        return dedup_keys

    def _extract_record_dedup_key(row: dict[str, Any]) -> str | None:
        dedup_key = row.get("dedup_key")
        if isinstance(dedup_key, str) and dedup_key:
            return dedup_key

        text = row.get("text")
        if isinstance(text, str) and text.strip():
            return _build_feedback_dedup_key(text)

        return None

    def _load_latest_records_by_dedup_key(path: Path) -> dict[str, dict[str, Any]]:
        if not path.exists():
            return {}

        latest_records: dict[str, dict[str, Any]] = {}
        with path.open("r", encoding="utf-8") as file_obj:
            for raw_line in file_obj:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                dedup_key = _extract_record_dedup_key(row)
                if dedup_key:
                    latest_records[dedup_key] = row

        return latest_records

    def _load_conflicted_dedup_keys(path: Path) -> set[str]:
        if not path.exists():
            return set()

        conflicted_keys: set[str] = set()
        with path.open("r", encoding="utf-8") as file_obj:
            for raw_line in file_obj:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                dedup_key = _extract_record_dedup_key(row)
                if dedup_key:
                    conflicted_keys.add(dedup_key)

        return conflicted_keys

    def lookup_feedback_override(
        *,
        text: str,
        output_dir: Path | None = None,
    ) -> dict[str, Any] | None:
        cleaned_text = text.strip()
        if not cleaned_text:
            return None

        stored_dir = output_dir or DEFAULT_FEEDBACK_DIR
        corrections_path = stored_dir / "misclassified_samples.jsonl"
        conflicts_path = stored_dir / "feedback_conflicts.jsonl"
        dedup_key = _build_feedback_dedup_key(cleaned_text)

        if dedup_key in _load_conflicted_dedup_keys(conflicts_path):
            return None

        existing_records = _load_latest_records_by_dedup_key(corrections_path)
        record = existing_records.get(dedup_key)
        if record is None:
            return None

        confirmed_label = record.get("confirmed_label")
        if not isinstance(confirmed_label, str):
            return None

        try:
            normalized_label = _normalize_feedback_label(confirmed_label)
        except ValueError:
            return None

        boundary = record.get("boundary")
        return {
            "feedback_id": record.get("feedback_id"),
            "dedup_key": dedup_key,
            "confirmed_label": normalized_label,
            "boundary": boundary if isinstance(boundary, int) and boundary >= 0 else None,
            "domain_hint": record.get("domain_hint"),
            "source": "manual_feedback_exact_match",
        }

    def persist_feedback(
        *,
        text: str,
        predicted_label: str,
        confirmed_correct: bool,
        confirmed_label: str | None = None,
        tags: list[str] | None = None,
        note: str | None = None,
        source: str,
        model_version: str | None = None,
        confidence: float | None = None,
        ai_percentage: int | None = None,
        human_percentage: int | None = None,
        boundary: int | None = None,
        domain_hint: str | None = None,
        output_dir: Path | None = None,
    ) -> dict[str, Any]:
        cleaned_text = text.strip()
        if not cleaned_text:
            raise ValueError("text must not be blank")

        normalized_predicted = _normalize_feedback_label(predicted_label)
        normalized_confirmed = (
            _normalize_feedback_label(confirmed_label)
            if confirmed_label is not None
            else normalized_predicted
        )

        timestamp = datetime.now().isoformat()
        feedback_id = uuid.uuid4().hex
        stored_dir = output_dir or DEFAULT_FEEDBACK_DIR
        corrections_path = stored_dir / "misclassified_samples.jsonl"
        conflicts_path = stored_dir / "feedback_conflicts.jsonl"
        dedup_key = _build_feedback_dedup_key(cleaned_text)
        feedback_tags = _derive_feedback_tags(
            predicted_label=normalized_predicted,
            confirmed_label=normalized_confirmed,
            tags=tags,
        )

        record = {
            "feedback_id": feedback_id,
            "created_at": timestamp,
            "text": cleaned_text,
            "predicted_label": normalized_predicted,
            "confirmed_correct": confirmed_correct,
            "confirmed_label": normalized_confirmed,
            "tags": feedback_tags,
            "note": note.strip() if note else None,
            "source": source,
            "model_version": model_version,
            "confidence": confidence,
            "ai_percentage": ai_percentage,
            "human_percentage": human_percentage,
            "boundary": boundary,
            "domain_hint": domain_hint,
            "dedup_key": dedup_key,
            "dataset_type": "manual_correction",
        }

        if not confirmed_correct:
            corrections_path.parent.mkdir(parents=True, exist_ok=True)
            with FEEDBACK_WRITE_LOCK:
                existing_records = _load_latest_records_by_dedup_key(corrections_path)
                existing_record = existing_records.get(dedup_key)
                if existing_record is None:
                    with corrections_path.open("a", encoding="utf-8") as file_obj:
                        json.dump(record, file_obj, ensure_ascii=False)
                        file_obj.write("\n")
                    misclassified_saved = True
                    conflict_detected = False
                else:
                    existing_confirmed_label = existing_record.get("confirmed_label")
                    if existing_confirmed_label == normalized_confirmed:
                        misclassified_saved = False
                        conflict_detected = False
                    else:
                        conflict_record = {
                            **record,
                            "event_type": "conflicting_manual_feedback",
                            "existing_feedback_id": existing_record.get("feedback_id"),
                            "existing_confirmed_label": existing_confirmed_label,
                        }
                        with conflicts_path.open("a", encoding="utf-8") as file_obj:
                            json.dump(conflict_record, file_obj, ensure_ascii=False)
                            file_obj.write("\n")
                        misclassified_saved = False
                        conflict_detected = True
        else:
            misclassified_saved = False
            conflict_detected = False

        return {
            "feedback_id": feedback_id,
            "created_at": timestamp,
            "stored_dir": str(stored_dir),
            "corrections_path": str(corrections_path),
            "conflicts_path": str(conflicts_path),
            "misclassified_saved": misclassified_saved,
            "duplicate_skipped": not confirmed_correct and not misclassified_saved,
            "conflict_detected": conflict_detected,
        }


def get_client_ip(request: Request) -> str:
    """Extract client IP from request.

    Trust order (most-to-least spoof-resistant):
    1. X-Real-IP — nginx sets this to $remote_addr and overwrites client value
    2. X-Forwarded-For rightmost entry — closest to the trusted proxy; clients
       can prepend but cannot remove what the proxy appends
    3. request.client.host — direct TCP peer (correct when no proxy)

    Taking the FIRST X-Forwarded-For entry would let clients spoof their IP
    since nginx uses $proxy_add_x_forwarded_for (append, not overwrite).
    """
    real_ip = request.headers.get("x-real-ip")
    if real_ip and real_ip.strip():
        return real_ip.strip()
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        parts = [segment.strip() for segment in forwarded_for.split(",") if segment.strip()]
        if parts:
            return parts[-1]
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def enforce_rate_limit(request: Request, scope: str, max_requests: int) -> None:
    if max_requests <= 0:
        return

    client_ip = get_client_ip(request)
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS
    bucket_key = f"{scope}:{client_ip}"

    with RATE_LIMIT_LOCK:
        bucket = RATE_LIMIT_STATE[bucket_key]
        while bucket and bucket[0] < window_start:
            bucket.popleft()
        # Evict empty buckets to prevent unbounded memory growth
        if not bucket and bucket_key in RATE_LIMIT_STATE:
            del RATE_LIMIT_STATE[bucket_key]
            bucket = RATE_LIMIT_STATE[bucket_key]  # re-create via defaultdict
        if len(bucket) >= max_requests:
            raise HTTPException(status_code=429, detail="Too many requests, please retry later")
        bucket.append(now)


def verify_internal_token(header_token: str | None) -> None:
    if not ENFORCE_INTERNAL_TOKEN:
        return
    if not INTERNAL_API_TOKEN:
        raise HTTPException(status_code=500, detail="INTERNAL_API_TOKEN is not configured")
    if not header_token or not secrets.compare_digest(header_token.strip(), INTERNAL_API_TOKEN):
        raise HTTPException(status_code=401, detail="Unauthorized")


def _load_quantized_model(model_cls: type, fp32_path: str, state_dict_path: str):
    """Rebuild quantize_dynamic wrapper from FP32 config + load INT8 state_dict.

    state_dict files are cross-transformers-version compatible (tensors only),
    whereas pickled whole-model files break across 4.x / 5.x.
    """
    base = model_cls.from_pretrained(fp32_path, attn_implementation="eager")
    base.eval()
    wrapper = torch.quantization.quantize_dynamic(base, {torch.nn.Linear}, dtype=torch.qint8)
    state = torch.load(state_dict_path, map_location="cpu")
    wrapper.load_state_dict(state)
    wrapper.eval()
    return wrapper


class HybridTextDetector:
    def __init__(self) -> None:
        if (
            BertTokenizer is None
            or BertForSequenceClassification is None
            or BertForTokenClassification is None
        ):
            raise RuntimeError("transformers is required to load detector models")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading models on %s (INT8=%s) ...", self.device, USE_INT8)
        self.classifier_max_length = CLASSIFIER_MAX_LENGTH
        self.classifier_temperature = max(CLASSIFIER_TEMPERATURE, 1e-6)

        tokenizer_src = INT8_CLASSIFIER_PATH if USE_INT8 else CLASSIFIER_MODEL_PATH
        span_tokenizer_src = INT8_SPAN_PATH if USE_INT8 else SPAN_MODEL_PATH

        try:
            self.classifier_tokenizer = BertTokenizer.from_pretrained(tokenizer_src)
            if USE_INT8:
                self.classifier = _load_quantized_model(
                    BertForSequenceClassification,
                    CLASSIFIER_MODEL_PATH,
                    os.path.join(INT8_CLASSIFIER_PATH, "quantized_state_dict.pt"),
                )
            else:
                self.classifier = BertForSequenceClassification.from_pretrained(
                    CLASSIFIER_MODEL_PATH
                ).to(self.device)
                self.classifier.eval()
            logger.info(
                "Classifier loaded (%s, max_length=%d, temperature=%.4f).",
                tokenizer_src,
                self.classifier_max_length,
                self.classifier_temperature,
            )
        except Exception as exc:
            logger.error("Error loading classifier: %s", exc)
            raise

        try:
            self.span_tokenizer = BertTokenizer.from_pretrained(span_tokenizer_src)
            if USE_INT8:
                self.span_detector = _load_quantized_model(
                    BertForTokenClassification,
                    SPAN_MODEL_PATH,
                    os.path.join(INT8_SPAN_PATH, "quantized_state_dict.pt"),
                )
            else:
                self.span_detector = BertForTokenClassification.from_pretrained(SPAN_MODEL_PATH).to(
                    self.device
                )
                self.span_detector.eval()
            logger.info("Span detector loaded (%s).", span_tokenizer_src)
        except Exception as exc:
            logger.error("Error loading span detector: %s", exc)
            raise

    def classify(self, text: str) -> dict[str, float | str]:
        encoding = self.classifier_tokenizer(
            text,
            max_length=self.classifier_max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        with torch.inference_mode():
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            outputs = self.classifier(input_ids=input_ids, attention_mask=attention_mask)
            scaled_logits = outputs.logits[0] / self.classifier_temperature
            probs = torch.softmax(scaled_logits, dim=0)
            pred = torch.argmax(scaled_logits).item()

        return {
            "label": "AI" if pred == 1 else "Human",
            "confidence": probs[pred].item(),
            "prob_human": probs[0].item(),
            "prob_ai": probs[1].item(),
        }

    def classify_batch(self, texts: list[str]) -> list[dict[str, float]]:
        """Classify a batch of texts in a single forward pass."""
        if not texts:
            return []
        encoding = self.classifier_tokenizer(
            texts,
            max_length=self.classifier_max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with torch.inference_mode():
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            outputs = self.classifier(input_ids=input_ids, attention_mask=attention_mask)
            scaled_logits = outputs.logits / self.classifier_temperature
            probs = torch.softmax(scaled_logits, dim=-1).cpu()

        results: list[dict[str, float]] = []
        for row in probs:
            prob_human = float(row[0].item())
            prob_ai = float(row[1].item())
            results.append(
                {
                    "prob_human": prob_human,
                    "prob_ai": prob_ai,
                    "confidence": max(prob_human, prob_ai),
                }
            )
        return results

    def detect_boundary(self, text: str) -> dict[str, Any]:
        text_clean = text.replace("[SEP]", "")
        encoding = self.span_tokenizer(
            text_clean,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        with torch.inference_mode():
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            outputs = self.span_detector(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[0]
            # Token-level probability that the token is AI-generated (label 1).
            token_probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()

        tokens = self.span_tokenizer.convert_ids_to_tokens(input_ids[0])
        mask = attention_mask[0].cpu().numpy()

        boundary_idx = None
        for i in range(1, len(preds)):
            if preds[i - 1] == 0 and preds[i] == 1:
                boundary_idx = i
                break

        char_pos = 0
        boundary_char = None
        token_spans: list[dict[str, Any]] = []
        for i, token in enumerate(tokens):
            # Stop at padding; attention_mask becomes 0 beyond the real content.
            if int(mask[i]) == 0:
                break
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue
            if i == boundary_idx and boundary_char is None:
                boundary_char = char_pos
            token_text = token.replace("##", "")
            # Map [UNK] back to a single source character since the tokenizer
            # collapses unknown chars to one UNK token.
            token_len = 1 if token == "[UNK]" else len(token_text)
            token_spans.append(
                {
                    "token": token,
                    "charStart": char_pos,
                    "charEnd": char_pos + token_len,
                    "probAi": float(token_probs[i][1]),
                }
            )
            char_pos += token_len

        return {
            "boundary_token": boundary_idx,
            "boundary_char": boundary_char,
            "text": text_clean,
            "tokenSpans": token_spans,
        }


detector: HybridTextDetector | None = None
project_knowledge_index: ProjectKnowledgeIndex | None = None
PROJECT_QA_LOCK = Lock()
DETECTOR_INIT_LOCK = Lock()


@asynccontextmanager
async def lifespan(application: FastAPI):
    global detector
    detector = HybridTextDetector()
    # Warmup: 触发 PyTorch kernel JIT 编译 & 把模型权重从 swap 拉回内存,
    # 避免首个真实请求承担 10-30s 冷启动开销。
    try:
        detector.classify("预热")
        detector.classify_batch(["人工智能检测预热。", "这是一段示例文本。"])
        detector.detect_boundary("人工智能正在改变世界。这是一段用于预热的示例文本。")
        logger.info("Detector warmup complete.")
    except Exception as warm_exc:
        logger.warning("Detector warmup failed (non-fatal): %s", warm_exc)
    yield


app = FastAPI(
    title="AI Text Detection API",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "https://baxfor.fun,http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
]
CORS_ALLOW_CREDENTIALS = os.getenv("CORS_ALLOW_CREDENTIALS", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
if "*" in CORS_ORIGINS:
    CORS_ALLOW_CREDENTIALS = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Internal-Token"],
)


@app.get("/api/health")
async def health_check() -> dict[str, Any]:
    accuracy_pct = CLASSIFIER_METRICS.get("three_set_avg")
    system_info: dict[str, Any] = {}
    try:
        meminfo: dict[str, int] = {}
        with open("/proc/meminfo") as f:
            for line in f:
                key, _, value = line.partition(":")
                parts = value.strip().split()
                if parts:
                    meminfo[key] = int(parts[0])  # kB
        mem_total_mb = meminfo.get("MemTotal", 0) / 1024
        mem_avail_mb = meminfo.get("MemAvailable", 0) / 1024
        swap_total_mb = meminfo.get("SwapTotal", 0) / 1024
        swap_free_mb = meminfo.get("SwapFree", 0) / 1024

        proc_rss_kb = 0
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        proc_rss_kb = int(line.split()[1])
                        break
        except OSError:
            pass

        try:
            with open("/proc/loadavg") as f:
                load_parts = f.read().split()[:3]
        except OSError:
            load_parts = None

        system_info = {
            "memory": {
                "totalMB": round(mem_total_mb, 1),
                "availableMB": round(mem_avail_mb, 1),
                "usedPercent": (
                    round((mem_total_mb - mem_avail_mb) / mem_total_mb * 100, 1)
                    if mem_total_mb
                    else None
                ),
            },
            "swap": {
                "totalMB": round(swap_total_mb, 1),
                "usedMB": round(swap_total_mb - swap_free_mb, 1),
                "usedPercent": (
                    round((swap_total_mb - swap_free_mb) / swap_total_mb * 100, 1)
                    if swap_total_mb
                    else None
                ),
            },
            "processRssMB": round(proc_rss_kb / 1024, 1),
            "loadAvg": load_parts,
        }
    except Exception as exc:  # pragma: no cover
        system_info = {"error": str(exc)}

    return {
        "status": "ok",
        "detectorReady": detector is not None,
        "modelVersion": MODEL_VERSION,
        "decisionThreshold": DECISION_THRESHOLD,
        "maxLength": CLASSIFIER_MAX_LENGTH,
        "authEnabled": ENFORCE_INTERNAL_TOKEN,
        "accuracy": f"{accuracy_pct:.2f}%" if isinstance(accuracy_pct, (int, float)) else None,
        "metrics": {
            "threeSetAvg": CLASSIFIER_METRICS.get("three_set_avg"),
            "independentAccuracy": CLASSIFIER_METRICS.get("independent_accuracy"),
            "independentF1": CLASSIFIER_METRICS.get("independent_f1"),
            "eceAfterCalibration": CLASSIFIER_METRICS.get("ece_after"),
            "optimalTemperature": CLASSIFIER_METRICS.get("optimal_temperature"),
        },
        "temperature": CLASSIFIER_TEMPERATURE,
        "spanDetectorReady": detector is not None
        and hasattr(detector, "span_detector")
        and detector.span_detector is not None,
        "system": system_info,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/model-info")
async def model_info() -> dict[str, Any]:
    """Return full model metadata: training config, epoch history, per-source evaluation.

    Useful for reproducibility verification — a single curl exposes the exact
    numbers backing the thesis (accuracy, ECE, temperature, per-source breakdown).
    """
    full_metrics = CLASSIFIER_METRICS.get("_full") or {}
    training_log = CLASSIFIER_TRAINING_LOG

    return {
        "modelVersion": MODEL_VERSION,
        "classifierPath": CLASSIFIER_MODEL_PATH,
        "spanDetectorPath": SPAN_MODEL_PATH,
        "runtime": {
            "maxLength": CLASSIFIER_MAX_LENGTH,
            "decisionThreshold": DECISION_THRESHOLD,
            "temperature": CLASSIFIER_TEMPERATURE,
            "spanTriggerMinChars": SPAN_TRIGGER_MIN_CHARS,
        },
        "training": {
            "version": training_log.get("version"),
            "strategy": training_log.get("strategy"),
            "description": training_log.get("description"),
            "baseModel": training_log.get("base_model"),
            "trainSamples": training_log.get("train_samples"),
            "valSamples": training_log.get("val_samples"),
            "bestValAcc": training_log.get("best_val_acc"),
            "config": training_log.get("config"),
            "epochHistory": training_log.get("results"),
            "dataChangesVsPrev": training_log.get("data_changes_vs_v10"),
        },
        "evaluation": {
            "threeSetAvg": full_metrics.get("three_set_avg"),
            "coreV1TestClean": full_metrics.get("core_v1_test_clean"),
            "independentData": full_metrics.get("independent_data"),
            "mergedV2ValClean": full_metrics.get("merged_v2_val_clean"),
            "independentBySource": full_metrics.get("independent_data_by_source"),
            "independentErrors": full_metrics.get("independent_data_errors"),
            "calibration": full_metrics.get("independent_data_calibration"),
        },
    }


def get_project_knowledge_index(force_refresh: bool = False) -> ProjectKnowledgeIndex:
    """Return the lazily-built project knowledge index."""
    global project_knowledge_index

    with PROJECT_QA_LOCK:
        if project_knowledge_index is None:
            project_knowledge_index = ProjectKnowledgeIndex()
        if force_refresh or not project_knowledge_index.index_ready:
            project_knowledge_index.refresh()

        return project_knowledge_index


def ensure_detector_loaded() -> HybridTextDetector | None:
    """Load the detector on demand when startup warmup did not run.

    Normal production boot still initializes the detector during FastAPI lifespan,
    but this fallback keeps auxiliary agent features usable in scripts, tests, or
    alternate hosting setups where the lifespan hook was skipped.
    """
    global detector

    if detector is not None:
        return detector

    with DETECTOR_INIT_LOCK:
        if detector is not None:
            return detector
        try:
            detector = HybridTextDetector()
        except Exception as exc:
            logger.warning("[ensure_detector_loaded] detector init failed: %s", exc)
            detector = None
        return detector


class DetectRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_DETECT_TEXT_CHARS)


class SentenceResult(BaseModel):
    text: str
    isAI: bool
    confidence: float


class TokenSpan(BaseModel):
    token: str
    charStart: int
    charEnd: int
    probAi: float


class DetectionResponse(BaseModel):
    type: str
    confidence: float
    humanPercentage: int
    aiPercentage: int
    boundary: int | None = None
    sentences: list[SentenceResult]
    tokenSpans: list[TokenSpan] | None = None
    processingTime: int
    modelVersion: str | None = None
    decisionThreshold: float | None = None
    riskFlags: list[str] | None = None
    domainHint: str | None = None
    reasonSummary: str | None = None
    reasonSignals: list[str] | None = None
    feedbackRequired: bool = True


class FeedbackRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_DETECT_TEXT_CHARS)
    predictedType: Literal["human", "ai", "mixed"]
    confirmedCorrect: bool
    confirmedLabel: Literal["human", "ai", "mixed"] | None = None
    tags: list[str] = Field(default_factory=list, max_length=16)
    note: str | None = Field(default=None, max_length=500)
    confidence: float | None = Field(default=None, ge=0.0, le=100.0)
    humanPercentage: int | None = Field(default=None, ge=0, le=100)
    aiPercentage: int | None = Field(default=None, ge=0, le=100)
    boundary: int | None = Field(default=None, ge=0)
    modelVersion: str | None = Field(default=None, max_length=128)
    domainHint: str | None = Field(default=None, max_length=64)


class FeedbackResponse(BaseModel):
    status: str
    feedbackId: str
    misclassifiedSaved: bool
    storedAt: str


class ProjectQARequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    topK: int = Field(default=5, ge=1, le=8)
    useLLM: bool = True
    forceRefresh: bool = False
    agentMode: Literal["defense", "technical", "metrics", "critical"] | None = None
    history: list[dict[str, str]] = Field(default_factory=list, max_length=10)
    analysisText: str | None = Field(default=None, max_length=4000)
    answerLength: Literal["brief", "standard", "detailed"] = "standard"
    speakingStyle: Literal["natural", "formal", "confident", "honest"] = "natural"
    speakerProfile: str | None = Field(default=None, max_length=1200)
    modelPresetId: str | None = Field(default=None, max_length=64)
    sessionId: str | None = Field(default=None, max_length=64)


def _qa_cache_key(payload: ProjectQARequest) -> str:
    return hashlib.md5(
        json.dumps(
            {
                "q": payload.question.strip(),
                "k": payload.topK,
                "mode": payload.agentMode,
                "length": payload.answerLength,
                "style": payload.speakingStyle,
                "preset": payload.modelPresetId,
                "sid": payload.sessionId,
            },
            ensure_ascii=False,
        ).encode()
    ).hexdigest()


def _qa_cache_get(payload: ProjectQARequest) -> dict[str, Any] | None:
    key = _qa_cache_key(payload)
    entry = _qa_cache.get(key)
    if entry is None:
        return None
    ts, data = entry
    if time.time() - ts > _QA_CACHE_TTL:
        _qa_cache.pop(key, None)
        return None
    return data


def _qa_cache_put(payload: ProjectQARequest, response_data: dict[str, Any]) -> None:
    key = _qa_cache_key(payload)
    if len(_qa_cache) >= _QA_CACHE_MAX:
        oldest_key = min(_qa_cache, key=lambda k: _qa_cache[k][0])
        del _qa_cache[oldest_key]
    _qa_cache[key] = (time.time(), response_data)


class ProjectQAModelPresetOption(BaseModel):
    id: str
    label: str
    provider: str
    model: str
    isDefault: bool = False


class ProjectQAModelPresetListResponse(BaseModel):
    presets: list[ProjectQAModelPresetOption]
    defaultPresetId: str


class ProjectQASource(BaseModel):
    path: str
    score: float
    excerpt: str


class ProjectQACodeReference(BaseModel):
    symbol: str
    path: str
    section: str | None = None
    snippet: str


class ProjectQAEvidenceReference(BaseModel):
    label: str
    path: str
    excerpt: str
    context: str


class ProjectQAToolTrace(BaseModel):
    tool: str
    status: Literal["used", "skipped", "unavailable"]
    detail: str


class ProjectQAResponse(BaseModel):
    answer: str
    mode: Literal["extractive", "rag"]
    agentMode: Literal["defense", "technical", "metrics", "critical"]
    answerFrame: str
    answerLength: Literal["brief", "standard", "detailed"]
    speakingStyle: Literal["natural", "formal", "confident", "honest"]
    model: str | None = None
    modelPresetId: str | None = None
    modelLabel: str | None = None
    sourceCount: int
    indexSourceCount: int
    processingTime: int
    sources: list[ProjectQASource]
    codeReferences: list[ProjectQACodeReference] = Field(default_factory=list)
    evidenceReferences: list[ProjectQAEvidenceReference] = Field(default_factory=list)
    toolTrace: list[ProjectQAToolTrace]
    suggestedQuestions: list[str]
    memorySummary: str | None = None
    effectiveSpeakerProfile: str | None = None
    sessionId: str | None = None


class ProjectQAMaterial(BaseModel):
    name: str
    path: str
    sizeBytes: int
    sourceType: str
    uploadedAt: str


class ProjectQAMaterialListResponse(BaseModel):
    total: int
    materials: list[ProjectQAMaterial]


class ProjectQAMaterialUploadResponse(BaseModel):
    status: str
    uploaded: list[ProjectQAMaterial]
    skipped: list[str]


PROJECT_QA_ALLOWED_SUFFIXES = {".md", ".txt", ".json", ".docx", ".pdf", ".pptx"}
PROJECT_QA_MAX_UPLOAD_BYTES = int(os.getenv("PROJECT_QA_MAX_UPLOAD_BYTES", str(15 * 1024 * 1024)))


def sanitize_project_material_name(filename: str) -> str:
    """Normalize user-uploaded filenames for safe local storage."""
    cleaned = Path(filename).name.strip().replace("\x00", "")
    cleaned = re.sub(r"[^A-Za-z0-9._\-\u4e00-\u9fff]+", "_", cleaned)
    cleaned = cleaned.strip("._")
    if not cleaned:
        cleaned = f"material_{uuid.uuid4().hex[:8]}"
    return cleaned[:120]


def build_project_material_record(path: Path) -> ProjectQAMaterial:
    """Convert a stored upload into API-friendly metadata."""
    stat = path.stat()
    upload_root = PATHS.ensure_dir(PATHS.project_qa_uploads_dir)
    try:
        relative_path = path.relative_to(upload_root).as_posix()
    except ValueError:
        relative_path = path.name
    return ProjectQAMaterial(
        name=path.name,
        path=relative_path,
        sizeBytes=stat.st_size,
        sourceType=path.suffix.lower().lstrip("."),
        uploadedAt=datetime.fromtimestamp(stat.st_mtime).isoformat(),
    )


def infer_domain_hint(text: str) -> str:
    if FORMAL_PATTERN.search(text):
        return "formal"
    if TECH_PATTERN.search(text):
        return "technical"
    if CASUAL_PATTERN.search(text):
        return "casual"
    return "general"


def collect_risk_flags(
    text: str,
    confidence: float,
    boundary_sentence_index: int | None,
    result_type: str,
) -> list[str]:
    flags: list[str] = []
    text_len = len(text)

    if text_len < 128:
        flags.append("short_text")
    if text_len > 2048:
        flags.append("long_text")
    if text_len > 5000:
        flags.append("extreme_length")
    if confidence < 65:
        flags.append("low_confidence")
    if TEMPLATE_LIKE_PATTERN.search(text):
        flags.append("template_like")
    if result_type == "mixed" and boundary_sentence_index is None:
        flags.append("mixed_without_boundary")
    return flags


def build_reason_analysis(
    *,
    result_type: str,
    confidence: float,
    ai_percentage: int,
    human_percentage: int,
    boundary_sentence_index: int | None,
    domain_hint: str,
    risk_flags: list[str],
) -> tuple[str, list[str]]:
    signals: list[str] = []
    score_gap = abs(ai_percentage - human_percentage)

    if result_type == "ai":
        if score_gap >= 40:
            signals.append(
                f"AI倾向明显高于人类倾向（AI {ai_percentage}% / 人类 {human_percentage}%）"
            )
        else:
            signals.append(
                f"AI倾向略高于人类倾向（AI {ai_percentage}% / 人类 {human_percentage}%）"
            )
    elif result_type == "human":
        if score_gap >= 40:
            signals.append(
                f"人类倾向明显高于AI倾向（人类 {human_percentage}% / AI {ai_percentage}%）"
            )
        else:
            signals.append(
                f"人类倾向略高于AI倾向（人类 {human_percentage}% / AI {ai_percentage}%）"
            )
    else:
        if score_gap <= 15:
            signals.append(
                f"AI与人类倾向接近（AI {ai_percentage}% / 人类 {human_percentage}%），未形成单边优势"
            )
        else:
            signals.append(
                f"文本同时出现两类特征（AI {ai_percentage}% / 人类 {human_percentage}%）"
            )

    if boundary_sentence_index is not None:
        signals.append(f"第 {boundary_sentence_index + 1} 句附近检测到明显风格切换")
    elif result_type != "mixed":
        signals.append("当前结果未发现明确的混合边界，整体风格相对连续")

    domain_hint_messages = {
        "formal": "文本偏正式通知/公告语体，属于规则感较强的写作场景",
        "technical": "文本包含较多技术术语，属于专业表达场景",
        "casual": "文本带有较明显口语化表达，风格更接近日常交流",
        "general": "文本属于通用表达场景，模型主要依赖整体语言模式判断",
    }
    signals.append(domain_hint_messages.get(domain_hint, domain_hint_messages["general"]))

    risk_flag_messages = {
        "short_text": "文本较短，模型可用线索有限，建议结合人工复核",
        "long_text": "文本较长，内部可能包含多段风格，建议重点查看句子级分析",
        "extreme_length": "文本超长，局部段落可能对整体判断产生扰动",
        "low_confidence": "当前样本处于低置信区间，结论应以人工复核为准",
        "template_like": "文本出现模板化或提示词式表达，这是模型重点关注的高风险信号",
        "mixed_without_boundary": "模型认为可能存在混合特征，但暂未定位到清晰边界",
    }
    for flag in risk_flags:
        message = risk_flag_messages.get(flag)
        if message:
            signals.append(message)

    if result_type == "ai":
        summary = (
            f"模型当前更倾向判为 AI 生成，主要依据是 AI 倾向 {ai_percentage}% "
            f"高于人类倾向 {human_percentage}%。"
        )
    elif result_type == "human":
        summary = (
            f"模型当前更倾向判为人类写作，主要依据是人类倾向 {human_percentage}% "
            f"高于 AI 倾向 {ai_percentage}%。"
        )
    else:
        if boundary_sentence_index is not None:
            summary = (
                f"模型当前更倾向判为混合文本，因为第 {boundary_sentence_index + 1} 句附近"
                "出现了风格切换。"
            )
        else:
            summary = (
                f"模型当前更倾向判为混合文本，因为 AI 倾向 {ai_percentage}% 与人类倾向 "
                f"{human_percentage}% 接近，未形成稳定单边判断。"
            )

    if "low_confidence" in risk_flags:
        summary += " 当前置信度偏低，建议优先采用人工确认。"
    elif "short_text" in risk_flags:
        summary += " 但文本较短，解释依据相对有限。"

    summary += " 这些依据属于解释性提示，不等同于底层参数级归因。"

    return summary, signals[:6]


def split_sentences(text: str) -> list[str]:
    parts = SENTENCE_SPLIT_PATTERN.split(text)
    temp_sentences: list[str] = []
    current = ""

    for part in parts:
        if SENTENCE_SPLIT_PATTERN.match(part):
            current += part
            temp_sentences.append(current)
            current = ""
        else:
            if current:
                temp_sentences.append(current)
            current = part
    if current:
        temp_sentences.append(current)

    return [sentence for sentence in temp_sentences if sentence.strip()]


def build_feedback_override_response(
    *,
    text: str,
    override: dict[str, Any],
    processing_time: int,
) -> DetectionResponse:
    """Build a response from an exact-match manual correction record."""
    result_type = override["confirmed_label"]
    boundary_sentence_index = override.get("boundary")
    final_sentences = split_sentences(text)

    if result_type == "human":
        human_percentage = 100
        ai_percentage = 0
        sentence_results = [
            SentenceResult(text=sentence, isAI=False, confidence=100.0)
            for sentence in final_sentences
        ]
    elif result_type == "ai":
        human_percentage = 0
        ai_percentage = 100
        sentence_results = [
            SentenceResult(text=sentence, isAI=True, confidence=100.0)
            for sentence in final_sentences
        ]
    else:
        human_percentage = 50
        ai_percentage = 50
        if boundary_sentence_index is not None and not (
            0 < boundary_sentence_index < len(final_sentences)
        ):
            boundary_sentence_index = None

        if boundary_sentence_index is not None:
            sentence_results = [
                SentenceResult(
                    text=sentence,
                    isAI=idx >= boundary_sentence_index,
                    confidence=100.0,
                )
                for idx, sentence in enumerate(final_sentences)
            ]
        else:
            per_sentence = detector.classify_batch(final_sentences) if final_sentences else []
            sentence_results = []
            for idx, sentence in enumerate(final_sentences):
                sent_probs = per_sentence[idx] if idx < len(per_sentence) else None
                sent_confidence = (sent_probs["confidence"] * 100) if sent_probs else 100.0
                is_ai = (sent_probs["prob_ai"] >= DECISION_THRESHOLD) if sent_probs else False
                sentence_results.append(
                    SentenceResult(
                        text=sentence,
                        isAI=is_ai,
                        confidence=sent_confidence,
                    )
                )

    risk_flags = ["feedback_override_exact_match"]
    if boundary_sentence_index is not None:
        risk_flags.append("boundary_from_manual_feedback")

    if result_type == "human":
        type_zh = "人类写作"
    elif result_type == "ai":
        type_zh = "AI生成"
    else:
        type_zh = "混合文本"

    reason_summary = (
        "该文本命中人工确认误判记忆库中的完全相同样本，因此本次直接返回历史人工确认标签，"
        "不再沿用模型的原始顶层判定。"
    )
    reason_signals = [
        "命中历史人工确认的完全相同文本",
        f"沿用人工确认标签：{type_zh}",
        "该覆盖仅对 exact match 生效，不对润色、续写或改写文本生效",
    ]
    if boundary_sentence_index is not None:
        reason_signals.append(f"沿用历史边界信息：第 {boundary_sentence_index + 1} 句附近")

    return DetectionResponse(
        type=result_type,
        confidence=100.0,
        humanPercentage=human_percentage,
        aiPercentage=ai_percentage,
        boundary=boundary_sentence_index,
        sentences=sentence_results,
        tokenSpans=None,
        processingTime=processing_time,
        modelVersion=MODEL_VERSION,
        decisionThreshold=DECISION_THRESHOLD,
        riskFlags=risk_flags,
        domainHint=override.get("domain_hint") or infer_domain_hint(text),
        reasonSummary=reason_summary,
        reasonSignals=reason_signals,
        feedbackRequired=True,
    )


@app.post(
    "/api/detect",
    response_model=DetectionResponse,
    response_model_exclude_none=True,
)
def detect_text(
    payload: DetectRequest,
    http_request: Request,
    x_internal_token: str | None = Header(default=None, alias="X-Internal-Token"),
) -> DetectionResponse:
    verify_internal_token(x_internal_token)
    enforce_rate_limit(http_request, "detect", DETECT_RATE_LIMIT_PER_WINDOW)

    if not detector:
        raise HTTPException(status_code=500, detail="Model not initialized")

    start_time = time.time()
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")

    feedback_override = lookup_feedback_override(text=text)
    if feedback_override is not None:
        processing_time = int((time.time() - start_time) * 1000)
        return build_feedback_override_response(
            text=text,
            override=feedback_override,
            processing_time=processing_time,
        )

    cls_result = detector.classify(text)
    confidence = float(cls_result["confidence"]) * 100
    prob_ai = float(cls_result["prob_ai"])
    prob_human = float(cls_result["prob_human"])

    ai_percentage = int(prob_ai * 100)
    human_percentage = int(prob_human * 100)

    result_type = "mixed"
    boundary_char = None

    if prob_ai >= DECISION_THRESHOLD:
        result_type = "ai"
    elif prob_human >= DECISION_THRESHOLD:
        result_type = "human"

    # Span detector now triggers by length threshold rather than classifier result.
    # This catches mixed texts where the classifier confidently mis-labels one side.
    token_spans: list[dict[str, Any]] = []
    if len(text) >= SPAN_TRIGGER_MIN_CHARS:
        boundary_res = detector.detect_boundary(text)
        if boundary_res["boundary_char"] is not None:
            boundary_char = int(boundary_res["boundary_char"])
        token_spans = boundary_res.get("tokenSpans") or []

    final_sentences = split_sentences(text)

    boundary_sentence_index = None
    running_char_count = 0
    for idx, sentence in enumerate(final_sentences):
        sent_len = len(sentence)
        if (
            boundary_char is not None
            and running_char_count <= boundary_char < running_char_count + sent_len
        ):
            boundary_sentence_index = idx
        running_char_count += sent_len

    # If the span detector found a valid boundary with content on both sides,
    # promote the result to "mixed" regardless of the classifier's single label.
    if boundary_sentence_index is not None and 0 < boundary_sentence_index < len(final_sentences):
        result_type = "mixed"
    else:
        # No valid boundary — clear stale boundary_char to avoid misleading downstream.
        boundary_char = None
        boundary_sentence_index = None

    # Per-sentence classification (batched in a single forward pass).
    per_sentence: list[dict[str, float]] = []
    if final_sentences:
        per_sentence = detector.classify_batch(final_sentences)

    sentence_results: list[SentenceResult] = []
    for idx, sentence in enumerate(final_sentences):
        sent_probs = per_sentence[idx] if idx < len(per_sentence) else None
        if sent_probs is not None:
            sent_prob_ai = sent_probs["prob_ai"]
            sent_confidence = sent_probs["confidence"] * 100
            if result_type == "mixed" and boundary_sentence_index is not None:
                is_ai = idx >= boundary_sentence_index
            else:
                is_ai = sent_prob_ai >= DECISION_THRESHOLD
        else:
            sent_confidence = confidence
            is_ai = result_type == "ai"

        sentence_results.append(
            SentenceResult(
                text=sentence,
                isAI=is_ai,
                confidence=sent_confidence,
            )
        )

    processing_time = int((time.time() - start_time) * 1000)

    model_version = MODEL_VERSION
    decision_threshold = DECISION_THRESHOLD
    domain_hint = infer_domain_hint(text)
    risk_flags = collect_risk_flags(
        text=text,
        confidence=confidence,
        boundary_sentence_index=boundary_sentence_index,
        result_type=result_type,
    )
    reason_summary, reason_signals = build_reason_analysis(
        result_type=result_type,
        confidence=confidence,
        ai_percentage=ai_percentage,
        human_percentage=human_percentage,
        boundary_sentence_index=boundary_sentence_index,
        domain_hint=domain_hint,
        risk_flags=risk_flags,
    )

    exposed_token_spans: list[TokenSpan] | None = None
    if EXPOSE_TOKEN_PROBS and token_spans:
        exposed_token_spans = [
            TokenSpan(
                token=span["token"],
                charStart=span["charStart"],
                charEnd=span["charEnd"],
                probAi=span["probAi"],
            )
            for span in token_spans
        ]

    return DetectionResponse(
        type=result_type,
        confidence=confidence,
        humanPercentage=human_percentage,
        aiPercentage=ai_percentage,
        boundary=boundary_sentence_index,
        sentences=sentence_results,
        tokenSpans=exposed_token_spans,
        processingTime=processing_time,
        modelVersion=model_version,
        decisionThreshold=decision_threshold,
        riskFlags=risk_flags,
        domainHint=domain_hint,
        reasonSummary=reason_summary,
        reasonSignals=reason_signals,
        feedbackRequired=True,
    )


BATCH_MAX_ITEMS = int(os.getenv("DETECT_BATCH_MAX_ITEMS", "50"))


class BatchDetectRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=BATCH_MAX_ITEMS)


class BatchItemResult(BaseModel):
    index: int
    type: str
    confidence: float
    humanPercentage: int
    aiPercentage: int
    charCount: int
    error: str | None = None


class BatchDetectResponse(BaseModel):
    modelVersion: str
    decisionThreshold: float
    total: int
    processingTime: int
    results: list[BatchItemResult]


@app.post(
    "/api/detect/batch",
    response_model=BatchDetectResponse,
    response_model_exclude_none=True,
)
def detect_text_batch(
    payload: BatchDetectRequest,
    http_request: Request,
    x_internal_token: str | None = Header(default=None, alias="X-Internal-Token"),
) -> BatchDetectResponse:
    """Classify up to N texts in a single call. Uses a single batched forward pass.

    Intended for reproducibility demos (e.g. run 10 thesis test cases) —
    each item consumes one detect rate-limit slot to prevent abuse.
    """
    verify_internal_token(x_internal_token)

    for _ in range(len(payload.texts)):
        enforce_rate_limit(http_request, "detect", DETECT_RATE_LIMIT_PER_WINDOW)

    if not detector:
        raise HTTPException(status_code=500, detail="Model not initialized")

    start_time = time.time()
    cleaned: list[tuple[int, str]] = []
    results: list[BatchItemResult] = [
        BatchItemResult(
            index=idx,
            type="invalid",
            confidence=0.0,
            humanPercentage=0,
            aiPercentage=0,
            charCount=0,
            error="empty text",
        )
        for idx in range(len(payload.texts))
    ]

    for idx, raw_text in enumerate(payload.texts):
        stripped = raw_text.strip()
        if not stripped:
            continue
        if len(stripped) > MAX_DETECT_TEXT_CHARS:
            results[idx] = BatchItemResult(
                index=idx,
                type="invalid",
                confidence=0.0,
                humanPercentage=0,
                aiPercentage=0,
                charCount=len(stripped),
                error=f"text exceeds {MAX_DETECT_TEXT_CHARS} chars",
            )
            continue
        cleaned.append((idx, stripped))

    if cleaned:
        batch_probs = detector.classify_batch([text for _, text in cleaned])
        for (idx, text), probs in zip(cleaned, batch_probs, strict=True):
            prob_ai = probs["prob_ai"]
            prob_human = probs["prob_human"]
            if prob_ai >= DECISION_THRESHOLD:
                item_type = "ai"
            elif prob_human >= DECISION_THRESHOLD:
                item_type = "human"
            else:
                item_type = "mixed"
            results[idx] = BatchItemResult(
                index=idx,
                type=item_type,
                confidence=probs["confidence"] * 100,
                humanPercentage=int(prob_human * 100),
                aiPercentage=int(prob_ai * 100),
                charCount=len(text),
            )

    processing_time = int((time.time() - start_time) * 1000)

    return BatchDetectResponse(
        modelVersion=MODEL_VERSION,
        decisionThreshold=DECISION_THRESHOLD,
        total=len(payload.texts),
        processingTime=processing_time,
        results=results,
    )


@app.post(
    "/api/feedback",
    response_model=FeedbackResponse,
)
def submit_feedback(
    payload: FeedbackRequest,
    http_request: Request,
    x_internal_token: str | None = Header(default=None, alias="X-Internal-Token"),
) -> FeedbackResponse:
    """Persist manual confirmation results for closed-loop data collection."""
    verify_internal_token(x_internal_token)
    enforce_rate_limit(http_request, "detect", DETECT_RATE_LIMIT_PER_WINDOW)

    if not payload.confirmedCorrect and payload.confirmedLabel is None:
        raise HTTPException(
            status_code=422,
            detail="confirmedLabel is required when confirmedCorrect is false",
        )

    try:
        stored = persist_feedback(
            text=payload.text,
            predicted_label=payload.predictedType,
            confirmed_correct=payload.confirmedCorrect,
            confirmed_label=payload.confirmedLabel,
            tags=payload.tags,
            note=payload.note,
            source="api_feedback",
            model_version=payload.modelVersion,
            confidence=payload.confidence,
            ai_percentage=payload.aiPercentage,
            human_percentage=payload.humanPercentage,
            boundary=payload.boundary,
            domain_hint=payload.domainHint,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OSError as exc:
        logger.error("[submit_feedback] failed to persist feedback: %s", exc)
        raise HTTPException(status_code=500, detail="Feedback storage unavailable") from exc
    except Exception as exc:
        logger.exception("[submit_feedback] unexpected feedback submission error")
        raise HTTPException(status_code=500, detail="Feedback submission failed") from exc

    return FeedbackResponse(
        status="ok",
        feedbackId=stored["feedback_id"],
        misclassifiedSaved=stored["misclassified_saved"],
        storedAt=stored["created_at"],
    )


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[dict[str, Any]] = Field(min_length=1, max_length=CHAT_MAX_MESSAGES)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, ge=1, le=CHAT_MAX_TOKENS)


AGENT_MODE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "metrics": ("准确率", "f1", "ece", "指标", "多少", "性能"),
    "critical": ("局限", "缺点", "风险", "不足", "质疑", "过拟合"),
    "technical": ("原理", "为什么", "架构", "bert", "sep", "边界", "训练", "技术"),
}

_AGENT_MODE_NAMES = ("metrics", "critical", "technical", "defense")


def _score_agent_mode(question: str) -> tuple[str, float]:
    """Score each mode by keyword hits; return (best_mode, confidence)."""
    lowered = question.lower()
    scores: dict[str, int] = {m: 0 for m in AGENT_MODE_KEYWORDS}
    for mode, keywords in AGENT_MODE_KEYWORDS.items():
        scores[mode] = sum(1 for kw in keywords if kw in lowered)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_mode, best_score = ranked[0]
    if best_score == 0:
        return "defense", 0.0
    second_score = ranked[1][1]
    confidence = 1.0 if second_score == 0 else best_score / (best_score + second_score + 0.01)
    return best_mode, confidence


async def _llm_classify_agent_mode(question: str) -> str | None:
    """Use a lightweight LLM call to classify the question into one of 4 modes."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    api_base = os.getenv("OPENAI_BASE_URL", "https://api.hotaruapi.top/v1")
    model_name = DEFAULT_CHAT_MODEL.strip()
    if not model_name:
        return None
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            resp = await client.post(
                f"{api_base}/chat/completions",
                headers=build_upstream_chat_headers(api_key),
                json={
                    "model": model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "你是分类器。根据用户关于AI文本检测项目的问题，"
                                "回复且仅回复以下之一：metrics critical technical defense。"
                                "metrics=指标/数字/性能对比, critical=局限/风险/质疑/不足, "
                                "technical=原理/架构/机制/训练, defense=其他/答辩准备/项目介绍。"
                            ),
                        },
                        {"role": "user", "content": question[:200]},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 8,
                },
            )
        if resp.status_code != 200:
            return None
        text = (
            resp.json()
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
            .lower()
        )
        for mode in _AGENT_MODE_NAMES:
            if mode in text:
                return mode
        return None
    except Exception:
        return None


async def infer_project_agent_mode(
    question: str,
    requested_mode: Literal["defense", "technical", "metrics", "critical"] | None,
) -> Literal["defense", "technical", "metrics", "critical"]:
    """Pick the most suitable defense-copilot mode for the question."""
    if requested_mode is not None:
        return requested_mode

    best_mode, confidence = _score_agent_mode(question)
    if confidence >= 0.8:
        return cast(Literal["defense", "technical", "metrics", "critical"], best_mode)

    llm_mode = await _llm_classify_agent_mode(question)
    if llm_mode is not None:
        return cast(Literal["defense", "technical", "metrics", "critical"], llm_mode)
    return cast(
        Literal["defense", "technical", "metrics", "critical"],
        best_mode if confidence > 0 else "defense",
    )


def summarize_project_agent_history(history: list[dict[str, str]]) -> str | None:
    """Compress recent turns into a lightweight memory summary for prompting."""
    if not history:
        return None

    lines: list[str] = []
    for message in history[-6:]:
        role = (message.get("role") or "user").strip().lower()
        content = (message.get("content") or "").strip()
        if not content:
            continue
        label = "用户" if role == "user" else "助教"
        compact = re.sub(r"\s+", " ", content)
        if len(compact) > 120:
            compact = compact[:117].rstrip() + "..."
        lines.append(f"- {label}: {compact}")

    if not lines:
        return None
    return "\n".join(lines)


def resolve_project_speaker_profile(custom_profile: str | None) -> str:
    """Resolve the effective first-person defense profile for prompting."""
    cleaned = (custom_profile or "").strip()
    return cleaned or DEFAULT_DEFENSE_PROFILE


def resolve_project_answer_style(
    answer_length: Literal["brief", "standard", "detailed"],
    speaking_style: Literal["natural", "formal", "confident", "honest"],
) -> tuple[str, str]:
    """Map UI-level style controls to concrete prompting instructions."""
    length_instruction = {
        "brief": "长度控制：适合现场快答，控制在 3 到 5 句，不展开次要背景。",
        "standard": "长度控制：适合常规答辩回答，1 段结论 + 2 到 4 条支撑点。",
        "detailed": "长度控制：适合深入解释，可以补充背景、对比、边界和改进方向。",
    }[answer_length]
    style_instruction = {
        "natural": "表达风格：自然清楚，默认像向老师当面解释项目；若问题是在准备答辩话术，再切换为学生现场口语。",
        "formal": "表达风格：更正式规范，适合论文汇报口径，减少口语词。",
        "confident": "表达风格：结论更明确，先亮观点，再解释依据，但不要夸大。",
        "honest": "表达风格：更加审慎诚实，明确证据边界，避免把不确定内容说满。",
    }[speaking_style]
    return length_instruction, style_instruction


def should_use_project_model_info(
    question: str,
    agent_mode: Literal["defense", "technical", "metrics", "critical"],
) -> bool:
    """Decide whether runtime model metadata should be injected."""
    if agent_mode in {"metrics", "critical"}:
        return True

    lowered = question.lower()
    return any(
        keyword in lowered
        for keyword in ("v11", "v10", "模型", "训练", "温度", "校准", "准确率", "样本")
    )


def build_project_model_snapshot() -> str | None:
    """Build a compact textual snapshot from runtime model metadata."""
    if not CLASSIFIER_METRICS and not CLASSIFIER_TRAINING_LOG:
        return None

    lines = [f"当前生产模型: {MODEL_VERSION}"]

    if CLASSIFIER_METRICS.get("three_set_avg") is not None:
        lines.append(f"三集平均准确率: {CLASSIFIER_METRICS['three_set_avg']}")
    if CLASSIFIER_METRICS.get("independent_accuracy") is not None:
        lines.append(f"独立评估集准确率: {CLASSIFIER_METRICS['independent_accuracy']}")
    if CLASSIFIER_METRICS.get("ece_after") is not None:
        lines.append(f"ECE: {CLASSIFIER_METRICS['ece_after']}")
    if CLASSIFIER_METRICS.get("optimal_temperature") is not None:
        lines.append(f"温度缩放 T: {CLASSIFIER_METRICS['optimal_temperature']}")
    if CLASSIFIER_TRAINING_LOG.get("train_samples") is not None:
        lines.append(f"训练样本数: {CLASSIFIER_TRAINING_LOG['train_samples']}")
    if CLASSIFIER_TRAINING_LOG.get("strategy"):
        lines.append(f"训练策略: {CLASSIFIER_TRAINING_LOG['strategy']}")

    return "\n".join(lines)


def format_project_metric_percent(value: Any) -> str | None:
    """Render runtime metrics as stable percent strings."""
    if not isinstance(value, (int, float)):
        return None

    numeric = float(value)
    if numeric <= 1.0:
        numeric *= 100.0
    return f"{numeric:.2f}%"


def build_project_local_answer(
    question: str,
    *,
    agent_mode: Literal["defense", "technical", "metrics", "critical"],
    answer_frame_title: str,
    hits: list[KnowledgeHit],
    model_snapshot: str | None,
) -> str:
    """Build a human-readable fallback answer when LLM synthesis is unavailable."""
    lowered = question.lower()
    model_name = MODEL_VERSION or "bert_v11c_boundary_fix"
    three_set_avg = format_project_metric_percent(CLASSIFIER_METRICS.get("three_set_avg"))
    independent_accuracy = format_project_metric_percent(
        CLASSIFIER_METRICS.get("independent_accuracy")
    )
    validation_accuracy = None
    if isinstance(CLASSIFIER_METRICS.get("_full"), dict):
        validation_accuracy = format_project_metric_percent(
            CLASSIFIER_METRICS["_full"].get("core_v1_test_clean")
        )
    token_accuracy = "96.69%"
    calibration_t = CLASSIFIER_METRICS.get("optimal_temperature")
    ece_after = CLASSIFIER_METRICS.get("ece_after")
    train_samples = CLASSIFIER_TRAINING_LOG.get("train_samples")

    if answer_frame_title == "30秒总述":
        return (
            "根据当前项目资料，可概括为：该系统聚焦中文 AI 生成文本检测，目标是判断一段文本是人写、AI 写，"
            "还是人机混合。方法上，以 BERT 微调为核心，采用“分类检测 + 边界定位”的双层方案，"
            "并结合 [SEP] 边界标记来增强混合文本识别。结果上，当前推荐模型"
            f" {model_name} 的三集平均准确率约 {three_set_avg or '98%+'}，"
            f"独立评估集准确率约 {independent_accuracy or '98%+'}。应用上，这套系统既能做答辩演示，"
            "也可以服务于作业审核、内容风控这类中文 AI 文本识别场景。"
        )

    if answer_frame_title == "创新点三段式":
        return (
            "根据当前项目资料，可将创新概括为三点。第一，针对中文混合文本场景，"
            "系统没有只做二分类，而是把“是否为 AI”判断和“边界在哪里”定位结合起来，形成双层检测结构。"
            "第二，引入了 [SEP] 边界标记，让模型更容易学习人类段落和 AI 段落的切换位置。"
            "第三，除了训练集结果，还补了独立评估、校准和答辩演示链路，让模型效果、"
            "置信度和实际展示都能闭环。"
        )

    if answer_frame_title == "技术选型答辩":
        return (
            "根据当前项目资料，选择 BERT 而不是 GPT、LLaMA 这类大模型，主要有三点原因。"
            "第一，这个任务本质上是判别任务，不是生成任务，BERT 在文本分类上更直接。"
            "第二，BERT 微调的训练和部署成本更低，更适合本科毕设这种可复现、可落地的方案。"
            "第三，该项目还要做边界标记和 span 级定位，BERT 这类编码器结构更容易和 [SEP] 标记、"
            "Token 级检测器配合。"
        )

    if answer_frame_title == "指标口径答辩" or agent_mode == "metrics":
        detail_parts = [f"当前推荐模型是 {model_name}。"]
        if three_set_avg:
            detail_parts.append(f"三集平均准确率是 {three_set_avg}。")
        if independent_accuracy:
            detail_parts.append(f"独立评估集准确率是 {independent_accuracy}。")
        if validation_accuracy:
            detail_parts.append(f"验证集准确率大约是 {validation_accuracy}。")
        detail_parts.append(f"Token 级边界检测准确率约为 {token_accuracy}。")
        if calibration_t is not None and ece_after is not None:
            detail_parts.append(f"另外做了温度缩放校准，T={calibration_t}，ECE={ece_after}。")
        if train_samples:
            detail_parts.append(f"训练样本规模大约是 {train_samples} 条。")
        return "根据当前项目资料，可直接概括为：" + "".join(detail_parts)

    if answer_frame_title == "承认-解释-改进" or agent_mode == "critical":
        return (
            "根据当前项目资料，这个项目目前确实还有泛化和数据边界上的局限，不应被表述为已经完全解决。"
            "从现有结果看，当前模型在仓库记录的多组评估里表现比较稳定，"
            f"三集平均准确率约 {three_set_avg or '98%+'}，说明它在已覆盖场景下是有效的；"
            "但如果换到更新的模型来源、不同题材或者更强的规避写法，性能仍然可能波动。"
            "因此更稳妥的结论是：现阶段系统已经通过独立评估、校准和边界修复尽量降低过拟合风险，"
            "下一步还需要继续补跨域数据、补新模型样本，并做更严格的外部验证。"
        )

    if model_snapshot and any(
        token in lowered for token in ("模型", "准确率", "训练", "样本", "v11", "v10")
    ):
        return "根据当前运行时模型信息，可概括为：\n" + model_snapshot.replace("\n", "；") + "。"

    answer = build_project_local_answer(
        question,
        agent_mode=agent_mode,
        answer_frame_title=answer_frame_title,
        hits=hits,
        model_snapshot=model_snapshot,
    )
    if answer.startswith("根据仓库当前资料，可以先这样回答："):
        answer = answer.replace(
            "根据仓库当前资料，可以先这样回答：", "根据仓库当前资料，可概括为：", 1
        )
    return answer


def run_project_agent_live_detection(text: str) -> str | None:
    """Optionally analyze pasted text with the live detector."""
    cleaned = text.strip()
    if not cleaned:
        return None
    active_detector = ensure_detector_loaded()
    if active_detector is None:
        return "实时检测工具当前不可用：后端检测器尚未初始化。"

    cls_result = active_detector.classify(cleaned)
    confidence = float(cls_result["confidence"]) * 100
    prob_ai = float(cls_result["prob_ai"])
    prob_human = float(cls_result["prob_human"])
    result_type = "mixed"
    if prob_ai >= DECISION_THRESHOLD:
        result_type = "ai"
    elif prob_human >= DECISION_THRESHOLD:
        result_type = "human"

    boundary_hint = "未触发边界检测"
    if len(cleaned) >= SPAN_TRIGGER_MIN_CHARS:
        boundary_res = active_detector.detect_boundary(cleaned)
        boundary_char = boundary_res.get("boundary_char")
        if boundary_char is not None:
            boundary_hint = f"检测到潜在边界字符位置: {boundary_char}"
        else:
            boundary_hint = "已触发边界检测，但未定位到稳定边界"

    return (
        f"实时检测结果: {result_type}；置信度 {confidence:.2f}%；"
        f"AI 倾向 {prob_ai * 100:.1f}%；人类倾向 {prob_human * 100:.1f}%；"
        f"{boundary_hint}；文本域提示: {infer_domain_hint(cleaned)}。"
    )


_SUGGESTION_POOL: dict[str, list[str]] = {
    "defense": [
        "本项目的研究目标、方法与应用价值分别是什么？",
        "本文的核心创新点可以概括为哪三点？",
        "该系统的实际应用场景和工程价值体现在哪里？",
        "如果老师不懂技术，你怎么一句话讲清楚这个项目？",
        "你的数据是怎么构建的？覆盖了哪些模型和来源？",
    ],
    "technical": [
        "为什么 [SEP] 边界标记能够提升混合文本检测效果？",
        "为什么本文选择 BERT 而不是 GPT / LLaMA 一类生成模型？",
        "双层检测架构中，分类器与边界检测器分别承担什么作用？",
        "为什么不直接用零样本方法或水印检测？",
        "最大长度设为 256 是怎么决定的？",
    ],
    "metrics": [
        "V11c 相比 V10 的性能提升主要来自哪些因素？",
        "Temperature Scaling 与 ECE 在本文中分别说明什么？",
        "当前推荐模型的核心指标有哪些，它们分别代表什么？",
        "98.69% 和 98.56% 这两个数字为什么都出现？",
        "99.28% 的召回率意味着什么？",
    ],
    "critical": [
        "本项目目前的主要局限性是什么？",
        "数据集设计如何支撑跨模型泛化能力评估？",
        "如果面对分布外新模型，当前方法可能出现哪些风险？",
        "你怎么证明这不是过拟合？",
        "准确率这么高，不会是数据泄露吗？",
    ],
}

_FOLLOWUP_POOL = [
    "能展开讲讲数据治理具体做了什么吗？",
    "对比一下你的方法和 BERT-BiGRU 的区别？",
    "混合文本检测的边界定位准确率如何？",
    "如果换成英文文本，当前方法还能直接用吗？",
    "在误报和漏报之间，本文是如何取舍的？",
    "你这个系统到底能不能真正拿来用？",
]


def build_project_agent_suggestions(
    agent_mode: Literal["defense", "technical", "metrics", "critical"],
    asked_questions: list[str] | None = None,
) -> list[str]:
    """Return context-aware next-question suggestions."""
    pool = list(_SUGGESTION_POOL.get(agent_mode, _SUGGESTION_POOL["defense"]))
    asked_lower = {q.strip().lower() for q in (asked_questions or [])}
    pool = [s for s in pool if s.strip().lower() not in asked_lower]
    followups = [f for f in _FOLLOWUP_POOL if f.strip().lower() not in asked_lower]
    result = pool[:2]
    if followups:
        result.append(followups[0])
    return result or _SUGGESTION_POOL[agent_mode][:3]


def select_project_answer_frame(
    question: str,
    agent_mode: Literal["defense", "technical", "metrics", "critical"],
) -> tuple[str, str]:
    """Choose a structured answer frame for common defense questions."""
    lowered = question.lower()

    if any(
        token in lowered
        for token in ("30 秒", "30秒", "一分钟", "概括", "介绍整个项目", "整体介绍")
    ):
        return (
            "30秒总述",
            "回答结构：1 句话讲课题目标；1 句话讲方法；1 句话讲结果；1 句话讲应用价值。",
        )
    if any(token in lowered for token in ("创新", "创新点", "贡献", "亮点")):
        return (
            "创新点三段式",
            "回答结构：先给总判断；再列 2 到 3 个创新点；最后补一句相对已有方法的区别。",
        )
    if any(token in lowered for token in ("为什么选 bert", "为什么用 bert", "gpt", "llama")):
        return (
            "技术选型答辩",
            "回答结构：任务类型匹配 -> 工程部署成本 -> 与本项目机制（如 [SEP]）的适配性。",
        )
    if any(token in lowered for token in ("准确率", "f1", "ece", "指标", "多少", "提升了")):
        return (
            "指标口径答辩",
            "回答结构：先报核心数字；再做版本对比；最后解释这些数字说明了什么。",
        )
    if any(token in lowered for token in ("局限", "风险", "不足", "质疑", "过拟合", "泛化")):
        return (
            "承认-解释-改进",
            "回答结构：先承认问题；再说明证据和原因；最后给出当前补救和未来改进。",
        )
    if any(token in lowered for token in ("对比", "比较", "区别", "差异", "vs", "和.*比")):
        return (
            "方案对比答辩",
            "回答结构：先说对比维度；再逐维度分析差异；最后给出选型结论和取舍说明。",
        )
    if any(
        token in lowered for token in ("数据治理", "数据清洗", "训练集", "v10", "v11", "数据中心")
    ):
        return (
            "数据治理答辩",
            "回答结构：先说问题（标签噪声/弱域覆盖/长文缺失）；再说做了什么（清/补/修）；最后说效果（控制变量实验结果）。",
        )
    if any(token in lowered for token in ("跨模型", "泛化", "新模型", "gpt-4", "gpt-5", "llama")):
        return (
            "泛化能力答辩",
            "回答结构：先说训练覆盖范围；再说独立评估集证据；最后说已知边界和持续扩展计划。",
        )
    if any(token in lowered for token in ("演示", "展示", "跑一下", "试一下", "现场检测")):
        return (
            "工程演示引导",
            "回答结构：简述系统能力 -> 指出演示入口 -> 说明预期结果 -> 补充边界情况。",
        )
    if any(
        token in lowered
        for token in (
            "边界定位准确率",
            "span检测",
            "混合文本检测",
            "续写检测",
            "边界标记机制",
            "边界机制",
        )
    ):
        return (
            "边界检测答辩",
            "回答结构：先说双层架构动机；再讲 [SEP] 机制原理；最后报边界定位指标和适用场景。",
        )
    if agent_mode == "defense":
        return (
            "标准说明口径",
            "回答结构：先给结论；再给 2 到 3 条支撑点；最后补一句该结论的适用边界或延伸说明。",
        )
    if agent_mode == "technical":
        return (
            "原理解释链",
            "回答结构：先讲要解决的问题；再讲机制；最后讲机制为什么带来效果提升。",
        )
    if agent_mode == "metrics":
        return (
            "结果解释链",
            "回答结构：数字 -> 对比 -> 解释 -> 谨慎结论。",
        )
    return (
        "风险回应口径",
        "回答结构：先承认边界；再解释现状；最后讲改进路线，避免硬辩。",
    )


def build_project_agent_evidence_blocks(hits: list[KnowledgeHit]) -> list[str]:
    """Build richer evidence blocks for the LLM than the UI excerpts.

    The UI only needs short excerpts, but the copilot should see the actual
    retrieved chunk content; otherwise numeric tables and short metric lines can
    be truncated away before synthesis.
    """
    blocks: list[str] = []
    for index, hit in enumerate(hits, start=1):
        raw_content = getattr(hit.chunk, "content", None) or getattr(hit, "excerpt", "")
        content = re.sub(r"\s+", " ", raw_content).strip()
        if len(content) > 1200:
            content = content[:1197].rstrip() + "..."
        blocks.append(f"[{index}] {hit.chunk.path}\n{content}")
    return blocks


def build_project_qa_messages(
    *,
    question: str,
    evidence_blocks: list[str],
    agent_mode: Literal["defense", "technical", "metrics", "critical"],
    answer_frame_title: str,
    answer_frame_instruction: str,
    answer_length_instruction: str,
    speaking_style_instruction: str,
    speaker_profile: str,
    history_summary: str | None,
    model_snapshot: str | None,
    live_detection_summary: str | None,
    tool_trace: list[ProjectQAToolTrace],
) -> list[dict[str, str]]:
    """Create an evidence-grounded prompt for the defense copilot agent."""
    evidence_text = "\n\n".join(evidence_blocks)
    tool_text = "\n".join(f"- {item.tool}: {item.status} ({item.detail})" for item in tool_trace)
    mode_instruction = {
        "defense": "回答要像项目说明与学术解释，先给结论，再给3条以内支撑点。",
        "technical": "回答要讲原理、机制和因果链，不要只背结论。",
        "metrics": "回答要优先给指标、对比关系和这些指标说明了什么。",
        "critical": "回答要先承认局限，再解释原因，最后给出改进方向，避免硬辩。",
    }[agent_mode]

    user_sections = [f"问题：{question}"]
    if history_summary:
        user_sections.append(f"最近对话摘要：\n{history_summary}")
    if model_snapshot:
        user_sections.append(f"运行时模型信息：\n{model_snapshot}")
    if live_detection_summary:
        user_sections.append(f"实时检测工具输出：\n{live_detection_summary}")
    if tool_text:
        user_sections.append(f"本轮工具轨迹：\n{tool_text}")
    user_sections.append(f"仓库证据：\n{evidence_text or '当前未命中仓库证据。'}")

    return [
        {
            "role": "system",
            "content": (
                "你是这个项目的讲解助手，面向答辩评委和指导老师。"
                f"项目背景与学生身份：{speaker_profile}"
                "你只能根据给定的仓库证据回答，不要编造仓库里没有出现的事实。"
                f"当前模式是 {agent_mode}。{mode_instruction}"
                f"当前回答模板是：{answer_frame_title}。{answer_frame_instruction}"
                f"{answer_length_instruction}{speaking_style_instruction}"
                "\n## 回答规范\n"
                "1. 像学生当面给老师讲解一样自然，不要像百科条目\n"
                "2. 先给结论，再给 2-4 条依据，每条都要标注来源 [1] [2]\n"
                "3. 提到具体实现时，标注文件路径（如 `api/api.py:L45`）或用行内代码引用\n"
                "4. 如果评委问’带我看’或’给我展示’，给出文件路径和关键行号\n"
                "5. 如果证据不足，直接说’当前仓库里没有足够证据支持这个结论’\n"
                "6. 默认用客观说明口吻；如果问题明显是’我该怎么说’类，切换第一人称\n"
                "7. 不要附加’老师继续追问’或’您还可以问’之类的提示语"
            ),
        },
        {
            "role": "user",
            "content": "\n\n".join(user_sections),
        },
    ]


def normalize_openai_text_payload(payload: Any) -> str:
    """Extract plain text from string or block-based OpenAI-compatible payloads."""
    if isinstance(payload, str):
        return payload

    if not isinstance(payload, list):
        return ""

    parts: list[str] = []
    for item in payload:
        if isinstance(item, str):
            parts.append(item)
            continue
        if not isinstance(item, dict):
            continue

        text = item.get("text")
        if isinstance(text, str):
            parts.append(text)
            continue

        nested_content = item.get("content")
        if isinstance(nested_content, str):
            parts.append(nested_content)

    return "".join(parts)


def extract_openai_message_field(message: dict[str, Any], field_name: str) -> str:
    """Read a text field from an OpenAI-compatible message payload."""
    return normalize_openai_text_payload(message.get(field_name)) or (
        message.get(field_name) if isinstance(message.get(field_name), str) else ""
    )


def extract_openai_delta_text(delta: dict[str, Any]) -> tuple[str, str]:
    """Extract answer and reasoning deltas from streamed OpenAI-compatible payloads."""
    answer_delta = extract_openai_message_field(delta, "content")
    reasoning_delta = extract_openai_message_field(delta, "reasoning_content")

    content_blocks = delta.get("content")
    if isinstance(content_blocks, list):
        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            block_text = ""
            text_value = block.get("text")
            if isinstance(text_value, str):
                block_text = text_value
            elif isinstance(block.get("content"), str):
                block_text = block["content"]

            block_type = str(block.get("type", "")).lower()
            if block_type in {"reasoning", "thinking", "reasoning_content"}:
                reasoning_delta += block_text
            elif block_type in {"text", "output_text"}:
                answer_delta += block_text

    if not reasoning_delta and isinstance(delta.get("reasoning"), str):
        reasoning_delta = delta["reasoning"]

    return answer_delta, reasoning_delta


def encode_project_qa_stream_event(event_type: str, **payload: Any) -> bytes:
    """Serialize a project QA stream event as newline-delimited JSON."""
    body = {"type": event_type, **payload}
    return (json.dumps(body, ensure_ascii=False) + "\n").encode("utf-8")


def extract_project_answer_code_terms(answer: str) -> list[str]:
    """Find distinct inline code terms worth turning into clickable references."""
    terms: list[str] = []
    seen: set[str] = set()
    for match in re.findall(r"`([^`]{2,80})`", answer):
        cleaned = match.strip()
        if not cleaned or cleaned.casefold() in seen:
            continue
        seen.add(cleaned.casefold())
        terms.append(cleaned)
    return terms


def extract_project_answer_evidence_labels(answer: str) -> list[str]:
    """Extract evidence labels like [1] [2] from the answer."""
    labels: list[str] = []
    seen: set[str] = set()
    for match in re.findall(r"\[(\d+)\]", answer):
        label = f"[{match}]"
        if label in seen:
            continue
        seen.add(label)
        labels.append(label)
    return labels


def extract_project_answer_symbol_candidates(answer: str, context: dict[str, Any]) -> list[str]:
    """Extract local symbol candidates from both inline-code and plain text mentions."""
    terms = extract_project_answer_code_terms(answer)
    seen: set[str] = {term.casefold() for term in terms}
    knowledge_index = context.get("knowledge_index")
    if knowledge_index is None:
        return terms

    symbol_map = getattr(knowledge_index, "code_symbol_index", {})
    answer_text = answer.casefold()
    for symbol_key, symbols in symbol_map.items():
        symbol = symbols[0].symbol
        if symbol.casefold() in seen:
            continue
        # Keep matching conservative to avoid turning ordinary prose into fake code references.
        if len(symbol) < 4:
            continue
        if symbol.casefold() not in answer_text:
            continue
        seen.add(symbol.casefold())
        terms.append(symbol)
    return terms


def build_project_code_snippet(content: str, term: str, window: int = 260) -> str:
    """Build a compact snippet centered around the referenced term."""
    lowered_content = content.lower()
    lowered_term = term.lower()
    position = lowered_content.find(lowered_term)
    if position < 0:
        compact = re.sub(r"\s+", " ", content).strip()
        return compact[:window].rstrip() + ("..." if len(compact) > window else "")

    start = max(0, position - window // 2)
    end = min(len(content), position + len(term) + window // 2)
    snippet = content[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(content):
        snippet = snippet + "..."
    return snippet


def build_project_excerpt_context(text: str, window: int = 420) -> str:
    """Expand a short excerpt into a slightly richer evidence context block."""
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= window:
        return compact
    return compact[:window].rstrip() + "..."


def build_project_answer_code_references(
    answer: str, context: dict[str, Any]
) -> list[ProjectQACodeReference]:
    """Resolve clickable code/doc references for inline code terms in the answer."""
    terms = extract_project_answer_symbol_candidates(answer, context)
    if not terms:
        return []

    references: list[ProjectQACodeReference] = []
    seen_keys: set[tuple[str, str]] = set()
    knowledge_index = context.get("knowledge_index")

    for term in terms:
        if knowledge_index is not None:
            matched_symbol = knowledge_index.resolve_code_symbol(term)
            if matched_symbol is not None:
                ref_key = (term.casefold(), matched_symbol.path)
                if ref_key in seen_keys:
                    continue
                seen_keys.add(ref_key)
                references.append(
                    ProjectQACodeReference(
                        symbol=term,
                        path=matched_symbol.path,
                        section=matched_symbol.signature,
                        snippet=matched_symbol.snippet,
                    )
                )
                continue

        candidate_chunks = [hit.chunk for hit in context.get("hits", [])]
        if knowledge_index is not None:
            candidate_chunks.extend(knowledge_index.chunks)

        lowered_term = term.lower()
        matched_chunk = None
        for chunk in candidate_chunks:
            haystacks = [chunk.title.lower(), (chunk.section or "").lower(), chunk.content.lower()]
            if any(lowered_term in haystack for haystack in haystacks):
                matched_chunk = chunk
                break

        if matched_chunk is None:
            continue

        ref_key = (term.casefold(), matched_chunk.path)
        if ref_key in seen_keys:
            continue
        seen_keys.add(ref_key)
        references.append(
            ProjectQACodeReference(
                symbol=term,
                path=matched_chunk.path,
                section=matched_chunk.section or None,
                snippet=build_project_code_snippet(matched_chunk.content, term),
            )
        )

    return references[:8]


def build_project_answer_evidence_references(
    answer: str, context: dict[str, Any]
) -> list[ProjectQAEvidenceReference]:
    """Resolve clickable evidence labels like [1] back to retrieval hits."""
    labels = extract_project_answer_evidence_labels(answer)
    hits = context.get("hits", [])
    if not labels or not hits:
        return []

    references: list[ProjectQAEvidenceReference] = []
    for label in labels:
        try:
            index = int(label.strip("[]")) - 1
        except ValueError:
            continue
        if index < 0 or index >= len(hits):
            continue
        hit = hits[index]
        content = getattr(hit.chunk, "content", None) or hit.excerpt
        references.append(
            ProjectQAEvidenceReference(
                label=label,
                path=hit.chunk.path,
                excerpt=hit.excerpt,
                context=build_project_excerpt_context(content),
            )
        )
    return references


# ---------------------------------------------------------------------------
# Server-side session storage (file-backed, lightweight)
# ---------------------------------------------------------------------------
SESSION_DIR = Path(os.getenv("DC_SESSION_DIR", str(PROJECT_ROOT / "sessions")))
SESSION_MAX_TURNS = 20
SESSION_TTL_SECONDS = int(os.getenv("DC_SESSION_TTL_SECONDS", "86400"))
_SESSION_LOCKS: dict[str, Lock] = defaultdict(Lock)
_safe_session_id_re = re.compile(r"^[a-zA-Z0-9_-]{4,64}$")


def _session_path(session_id: str) -> Path:
    return SESSION_DIR / f"{session_id}.jsonl"


def _is_safe_session_id(sid: str) -> bool:
    return bool(_safe_session_id_re.match(sid))


def load_session_history(session_id: str) -> list[dict[str, str]]:
    if not _is_safe_session_id(session_id):
        return []
    path = _session_path(session_id)
    if not path.exists():
        return []
    turns: list[dict[str, str]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if isinstance(entry, dict) and "role" in entry and "content" in entry:
                turns.append({"role": str(entry["role"]), "content": str(entry["content"])})
    except (OSError, json.JSONDecodeError):
        return []
    return turns[-SESSION_MAX_TURNS:]


def cleanup_expired_sessions() -> int:
    """Remove session files older than SESSION_TTL_SECONDS. Returns count removed."""
    if not SESSION_DIR.exists():
        return 0
    now = time.time()
    removed = 0
    for path in SESSION_DIR.glob("*.jsonl"):
        if now - path.stat().st_mtime > SESSION_TTL_SECONDS:
            path.unlink(missing_ok=True)
            removed += 1
    return removed


def list_active_sessions() -> list[dict[str, Any]]:
    """Return metadata for all non-expired session files."""
    if not SESSION_DIR.exists():
        return []
    now = time.time()
    sessions: list[dict[str, Any]] = []
    for path in sorted(SESSION_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
        mtime = path.stat().st_mtime
        if now - mtime > SESSION_TTL_SECONDS:
            continue
        turns = load_session_history(path.stem)
        sessions.append(
            {
                "sessionId": path.stem,
                "turnCount": len(turns),
                "lastActivity": datetime.fromtimestamp(mtime).isoformat(),
            }
        )
    return sessions[:50]


def save_session_turn(session_id: str, role: str, content: str) -> None:
    if not _is_safe_session_id(session_id):
        return
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    path = _session_path(session_id)
    lock = _SESSION_LOCKS[session_id]
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"role": role, "content": content}, ensure_ascii=False) + "\n")


def merge_session_history(
    payload: ProjectQARequest,
) -> tuple[list[dict[str, str]], str | None]:
    """Merge frontend history with server-side session. Return (history, effective_session_id)."""
    sid = payload.sessionId
    if not sid:
        return payload.history, None
    server_history = load_session_history(sid)
    if not server_history:
        return payload.history, sid
    combined = list(server_history) + list(payload.history)
    return combined[-SESSION_MAX_TURNS:], sid


# ---------------------------------------------------------------------------
# Agent retrieval loop: multi-round retrieval with query decomposition
# ---------------------------------------------------------------------------
AGENT_MAX_RETRIEVAL_ROUNDS = int(os.getenv("DC_AGENT_MAX_ROTRIEVAL_ROUNDS", "2"))
AGENT_EVIDENCE_MIN_HITS = int(os.getenv("DC_AGENT_EVIDENCE_MIN_HITS", "3"))
AGENT_EVIDENCE_TOP_SCORE = float(os.getenv("DC_AGENT_EVIDENCE_TOP_SCORE", "0.3"))
AGENT_ENABLE_SELFCHECK = os.getenv("DC_AGENT_ENABLE_SELFCHECK", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}


async def _decompose_question(question: str) -> list[str]:
    """Use LLM to split a complex question into 2-3 focused sub-queries."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return []
    api_base = os.getenv("OPENAI_BASE_URL", "https://api.hotaruapi.top/v1")
    model_name = DEFAULT_CHAT_MODEL.strip()
    if not model_name:
        return []
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            resp = await client.post(
                f"{api_base}/chat/completions",
                headers=build_upstream_chat_headers(api_key),
                json={
                    "model": model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "你是查询分解器。将用户的复杂问题拆成 2-3 个独立子查询，"
                                "每个子查询覆盖原问题的一个方面。每行一个子查询，不要编号，不要解释。"
                                "如果问题本身已经足够简单，直接原样输出即可。"
                            ),
                        },
                        {"role": "user", "content": question[:300]},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 150,
                },
            )
        if resp.status_code != 200:
            return []
        text = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        if not text:
            return []
        lines = [line.strip().lstrip("0123456789.-) ") for line in text.split("\n") if line.strip()]
        return [line for line in lines if len(line) >= 4][:3]
    except Exception:
        return []


def _evidence_sufficient(question: str, hits: list[KnowledgeHit]) -> bool:
    """Check if current retrieval results are sufficient to answer the question."""
    if len(hits) >= AGENT_EVIDENCE_MIN_HITS and hits and hits[0].score >= AGENT_EVIDENCE_TOP_SCORE:
        return True
    if len(hits) >= AGENT_EVIDENCE_MIN_HITS * 2:
        return True
    return False


def _deduplicate_hits(hits: list[KnowledgeHit]) -> list[KnowledgeHit]:
    """Remove duplicate hits (same path + similar content), keeping highest score."""
    seen: dict[str, KnowledgeHit] = {}
    for hit in hits:
        chunk_content = getattr(hit.chunk, "content", "") or ""
        key = f"{hit.chunk.path}:{chunk_content[:80]}"
        if key not in seen or hit.score > seen[key].score:
            seen[key] = hit
    return sorted(seen.values(), key=lambda h: h.score, reverse=True)


async def agent_retrieve_loop(
    question: str,
    knowledge_index: Any,
    top_k: int = 5,
) -> tuple[list[KnowledgeHit], list[ProjectQAToolTrace]]:
    """Multi-round retrieval: original query → check → decompose → re-retrieve."""
    all_traces: list[ProjectQAToolTrace] = []

    # Round 1: Direct retrieval with the original question
    hits = knowledge_index.search(question, top_k=top_k)
    all_traces.append(
        ProjectQAToolTrace(
            tool="repository_search",
            status="used" if hits else "skipped",
            detail=f"命中 {len(hits)} 个仓库证据片段",
        )
    )

    if _evidence_sufficient(question, hits) or AGENT_MAX_RETRIEVAL_ROUNDS < 2:
        return _deduplicate_hits(hits)[:top_k], all_traces

    # Round 2: Decompose and re-retrieve
    sub_queries = await _decompose_question(question)
    if not sub_queries:
        all_traces.append(
            ProjectQAToolTrace(
                tool="query_decomposition",
                status="skipped",
                detail="LLM 分解不可用或返回为空，使用第一轮结果",
            )
        )
        return _deduplicate_hits(hits)[:top_k], all_traces

    all_traces.append(
        ProjectQAToolTrace(
            tool="query_decomposition",
            status="used",
            detail=f"将问题分解为 {len(sub_queries)} 个子查询：{' / '.join(sq[:20] for sq in sub_queries)}",
        )
    )

    sub_top_k = max(3, top_k // 2)
    for sq in sub_queries:
        sub_hits = knowledge_index.search(sq, top_k=sub_top_k)
        hits.extend(sub_hits)

    all_traces.append(
        ProjectQAToolTrace(
            tool="retrieval_round_2",
            status="used",
            detail=f"子查询补充检索后共 {len(hits)} 个片段（去重前）",
        )
    )

    return _deduplicate_hits(hits)[:top_k], all_traces


async def selfcritique_answer(
    question: str, answer: str, evidence_blocks: list[str]
) -> dict[str, Any] | None:
    """LLM-based answer quality check: grounded, factual, complete."""
    if not AGENT_ENABLE_SELFCHECK:
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    api_base = os.getenv("OPENAI_BASE_URL", "https://api.hotaruapi.top/v1")
    model_name = DEFAULT_CHAT_MODEL.strip()
    if not model_name:
        return None
    evidence_text = "\n".join(evidence_blocks[:3])[:1500]
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{api_base}/chat/completions",
                headers=build_upstream_chat_headers(api_key),
                json={
                    "model": model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "你是回答质量审核员。根据给定证据评估回答质量。"
                                '输出 JSON：{"grounded": true/false, "hallucination_risk": "low/medium/high", '
                                '"completeness": "sufficient/partial/insufficient", "issue": "问题描述或null"}。'
                                "只输出 JSON，不要解释。"
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"问题：{question}\n\n"
                                f"回答：{answer[:1000]}\n\n"
                                f"证据：{evidence_text}"
                            ),
                        },
                    ],
                    "temperature": 0.0,
                    "max_tokens": 120,
                },
            )
        if resp.status_code != 200:
            return None
        text = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        if not text:
            return None
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return json.loads(text)
    except (json.JSONDecodeError, Exception):
        return None


async def prepare_project_qa_context(payload: ProjectQARequest, question: str) -> dict[str, Any]:
    """Assemble reusable project QA context shared by normal and streaming endpoints."""
    merged_history, effective_session_id = merge_session_history(payload)
    if effective_session_id:
        save_session_turn(effective_session_id, "user", question)
    agent_mode = await infer_project_agent_mode(question, payload.agentMode)
    answer_frame_title, answer_frame_instruction = select_project_answer_frame(question, agent_mode)
    answer_length_instruction, speaking_style_instruction = resolve_project_answer_style(
        payload.answerLength,
        payload.speakingStyle,
    )
    speaker_profile = resolve_project_speaker_profile(payload.speakerProfile)
    history_summary = summarize_project_agent_history(merged_history)

    knowledge_index = get_project_knowledge_index(force_refresh=payload.forceRefresh)
    hits, retrieval_traces = await agent_retrieve_loop(
        question, knowledge_index, top_k=payload.topK
    )
    evidence_blocks = build_project_agent_evidence_blocks(hits)
    sources = [
        ProjectQASource(path=hit.chunk.path, score=round(hit.score, 4), excerpt=hit.excerpt)
        for hit in hits
    ]
    tool_trace: list[ProjectQAToolTrace] = list(retrieval_traces)

    if history_summary:
        tool_trace.append(
            ProjectQAToolTrace(
                tool="conversation_memory",
                status="used",
                detail="使用最近多轮对话摘要保持上下文连续",
            )
        )

    model_snapshot = None
    if should_use_project_model_info(question, agent_mode):
        model_snapshot = build_project_model_snapshot()
        tool_trace.append(
            ProjectQAToolTrace(
                tool="runtime_model_info",
                status="used" if model_snapshot else "unavailable",
                detail=(
                    "注入当前模型版本、训练样本数和关键指标"
                    if model_snapshot
                    else "当前运行环境缺少模型元信息"
                ),
            )
        )

    live_detection_summary = None
    if payload.analysisText:
        live_detection_summary = run_project_agent_live_detection(payload.analysisText)
        tool_trace.append(
            ProjectQAToolTrace(
                tool="live_detector",
                status="used" if detector is not None else "unavailable",
                detail=(
                    "对附带文本执行了一次实时检测分析"
                    if detector is not None
                    else "后端检测器不可用，未执行实时文本分析"
                ),
            )
        )

    requested_model_preset = resolve_project_qa_model_preset(payload.modelPresetId)
    ordered_model_presets = [requested_model_preset] + [
        preset
        for preset in PROJECT_QA_MODEL_PRESETS
        if preset["id"] != requested_model_preset["id"]
    ]

    answer = build_extractive_answer(question, hits)
    if not sources and live_detection_summary:
        answer = f"当前仓库证据检索较少，但实时检测工具给出的结果是：{live_detection_summary}"
    elif not sources and model_snapshot:
        answer = f"当前仓库证据检索较少，不过运行时模型信息可以先这样回答：\n{model_snapshot}"

    return {
        "agent_mode": agent_mode,
        "answer": answer,
        "answer_frame_instruction": answer_frame_instruction,
        "answer_frame_title": answer_frame_title,
        "answer_length_instruction": answer_length_instruction,
        "evidence_blocks": evidence_blocks,
        "hits": hits,
        "history_summary": history_summary,
        "knowledge_index": knowledge_index,
        "live_detection_summary": live_detection_summary,
        "merged_history": merged_history,
        "effective_session_id": effective_session_id,
        "model_snapshot": model_snapshot,
        "ordered_model_presets": ordered_model_presets,
        "requested_model_preset": requested_model_preset,
        "sources": sources,
        "speaker_profile": speaker_profile,
        "speaking_style_instruction": speaking_style_instruction,
        "suggested_questions": build_project_agent_suggestions(
            agent_mode,
            asked_questions=[
                m.get("content", "") for m in merged_history if m.get("role") == "user"
            ],
        ),
        "tool_trace": tool_trace,
    }


def build_project_qa_response(
    *,
    payload: ProjectQARequest,
    context: dict[str, Any],
    answer: str,
    mode: Literal["extractive", "rag"],
    model_name: str | None,
    model_preset_id: str,
    model_label: str,
    start_time: float,
) -> ProjectQAResponse:
    """Convert resolved QA state into the public response model."""
    processing_time = int((time.time() - start_time) * 1000)
    knowledge_index = context["knowledge_index"]
    code_references = build_project_answer_code_references(answer, context)
    evidence_references = build_project_answer_evidence_references(answer, context)
    return ProjectQAResponse(
        answer=answer,
        mode=mode,
        agentMode=context["agent_mode"],
        answerFrame=context["answer_frame_title"],
        answerLength=payload.answerLength,
        speakingStyle=payload.speakingStyle,
        model=model_name,
        modelPresetId=model_preset_id,
        modelLabel=model_label,
        sourceCount=len(context["sources"]),
        indexSourceCount=knowledge_index.source_count,
        processingTime=processing_time,
        sources=context["sources"],
        codeReferences=code_references,
        evidenceReferences=evidence_references,
        toolTrace=context["tool_trace"],
        suggestedQuestions=context["suggested_questions"],
        memorySummary=context["history_summary"],
        effectiveSpeakerProfile=context["speaker_profile"],
        sessionId=context.get("effective_session_id"),
    )


def resolve_api_key(authorization_header: str | None) -> str | None:
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key

    if not authorization_header:
        return None

    auth = authorization_header.strip()
    if len(auth) > 7 and auth[:7].lower() == "bearer ":
        token = auth[7:].strip()
        return token or None
    # Reject bare "Bearer" with no actual token
    if auth.lower() == "bearer":
        return None
    return auth or None


@app.get(
    "/api/project-qa/model-presets",
    response_model=ProjectQAModelPresetListResponse,
    response_model_exclude_none=True,
)
async def list_project_qa_model_presets(
    http_request: Request,
    x_internal_token: str | None = Header(default=None, alias="X-Internal-Token"),
) -> ProjectQAModelPresetListResponse:
    """List selectable project QA model presets for the frontend."""
    verify_internal_token(x_internal_token)
    enforce_rate_limit(http_request, "chat", CHAT_RATE_LIMIT_PER_WINDOW)

    presets = [
        ProjectQAModelPresetOption(
            id=preset["id"],
            label=preset["label"],
            provider=preset["provider"],
            model=preset["model"],
            isDefault=preset["id"] == PROJECT_QA_DEFAULT_PRESET_ID,
        )
        for preset in PROJECT_QA_MODEL_PRESETS
    ]
    presets.sort(key=lambda preset: (not preset.isDefault, preset.label.casefold()))
    return ProjectQAModelPresetListResponse(
        presets=presets,
        defaultPresetId=PROJECT_QA_DEFAULT_PRESET_ID,
    )


@app.get(
    "/api/project-qa/materials",
    response_model=ProjectQAMaterialListResponse,
    response_model_exclude_none=True,
)
async def list_project_qa_materials(
    http_request: Request,
    x_internal_token: str | None = Header(default=None, alias="X-Internal-Token"),
) -> ProjectQAMaterialListResponse:
    verify_internal_token(x_internal_token)
    enforce_rate_limit(http_request, "chat", CHAT_RATE_LIMIT_PER_WINDOW)

    materials = [build_project_material_record(path) for path in list_uploaded_project_sources()]
    return ProjectQAMaterialListResponse(total=len(materials), materials=materials)


@app.get("/api/project-qa/sessions")
async def list_sessions(
    http_request: Request,
    x_internal_token: str | None = Header(default=None, alias="X-Internal-Token"),
) -> dict[str, Any]:
    verify_internal_token(x_internal_token)
    cleanup_expired_sessions()
    return {"sessions": list_active_sessions()}


@app.delete("/api/project-qa/sessions/{session_id}")
async def delete_session(
    session_id: str,
    http_request: Request,
    x_internal_token: str | None = Header(default=None, alias="X-Internal-Token"),
) -> dict[str, str]:
    verify_internal_token(x_internal_token)
    if not _is_safe_session_id(session_id):
        raise HTTPException(status_code=400, detail="Invalid session ID")
    path = _session_path(session_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    path.unlink()
    _SESSION_LOCKS.pop(session_id, None)
    return {"status": "ok", "deletedSession": session_id}


_FILE_VIEW_ALLOWED_DIRS = {
    "api",
    "scripts",
    "docs",
    "config",
    "configs",
    "frontend/app",
    "frontend/components",
    "frontend/lib",
}
_FILE_VIEW_ALLOWED_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".tsx",
    ".ts",
}
_FILE_VIEW_MAX_LINES = 200


def _resolve_safe_project_file(rel_path: str) -> Path | None:
    """Resolve a relative path to a safe project file, preventing traversal."""
    cleaned = rel_path.strip().lstrip("/\\")
    if not cleaned or ".." in cleaned.split("/"):
        return None
    resolved = (PROJECT_ROOT / cleaned).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError:
        return None
    if not resolved.is_file():
        return None
    if resolved.suffix.lower() not in _FILE_VIEW_ALLOWED_SUFFIXES:
        return None
    top_dir = cleaned.split("/")[0]
    if top_dir not in _FILE_VIEW_ALLOWED_DIRS:
        return None
    return resolved


@app.get("/api/project-qa/file-content")
async def get_project_file_content(
    http_request: Request,
    path: str = "",
    start_line: int = 1,
    end_line: int = 0,
    x_internal_token: str | None = Header(default=None, alias="X-Internal-Token"),
) -> dict[str, Any]:
    verify_internal_token(x_internal_token)
    if not path:
        raise HTTPException(status_code=400, detail="path parameter is required")
    resolved = _resolve_safe_project_file(path)
    if resolved is None:
        raise HTTPException(status_code=404, detail="File not found or not accessible")
    try:
        content = resolved.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Cannot read file: {exc}") from exc
    lines = content.splitlines()
    total_lines = len(lines)
    s = max(1, start_line)
    e = (
        min(total_lines, end_line)
        if end_line > 0
        else min(total_lines, s + _FILE_VIEW_MAX_LINES - 1)
    )
    selected = lines[s - 1 : e]
    numbered = "\n".join(f"{i:>4} | {line}" for i, line in zip(range(s, e + 1), selected))
    return {
        "path": path,
        "totalLines": total_lines,
        "startLine": s,
        "endLine": e,
        "content": numbered,
        "truncated": e < total_lines,
    }


_PROJECT_STRUCTURE = {
    "api/": {
        "description": "FastAPI 后端服务 — 检测 API + OpenAI 兼容接口 + 项目问答",
        "key_files": {
            "api/api.py": "主服务入口：所有端点定义、检测管线、Agent 逻辑",
            "api/CLAUDE.md": "API 模块文档",
        },
    },
    "scripts/training/": {
        "description": "模型训练脚本 — BERT 分类器、边界检测器、基线对比实验",
        "key_files": {
            "scripts/training/train_bert_improved.py": "BERTTrainer 主训练器",
            "scripts/training/train_span_detector.py": "Token 级边界检测器训练",
        },
    },
    "scripts/evaluation/": {
        "description": "评估脚本 — 完整测试集评估、单文本测试、综合对比",
    },
    "scripts/data_cleaning/": {
        "description": "数据清洗 — [SEP] 标记插入、Span 标签生成、训练集构建",
    },
    "scripts/generation/": {
        "description": "AI 文本生成 — 多模型批量生成、混合文本构造",
    },
    "datasets/": {
        "description": "数据集 — 训练/验证/测试集、评估集、反馈闭环数据",
        "key_files": {"datasets/registry.json": "数据集注册表（18 条元数据记录）"},
    },
    "models/": {
        "description": "训练好的模型权重 — bert_v11c_boundary_fix (分类器) + bert_span_detector (边界检测)",
    },
    "docs/": {
        "description": "项目文档 — 答辩口径、实验日志、计划、论文草稿",
        "key_files": {
            "docs/project/DEFENSE_CURRENT_STATUS.md": "答辩口径快照（最新）",
            "docs/project/ADVISOR_ACADEMIC_QA.md": "60 条答辩问答底稿",
        },
    },
    "frontend/": {
        "description": "Next.js 前端 — 检测界面 + 项目问答 Advisor 页面",
    },
}


@app.get("/api/project-qa/project-structure")
async def get_project_structure(
    http_request: Request,
    x_internal_token: str | None = Header(default=None, alias="X-Internal-Token"),
) -> dict[str, Any]:
    verify_internal_token(x_internal_token)
    return {"structure": _PROJECT_STRUCTURE, "root": str(PROJECT_ROOT)}


@app.post(
    "/api/project-qa/materials",
    response_model=ProjectQAMaterialUploadResponse,
    response_model_exclude_none=True,
)
async def upload_project_qa_materials(
    http_request: Request,
    files: list[Any] | None = None,
    x_internal_token: str | None = Header(default=None, alias="X-Internal-Token"),
) -> ProjectQAMaterialUploadResponse:
    verify_internal_token(x_internal_token)
    enforce_rate_limit(http_request, "chat", CHAT_RATE_LIMIT_PER_WINDOW)

    effective_files = files
    if effective_files is None:
        try:
            form = await http_request.form()
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail="Failed to parse uploaded form data"
            ) from exc
        effective_files = [
            value for value in form.getlist("files") if isinstance(value, UploadFile)
        ]

    if not effective_files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    upload_dir = PATHS.ensure_dir(PATHS.project_qa_uploads_dir)
    uploaded: list[ProjectQAMaterial] = []
    skipped: list[str] = []

    for upload in effective_files:
        original_name = (upload.filename or "").strip()
        if not original_name:
            skipped.append("unnamed file")
            continue

        sanitized_name = sanitize_project_material_name(original_name)
        suffix = Path(sanitized_name).suffix.lower()
        if suffix not in PROJECT_QA_ALLOWED_SUFFIXES:
            skipped.append(f"{original_name}: unsupported file type")
            continue

        payload = await upload.read()
        if len(payload) > PROJECT_QA_MAX_UPLOAD_BYTES:
            skipped.append(f"{original_name}: file too large")
            continue

        stem = Path(sanitized_name).stem[:80]
        digest = hashlib.sha256(payload).hexdigest()[:10]
        target_path = upload_dir / f"{stem}_{digest}{suffix}"
        target_path.write_bytes(payload)
        uploaded.append(build_project_material_record(target_path))

    if uploaded:
        get_project_knowledge_index(force_refresh=True)

    return ProjectQAMaterialUploadResponse(status="ok", uploaded=uploaded, skipped=skipped)


@app.post(
    "/api/project-qa",
    response_model=ProjectQAResponse,
    response_model_exclude_none=True,
)
async def project_qa(
    payload: ProjectQARequest,
    http_request: Request,
    authorization: str | None = Header(default=None),
    x_internal_token: str | None = Header(default=None, alias="X-Internal-Token"),
) -> ProjectQAResponse:
    verify_internal_token(x_internal_token)
    enforce_rate_limit(http_request, "chat", CHAT_RATE_LIMIT_PER_WINDOW)

    start_time = time.time()
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    if not payload.forceRefresh:
        cached = _qa_cache_get(payload)
        if cached is not None:
            return ProjectQAResponse(**cached)

    context = await prepare_project_qa_context(payload, question)
    answer = context["answer"]
    mode: Literal["extractive", "rag"] = "extractive"
    model_name: str | None = None
    requested_model_preset = context["requested_model_preset"]
    model_preset_id = requested_model_preset["id"]
    model_label = requested_model_preset["label"]
    if payload.useLLM and (
        context["sources"] or context["model_snapshot"] or context["live_detection_summary"]
    ):
        for preset_index, model_preset in enumerate(context["ordered_model_presets"]):
            candidate_api_key = model_preset.get("api_key") or resolve_api_key(authorization)
            candidate_api_base = model_preset.get("api_base") or os.getenv(
                "OPENAI_BASE_URL",
                "https://api.hotaruapi.top/v1",
            )
            candidate_model_name = model_preset.get("model") or DEFAULT_CHAT_MODEL.strip() or None
            candidate_label = model_preset["label"]

            if not candidate_api_key or not candidate_model_name:
                context["tool_trace"].append(
                    ProjectQAToolTrace(
                        tool="llm_synthesis",
                        status="skipped",
                        detail=f"预设模型 {candidate_label} 缺少可用配置，已跳过",
                    )
                )
                continue

            try:
                async with httpx.AsyncClient(timeout=UPSTREAM_CHAT_TIMEOUT_SECONDS) as client:
                    response = await client.post(
                        f"{candidate_api_base}/chat/completions",
                        headers=build_upstream_chat_headers(candidate_api_key),
                        json={
                            "model": candidate_model_name,
                            "messages": build_project_qa_messages(
                                question=question,
                                evidence_blocks=context["evidence_blocks"],
                                agent_mode=context["agent_mode"],
                                answer_frame_title=context["answer_frame_title"],
                                answer_frame_instruction=context["answer_frame_instruction"],
                                answer_length_instruction=context["answer_length_instruction"],
                                speaking_style_instruction=context["speaking_style_instruction"],
                                speaker_profile=context["speaker_profile"],
                                history_summary=context["history_summary"],
                                model_snapshot=context["model_snapshot"],
                                live_detection_summary=context["live_detection_summary"],
                                tool_trace=context["tool_trace"],
                            ),
                            "temperature": 0.2,
                            "max_tokens": min(payload.topK * 180, CHAT_MAX_TOKENS),
                        },
                    )

                if response.status_code != 200:
                    logger.warning(
                        "[project_qa] upstream preset %s returned status %s",
                        model_preset["id"],
                        response.status_code,
                    )
                    context["tool_trace"].append(
                        ProjectQAToolTrace(
                            tool="llm_synthesis",
                            status="unavailable",
                            detail=(
                                f"上游模型 {candidate_label} 接口返回 {response.status_code}"
                                + (
                                    "，已尝试切换下一个预设"
                                    if preset_index < len(context["ordered_model_presets"]) - 1
                                    else "，已回退本地抽取回答"
                                )
                            ),
                        )
                    )
                    continue

                body = response.json()
                message = body.get("choices", [{}])[0].get("message", {})
                llm_answer = extract_openai_message_field(message, "content").strip()
                reasoning_only = extract_openai_message_field(message, "reasoning_content").strip()
                if not llm_answer:
                    detail = (
                        f"上游模型 {candidate_label} 仅返回思考过程、未返回正文"
                        if reasoning_only
                        else f"上游模型 {candidate_label} 返回为空"
                    )
                    context["tool_trace"].append(
                        ProjectQAToolTrace(
                            tool="llm_synthesis",
                            status="unavailable",
                            detail=(
                                detail
                                + (
                                    "，已尝试切换下一个预设"
                                    if preset_index < len(context["ordered_model_presets"]) - 1
                                    else "，已回退本地抽取回答"
                                )
                            ),
                        )
                    )
                    continue

                answer = llm_answer
                mode = "rag"
                model_name = body.get("model", candidate_model_name)
                model_preset_id = model_preset["id"]
                model_label = candidate_label
                context["tool_trace"].append(
                    ProjectQAToolTrace(
                        tool="llm_synthesis",
                        status="used",
                        detail=(
                            f"使用上游模型 {candidate_label} / {model_name or DEFAULT_CHAT_MODEL} 综合证据生成答辩回答"
                            if preset_index == 0
                            else f"首选模型不可用，已切换到 {candidate_label} / {model_name or DEFAULT_CHAT_MODEL} 继续生成答辩回答"
                        ),
                    )
                )
                break
            except Exception as exc:
                logger.warning(
                    "[project_qa] preset %s failed, trying next: %s",
                    model_preset["id"],
                    exc,
                )
                context["tool_trace"].append(
                    ProjectQAToolTrace(
                        tool="llm_synthesis",
                        status="unavailable",
                        detail=(
                            f"上游模型 {candidate_label} 调用失败：{exc}"
                            + (
                                "，已尝试切换下一个预设"
                                if preset_index < len(context["ordered_model_presets"]) - 1
                                else "，已回退本地抽取回答"
                            )
                        ),
                    )
                )
    elif payload.useLLM:
        context["tool_trace"].append(
            ProjectQAToolTrace(
                tool="llm_synthesis",
                status="skipped",
                detail=("当前没有足够的证据上下文可交给上游模型综合"),
            )
        )

    critique = await selfcritique_answer(question, answer, context["evidence_blocks"])
    if critique:
        context["tool_trace"].append(
            ProjectQAToolTrace(
                tool="answer_selfcritique",
                status="used",
                detail=(
                    f"回答自检：grounded={critique.get('grounded')}, "
                    f"hallucination_risk={critique.get('hallucination_risk')}, "
                    f"completeness={critique.get('completeness')}"
                ),
            )
        )

    effective_sid = context.get("effective_session_id")
    if effective_sid and answer:
        save_session_turn(effective_sid, "assistant", answer)

    response_obj = build_project_qa_response(
        payload=payload,
        context=context,
        answer=answer,
        mode=mode,
        model_name=model_name,
        model_preset_id=model_preset_id,
        model_label=model_label,
        start_time=start_time,
    )
    _qa_cache_put(payload, response_obj.model_dump(exclude_none=True))
    return response_obj


@app.post("/api/project-qa/stream")
async def project_qa_stream(
    payload: ProjectQARequest,
    http_request: Request,
    authorization: str | None = Header(default=None),
    x_internal_token: str | None = Header(default=None, alias="X-Internal-Token"),
) -> StreamingResponse:
    """Stream project QA progress and answer deltas as newline-delimited JSON."""
    verify_internal_token(x_internal_token)
    enforce_rate_limit(http_request, "chat", CHAT_RATE_LIMIT_PER_WINDOW)

    start_time = time.time()
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    async def event_stream():
        try:
            if not payload.forceRefresh:
                cached = _qa_cache_get(payload)
                if cached is not None:
                    yield encode_project_qa_stream_event("status", message="命中缓存，直接返回")
                    yield encode_project_qa_stream_event(
                        "answer_delta", delta=cached.get("answer", "")
                    )
                    yield encode_project_qa_stream_event("final", response=cached)
                    return

            yield encode_project_qa_stream_event("status", message="正在检索仓库资料")
            context = await prepare_project_qa_context(payload, question)
            for trace in context["tool_trace"]:
                yield encode_project_qa_stream_event("trace", trace=trace.model_dump())

            answer = context["answer"]
            mode: Literal["extractive", "rag"] = "extractive"
            model_name: str | None = None
            requested_model_preset = context["requested_model_preset"]
            model_preset_id = requested_model_preset["id"]
            model_label = requested_model_preset["label"]
            answer_chunks: list[str] = []

            if payload.useLLM and (
                context["sources"] or context["model_snapshot"] or context["live_detection_summary"]
            ):
                for preset_index, model_preset in enumerate(context["ordered_model_presets"]):
                    candidate_api_key = model_preset.get("api_key") or resolve_api_key(
                        authorization
                    )
                    candidate_api_base = model_preset.get("api_base") or os.getenv(
                        "OPENAI_BASE_URL",
                        "https://api.hotaruapi.top/v1",
                    )
                    candidate_model_name = (
                        model_preset.get("model") or DEFAULT_CHAT_MODEL.strip() or None
                    )
                    candidate_label = model_preset["label"]
                    candidate_emitted_deltas = False
                    answer_chunks = []

                    if not candidate_api_key or not candidate_model_name:
                        trace = ProjectQAToolTrace(
                            tool="llm_synthesis",
                            status="skipped",
                            detail=f"预设模型 {candidate_label} 缺少可用配置，已跳过",
                        )
                        context["tool_trace"].append(trace)
                        yield encode_project_qa_stream_event("trace", trace=trace.model_dump())
                        continue

                    yield encode_project_qa_stream_event(
                        "status",
                        message=f"正在调用 {candidate_label} 生成回答",
                    )

                    try:
                        async with httpx.AsyncClient(
                            timeout=UPSTREAM_CHAT_TIMEOUT_SECONDS
                        ) as client:
                            async with client.stream(
                                "POST",
                                f"{candidate_api_base}/chat/completions",
                                headers=build_upstream_chat_headers(candidate_api_key),
                                json={
                                    "model": candidate_model_name,
                                    "messages": build_project_qa_messages(
                                        question=question,
                                        evidence_blocks=context["evidence_blocks"],
                                        agent_mode=context["agent_mode"],
                                        answer_frame_title=context["answer_frame_title"],
                                        answer_frame_instruction=context[
                                            "answer_frame_instruction"
                                        ],
                                        answer_length_instruction=context[
                                            "answer_length_instruction"
                                        ],
                                        speaking_style_instruction=context[
                                            "speaking_style_instruction"
                                        ],
                                        speaker_profile=context["speaker_profile"],
                                        history_summary=context["history_summary"],
                                        model_snapshot=context["model_snapshot"],
                                        live_detection_summary=context["live_detection_summary"],
                                        tool_trace=context["tool_trace"],
                                    ),
                                    "temperature": 0.2,
                                    "max_tokens": min(payload.topK * 180, CHAT_MAX_TOKENS),
                                    "stream": True,
                                },
                            ) as response:
                                if response.status_code != 200:
                                    logger.warning(
                                        "[project_qa_stream] upstream preset %s returned status %s",
                                        model_preset["id"],
                                        response.status_code,
                                    )
                                    trace = ProjectQAToolTrace(
                                        tool="llm_synthesis",
                                        status="unavailable",
                                        detail=(
                                            f"上游模型 {candidate_label} 接口返回 {response.status_code}"
                                            + (
                                                "，已尝试切换下一个预设"
                                                if preset_index
                                                < len(context["ordered_model_presets"]) - 1
                                                else "，已回退本地抽取回答"
                                            )
                                        ),
                                    )
                                    context["tool_trace"].append(trace)
                                    yield encode_project_qa_stream_event(
                                        "trace", trace=trace.model_dump()
                                    )
                                    continue

                                content_type = (response.headers.get("content-type") or "").lower()
                                if "application/json" in content_type:
                                    body = json.loads((await response.aread()).decode("utf-8"))
                                    reasoning_text = extract_openai_message_field(
                                        body.get("choices", [{}])[0].get("message", {}),
                                        "reasoning_content",
                                    ).strip()
                                    llm_answer = extract_openai_message_field(
                                        body.get("choices", [{}])[0].get("message", {}),
                                        "content",
                                    ).strip()
                                    if reasoning_text:
                                        candidate_emitted_deltas = True
                                        yield encode_project_qa_stream_event(
                                            "thinking_delta",
                                            delta=reasoning_text,
                                        )
                                    if llm_answer:
                                        candidate_emitted_deltas = True
                                        answer_chunks.append(llm_answer)
                                        yield encode_project_qa_stream_event(
                                            "answer_delta",
                                            delta=llm_answer,
                                        )
                                else:
                                    async for raw_line in response.aiter_lines():
                                        line = raw_line.strip()
                                        if not line:
                                            continue
                                        if line.startswith("data:"):
                                            line = line[5:].strip()
                                        if not line or line == "[DONE]":
                                            continue

                                        try:
                                            chunk = json.loads(line)
                                        except json.JSONDecodeError:
                                            continue

                                        choices = chunk.get("choices") or []
                                        if not choices:
                                            continue
                                        delta = choices[0].get("delta") or {}
                                        answer_delta, reasoning_delta = extract_openai_delta_text(
                                            delta
                                        )
                                        if reasoning_delta:
                                            candidate_emitted_deltas = True
                                            yield encode_project_qa_stream_event(
                                                "thinking_delta",
                                                delta=reasoning_delta,
                                            )
                                        if answer_delta:
                                            candidate_emitted_deltas = True
                                            answer_chunks.append(answer_delta)
                                            yield encode_project_qa_stream_event(
                                                "answer_delta",
                                                delta=answer_delta,
                                            )

                        llm_answer = "".join(answer_chunks).strip()
                        if not llm_answer:
                            detail = (
                                f"上游模型 {candidate_label} 仅返回思考过程、未返回正文"
                                if candidate_emitted_deltas
                                else f"上游模型 {candidate_label} 返回为空"
                            )
                            trace = ProjectQAToolTrace(
                                tool="llm_synthesis",
                                status="unavailable",
                                detail=(
                                    detail
                                    + (
                                        "，已尝试切换下一个预设"
                                        if preset_index < len(context["ordered_model_presets"]) - 1
                                        else "，已回退本地抽取回答"
                                    )
                                ),
                            )
                            context["tool_trace"].append(trace)
                            yield encode_project_qa_stream_event("trace", trace=trace.model_dump())
                            if candidate_emitted_deltas:
                                yield encode_project_qa_stream_event("reset_deltas")
                            continue

                        answer = llm_answer
                        mode = "rag"
                        model_name = candidate_model_name
                        model_preset_id = model_preset["id"]
                        model_label = candidate_label
                        trace = ProjectQAToolTrace(
                            tool="llm_synthesis",
                            status="used",
                            detail=(
                                f"使用上游模型 {candidate_label} / {model_name or DEFAULT_CHAT_MODEL} 综合证据生成答辩回答"
                                if preset_index == 0
                                else f"首选模型不可用，已切换到 {candidate_label} / {model_name or DEFAULT_CHAT_MODEL} 继续生成答辩回答"
                            ),
                        )
                        context["tool_trace"].append(trace)
                        yield encode_project_qa_stream_event("trace", trace=trace.model_dump())
                        break
                    except Exception as exc:
                        if candidate_emitted_deltas:
                            yield encode_project_qa_stream_event("reset_deltas")
                        logger.warning(
                            "[project_qa_stream] preset %s failed, trying next: %s",
                            model_preset["id"],
                            exc,
                        )
                        trace = ProjectQAToolTrace(
                            tool="llm_synthesis",
                            status="unavailable",
                            detail=(
                                f"上游模型 {candidate_label} 调用失败：{exc}"
                                + (
                                    "，已尝试切换下一个预设"
                                    if preset_index < len(context["ordered_model_presets"]) - 1
                                    else "，已回退本地抽取回答"
                                )
                            ),
                        )
                        context["tool_trace"].append(trace)
                        yield encode_project_qa_stream_event("trace", trace=trace.model_dump())
            elif payload.useLLM:
                trace = ProjectQAToolTrace(
                    tool="llm_synthesis",
                    status="skipped",
                    detail="当前没有足够的证据上下文可交给上游模型综合",
                )
                context["tool_trace"].append(trace)
                yield encode_project_qa_stream_event("trace", trace=trace.model_dump())

            if mode == "extractive":
                yield encode_project_qa_stream_event("status", message="正在整理仓库证据回答")
                yield encode_project_qa_stream_event("answer_delta", delta=answer)

            response_payload = build_project_qa_response(
                payload=payload,
                context=context,
                answer=answer,
                mode=mode,
                model_name=model_name,
                model_preset_id=model_preset_id,
                model_label=model_label,
                start_time=start_time,
            )
            critique = await selfcritique_answer(question, answer, context["evidence_blocks"])
            if critique:
                context["tool_trace"].append(
                    ProjectQAToolTrace(
                        tool="answer_selfcritique",
                        status="used",
                        detail=(
                            f"回答自检：grounded={critique.get('grounded')}, "
                            f"hallucination_risk={critique.get('hallucination_risk')}, "
                            f"completeness={critique.get('completeness')}"
                        ),
                    )
                )
                yield encode_project_qa_stream_event(
                    "trace", trace=context["tool_trace"][-1].model_dump()
                )
            effective_sid = context.get("effective_session_id")
            if effective_sid and answer:
                save_session_turn(effective_sid, "assistant", answer)
            response_data = response_payload.model_dump(exclude_none=True)
            _qa_cache_put(payload, response_data)
            yield encode_project_qa_stream_event(
                "final",
                response=response_data,
            )
        except HTTPException as exc:
            yield encode_project_qa_stream_event("error", message=exc.detail)
        except Exception as exc:
            logger.exception("[project_qa_stream] unexpected error")
            yield encode_project_qa_stream_event("error", message=str(exc) or "问答流失败")

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store"},
    )


@app.post("/v1/chat/completions")
@app.post("/api/chat/completions")
async def chat_completions(
    payload: ChatRequest,
    http_request: Request,
    authorization: str | None = Header(default=None),
    x_internal_token: str | None = Header(default=None, alias="X-Internal-Token"),
) -> dict[str, Any]:
    verify_internal_token(x_internal_token)
    enforce_rate_limit(http_request, "chat", CHAT_RATE_LIMIT_PER_WINDOW)

    api_key = resolve_api_key(authorization)
    api_base = os.getenv("OPENAI_BASE_URL", "https://api.hotaruapi.top/v1")
    model_name = (payload.model or DEFAULT_CHAT_MODEL).strip()

    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set")
    if not model_name:
        raise HTTPException(status_code=400, detail="model must not be empty")
    if len(model_name) > 128:
        raise HTTPException(status_code=400, detail="model name is too long")

    try:
        async with httpx.AsyncClient(timeout=UPSTREAM_CHAT_TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{api_base}/chat/completions",
                headers=build_upstream_chat_headers(api_key),
                json={
                    "model": model_name,
                    "messages": payload.messages,
                    "temperature": payload.temperature,
                    "max_tokens": payload.max_tokens,
                },
            )

        if response.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=f"Upstream provider returned status {response.status_code}",
            )

        return response.json()
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="Upstream provider timeout") from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[chat_completions] unexpected upstream error: %s", exc)
        raise HTTPException(status_code=500, detail="Chat completion failed") from exc


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
