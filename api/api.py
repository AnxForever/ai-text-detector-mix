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
from typing import Any, Literal

import httpx
import torch
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import BertForSequenceClassification, BertForTokenClassification, BertTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

CLASSIFIER_MODEL_PATH = os.getenv("DETECTOR_CLASSIFIER_MODEL", "models/bert_v11c_boundary_fix")
SPAN_MODEL_PATH = os.getenv("DETECTOR_SPAN_MODEL", "models/bert_span_detector")
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
DEFAULT_FEEDBACK_DIR = Path(os.getenv("DETECTOR_FEEDBACK_DIR", "/app/datasets/feedback_loop"))

try:
    from scripts.utils.feedback_loop import persist_feedback
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
        confirmations_path = stored_dir / "confirmations.jsonl"
        corrections_path = stored_dir / "misclassified_samples.jsonl"
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
        }

        _append_jsonl(confirmations_path, record)

        if not confirmed_correct:
            correction_record = {
                **record,
                "dataset_type": "manual_correction",
            }
            _append_jsonl(corrections_path, correction_record)

        return {
            "feedback_id": feedback_id,
            "created_at": timestamp,
            "stored_dir": str(stored_dir),
            "confirmations_path": str(confirmations_path),
            "corrections_path": str(corrections_path),
            "misclassified_saved": not confirmed_correct,
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


class HybridTextDetector:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading models on %s ...", self.device)
        self.classifier_max_length = CLASSIFIER_MAX_LENGTH
        self.classifier_temperature = max(CLASSIFIER_TEMPERATURE, 1e-6)

        try:
            self.classifier_tokenizer = BertTokenizer.from_pretrained(CLASSIFIER_MODEL_PATH)
            self.classifier = BertForSequenceClassification.from_pretrained(
                CLASSIFIER_MODEL_PATH
            ).to(self.device)
            self.classifier.eval()
            logger.info(
                "Classifier loaded (%s, max_length=%d, temperature=%.4f).",
                CLASSIFIER_MODEL_PATH,
                self.classifier_max_length,
                self.classifier_temperature,
            )
        except Exception as exc:
            logger.error("Error loading classifier: %s", exc)
            raise

        try:
            self.span_tokenizer = BertTokenizer.from_pretrained(SPAN_MODEL_PATH)
            self.span_detector = BertForTokenClassification.from_pretrained(SPAN_MODEL_PATH).to(
                self.device
            )
            self.span_detector.eval()
            logger.info("Span detector loaded (%s).", SPAN_MODEL_PATH)
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
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
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
