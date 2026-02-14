import os
import re
import secrets
import time
from collections import defaultdict, deque
from threading import Lock
from typing import Any

import requests
import torch
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import BertForSequenceClassification, BertForTokenClassification, BertTokenizer

CLASSIFIER_MODEL_PATH = os.getenv("DETECTOR_CLASSIFIER_MODEL", "models/bert_v11c_boundary_fix")
SPAN_MODEL_PATH = os.getenv("DETECTOR_SPAN_MODEL", "models/bert_span_detector")
CLASSIFIER_MAX_LENGTH = int(os.getenv("DETECTOR_MAX_LENGTH", "256"))
CLASSIFIER_TEMPERATURE = float(os.getenv("DETECTOR_TEMPERATURE", "0.8165"))
DECISION_THRESHOLD = float(os.getenv("DETECTOR_DECISION_THRESHOLD", "0.8"))
MODEL_VERSION = os.path.basename(CLASSIFIER_MODEL_PATH.rstrip("/\\"))
DEFAULT_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "deepseek-ai/deepseek-v3.1")

MAX_DETECT_TEXT_CHARS = int(os.getenv("DETECTOR_MAX_TEXT_CHARS", "10000"))
CHAT_MAX_MESSAGES = int(os.getenv("OPENAI_CHAT_MAX_MESSAGES", "50"))
CHAT_MAX_TOKENS = int(os.getenv("OPENAI_CHAT_MAX_TOKENS", "2048"))
UPSTREAM_CHAT_TIMEOUT_SECONDS = int(os.getenv("OPENAI_TIMEOUT_SECONDS", "60"))

RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
DETECT_RATE_LIMIT_PER_WINDOW = int(os.getenv("DETECT_RATE_LIMIT_PER_WINDOW", "60"))
CHAT_RATE_LIMIT_PER_WINDOW = int(os.getenv("CHAT_RATE_LIMIT_PER_WINDOW", "20"))

ENFORCE_INTERNAL_TOKEN = (
    os.getenv("ENFORCE_INTERNAL_TOKEN", "1").strip().lower() in {"1", "true", "yes", "on"}
)
INTERNAL_API_TOKEN = os.getenv("INTERNAL_API_TOKEN", "").strip()

INCLUDE_RISK_OBSERVABILITY = (
    os.getenv("DETECTOR_INCLUDE_RISK_OBSERVABILITY", "0").strip().lower()
    in {"1", "true", "yes", "on"}
)

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


def get_client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
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
        print(f"Loading models on {self.device}...")
        self.classifier_max_length = CLASSIFIER_MAX_LENGTH
        self.classifier_temperature = max(CLASSIFIER_TEMPERATURE, 1e-6)

        try:
            self.classifier_tokenizer = BertTokenizer.from_pretrained(CLASSIFIER_MODEL_PATH)
            self.classifier = BertForSequenceClassification.from_pretrained(
                CLASSIFIER_MODEL_PATH
            ).to(self.device)
            self.classifier.eval()
            print(
                f"Classifier loaded ({CLASSIFIER_MODEL_PATH}, "
                f"max_length={self.classifier_max_length}, "
                f"temperature={self.classifier_temperature})."
            )
        except Exception as exc:
            print(f"Error loading classifier: {exc}")
            raise

        try:
            self.span_tokenizer = BertTokenizer.from_pretrained(SPAN_MODEL_PATH)
            self.span_detector = BertForTokenClassification.from_pretrained(SPAN_MODEL_PATH).to(
                self.device
            )
            self.span_detector.eval()
            print(f"Span detector loaded ({SPAN_MODEL_PATH}).")
        except Exception as exc:
            print(f"Error loading span detector: {exc}")
            raise

    def classify(self, text: str) -> dict[str, float | str]:
        encoding = self.classifier_tokenizer(
            text,
            max_length=self.classifier_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
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

    def detect_boundary(self, text: str) -> dict[str, int | None | str]:
        text_clean = text.replace("[SEP]", "")
        encoding = self.span_tokenizer(
            text_clean,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            outputs = self.span_detector(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits[0], dim=-1).cpu()

        tokens = self.span_tokenizer.convert_ids_to_tokens(input_ids[0])
        labels = preds.numpy()

        boundary_idx = None
        for i in range(1, len(labels)):
            if labels[i - 1] == 0 and labels[i] == 1:
                boundary_idx = i
                break

        char_pos = 0
        boundary_char = None
        for i, token in enumerate(tokens):
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue
            token_text = token.replace("##", "")
            if i == boundary_idx:
                boundary_char = char_pos
                break
            char_pos += len(token_text)

        return {
            "boundary_token": boundary_idx,
            "boundary_char": boundary_char,
            "text": text_clean,
        }


app = FastAPI(title="AI Text Detection API")

CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "https://baxfor.fun,http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
]
CORS_ALLOW_CREDENTIALS = (
    os.getenv("CORS_ALLOW_CREDENTIALS", "0").strip().lower() in {"1", "true", "yes", "on"}
)
if "*" in CORS_ORIGINS:
    CORS_ALLOW_CREDENTIALS = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Internal-Token"],
)

detector: HybridTextDetector | None = None


@app.on_event("startup")
async def startup_event() -> None:
    global detector
    detector = HybridTextDetector()


@app.get("/api/health")
async def health_check() -> dict[str, Any]:
    return {
        "status": "ok",
        "detectorReady": detector is not None,
        "modelVersion": MODEL_VERSION,
        "decisionThreshold": DECISION_THRESHOLD,
        "maxLength": CLASSIFIER_MAX_LENGTH,
        "authEnabled": ENFORCE_INTERNAL_TOKEN,
    }


class DetectRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_DETECT_TEXT_CHARS)


class SentenceResult(BaseModel):
    text: str
    isAI: bool
    confidence: float


class DetectionResponse(BaseModel):
    type: str
    confidence: float
    humanPercentage: int
    aiPercentage: int
    boundary: int | None = None
    sentences: list[SentenceResult]
    processingTime: int
    modelVersion: str | None = None
    decisionThreshold: float | None = None
    riskFlags: list[str] | None = None
    domainHint: str | None = None


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
    label = cls_result["label"]
    confidence = float(cls_result["confidence"]) * 100
    prob_ai = float(cls_result["prob_ai"])
    prob_human = float(cls_result["prob_human"])

    ai_percentage = int(prob_ai * 100)
    human_percentage = int(prob_human * 100)

    result_type = "mixed"
    boundary_char = None

    threshold_percent = int(DECISION_THRESHOLD * 100)
    if ai_percentage > threshold_percent:
        result_type = "ai"
    elif human_percentage > threshold_percent:
        result_type = "human"

    if result_type in ["ai", "mixed"] or label == "AI":
        boundary_res = detector.detect_boundary(text)
        if boundary_res["boundary_char"] is not None:
            boundary_char = int(boundary_res["boundary_char"])
            result_type = "mixed"
        elif result_type == "mixed":
            result_type = "ai" if prob_ai > prob_human else "human"

    final_sentences = split_sentences(text)

    boundary_sentence_index = None
    running_char_count = 0
    for idx, sentence in enumerate(final_sentences):
        sent_len = len(sentence)
        if boundary_char is not None and running_char_count <= boundary_char < running_char_count + sent_len:
            boundary_sentence_index = idx
        running_char_count += sent_len

    sentence_results: list[SentenceResult] = []
    for idx, sentence in enumerate(final_sentences):
        if result_type == "ai":
            is_ai = True
        elif result_type == "human":
            is_ai = False
        else:
            if boundary_sentence_index is not None:
                is_ai = idx >= boundary_sentence_index
            else:
                is_ai = ai_percentage > 50

        sentence_results.append(
            SentenceResult(
                text=sentence,
                isAI=is_ai,
                confidence=confidence,
            )
        )

    processing_time = int((time.time() - start_time) * 1000)

    model_version: str | None = None
    decision_threshold: float | None = None
    risk_flags: list[str] | None = None
    domain_hint: str | None = None

    if INCLUDE_RISK_OBSERVABILITY:
        model_version = MODEL_VERSION
        decision_threshold = DECISION_THRESHOLD
        domain_hint = infer_domain_hint(text)
        risk_flags = collect_risk_flags(
            text=text,
            confidence=confidence,
            boundary_sentence_index=boundary_sentence_index,
            result_type=result_type,
        )

    return DetectionResponse(
        type=result_type,
        confidence=confidence,
        humanPercentage=human_percentage,
        aiPercentage=ai_percentage,
        boundary=boundary_sentence_index,
        sentences=sentence_results,
        processingTime=processing_time,
        modelVersion=model_version,
        decisionThreshold=decision_threshold,
        riskFlags=risk_flags,
        domainHint=domain_hint,
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
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip()
        return token or None
    return auth or None


@app.post("/v1/chat/completions")
def chat_completions(
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
        response = requests.post(
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
            timeout=UPSTREAM_CHAT_TIMEOUT_SECONDS,
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=f"Upstream provider returned status {response.status_code}",
            )

        return response.json()
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Upstream provider timeout")
    except requests.exceptions.JSONDecodeError:
        raise HTTPException(status_code=502, detail="Invalid JSON response from upstream provider")
    except HTTPException:
        raise
    except Exception as exc:
        print(f"[chat_completions] unexpected upstream error: {exc}")
        raise HTTPException(status_code=500, detail="Chat completion failed")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
