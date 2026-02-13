import os
import re
import time

import requests
import torch
import uvicorn
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import BertForSequenceClassification, BertForTokenClassification, BertTokenizer

CLASSIFIER_MODEL_PATH = os.getenv("DETECTOR_CLASSIFIER_MODEL", "models/bert_v11c_boundary_fix")
SPAN_MODEL_PATH = os.getenv("DETECTOR_SPAN_MODEL", "models/bert_span_detector")
CLASSIFIER_MAX_LENGTH = int(os.getenv("DETECTOR_MAX_LENGTH", "256"))
CLASSIFIER_TEMPERATURE = float(os.getenv("DETECTOR_TEMPERATURE", "0.8165"))
DECISION_THRESHOLD = float(os.getenv("DETECTOR_DECISION_THRESHOLD", "0.8"))
DEFAULT_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "deepseek-ai/deepseek-v3.1")
MODEL_VERSION = os.path.basename(CLASSIFIER_MODEL_PATH.rstrip("/\\"))
INCLUDE_RISK_OBSERVABILITY = (
    os.getenv("DETECTOR_INCLUDE_RISK_OBSERVABILITY", "0").strip().lower()
    in {"1", "true", "yes", "on"}
)

SENTENCE_SPLIT_PATTERN = re.compile(r'([。！？.!?])')
FORMAL_PATTERN = re.compile(r"(通知|公告|特此|敬请|请各位|温馨提示|须知)")
TECH_PATTERN = re.compile(r"(算法|模型|神经网络|数据库|API|代码|训练|部署|实验|推理|调参)")
CASUAL_PATTERN = re.compile(r"(哈哈|hh|呢|吧|啊|呀|呗|我觉得|说实话|有点)")
TEMPLATE_LIKE_PATTERN = re.compile(
    r"(分析请求|逐句分析|改进思路|好的，用户|用户希望|As an AI|as an ai)",
    re.IGNORECASE,
)

# --- HybridTextDetector Class (Adapted for API) ---
class HybridTextDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading models on {self.device}...")
        self.classifier_max_length = CLASSIFIER_MAX_LENGTH
        self.classifier_temperature = max(CLASSIFIER_TEMPERATURE, 1e-6)

        # Load classification model
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
        except Exception as e:
            print(f"Error loading classifier: {e}")
            raise e

        # Load span detection model
        try:
            self.span_tokenizer = BertTokenizer.from_pretrained(SPAN_MODEL_PATH)
            self.span_detector = BertForTokenClassification.from_pretrained(SPAN_MODEL_PATH).to(
                self.device
            )
            self.span_detector.eval()
            print(f"Span detector loaded ({SPAN_MODEL_PATH}).")
        except Exception as e:
            print(f"Error loading span detector: {e}")
            # We might want to continue if only span detector fails, but for now let's raise
            raise e

    def classify(self, text):
        """Determine if text is AI-generated"""
        encoding = self.classifier_tokenizer(
            text, max_length=self.classifier_max_length, padding='max_length',
            truncation=True, return_tensors='pt'
        )

        with torch.no_grad():
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            outputs = self.classifier(input_ids=input_ids, attention_mask=attention_mask)
            scaled_logits = outputs.logits[0] / self.classifier_temperature
            probs = torch.softmax(scaled_logits, dim=0)
            pred = torch.argmax(scaled_logits).item()

        return {
            'label': 'AI' if pred == 1 else 'Human',
            'confidence': probs[pred].item(),
            'prob_human': probs[0].item(),
            'prob_ai': probs[1].item()
        }
    
    def detect_boundary(self, text):
        """Detect boundary in mixed text"""
        # Remove [SEP] if present for span detection
        text_clean = text.replace('[SEP]', '')
        
        encoding = self.span_tokenizer(
            text_clean, max_length=512, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        
        with torch.no_grad():
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            outputs = self.span_detector(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits[0], dim=-1).cpu()
        
        # Find boundary
        tokens = self.span_tokenizer.convert_ids_to_tokens(input_ids[0])
        labels = preds.numpy()
        
        # Find transition from Human(0) to AI(1)
        boundary_idx = None
        for i in range(1, len(labels)):
            if labels[i-1] == 0 and labels[i] == 1:
                boundary_idx = i
                break
        
        # Map back to character position
        # This is an approximation as tokenization is lossy regarding exact char positions
        # But for this demo we can try to reconstruct
        
        char_pos = 0
        boundary_char = None
        
        for i, token in enumerate(tokens):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
            
            token_text = token.replace('##', '')
            
            if i == boundary_idx:
                boundary_char = char_pos
                break # Found it
            
            char_pos += len(token_text)
            
        return {
            'boundary_token': boundary_idx,
            'boundary_char': boundary_char,
            'text': text_clean
        }

# --- API Setup ---
app = FastAPI(title="AI Text Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global detector instance
detector = None

@app.on_event("startup")
async def startup_event():
    global detector
    detector = HybridTextDetector()


@app.get("/api/health")
async def health_check():
    """Health endpoint for frontend-backend connectivity checks."""
    return {
        "status": "ok",
        "detectorReady": detector is not None,
        "modelVersion": MODEL_VERSION,
        "decisionThreshold": DECISION_THRESHOLD,
        "maxLength": CLASSIFIER_MAX_LENGTH,
    }

class DetectRequest(BaseModel):
    text: str

class SentenceResult(BaseModel):
    text: str
    isAI: bool
    confidence: float

class DetectionResponse(BaseModel):
    type: str # "human" | "ai" | "mixed"
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
    """Infer a coarse domain hint from text content."""
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
    """Collect lightweight risk flags for downstream monitoring."""
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

@app.post(
    "/api/detect",
    response_model=DetectionResponse,
    response_model_exclude_none=True,
)
async def detect_text(request: DetectRequest):
    if not detector:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    start_time = time.time()
    text = request.text
    
    # 1. Classify
    cls_result = detector.classify(text)
    
    label = cls_result['label'] # 'AI' or 'Human'
    confidence = cls_result['confidence'] * 100
    prob_ai = cls_result['prob_ai']
    prob_human = cls_result['prob_human']
    
    ai_percentage = int(prob_ai * 100)
    human_percentage = int(prob_human * 100)
    
    # Determine type
    result_type = "mixed"
    boundary_char = None
    
    threshold_percent = int(DECISION_THRESHOLD * 100)
    if ai_percentage > threshold_percent:
        result_type = "ai"
    elif human_percentage > threshold_percent:
        result_type = "human"
    else:
        # Mixed likely, or uncertain. Let's run boundary detection to be sure or if it is labeled AI
        # Actually, the logic in visualize_detection says: if label == 'AI', check boundary.
        # But here we want to support 'mixed' explicitly.
        # Let's trust the classifier probabilities for now.
        result_type = "mixed"

    # If it's AI or Mixed, try to find boundary
    if result_type in ["ai", "mixed"] or label == 'AI':
        boundary_res = detector.detect_boundary(text)
        if boundary_res['boundary_char'] is not None:
             boundary_char = boundary_res['boundary_char']
             # If boundary is found, it strongly suggests mixed (Human -> AI)
             result_type = "mixed"
        elif result_type == "mixed":
            # If no boundary found but prob is mixed, maybe it's just fully AI or fully Human but uncertain
            # Let's fallback to dominant class
            result_type = "ai" if prob_ai > prob_human else "human"

    # Split into sentences for frontend
    # Regex from frontend: /[。！？.!?]/
    # We keep delimiters to reconstruct length properly, but for the list we just want the content
    # Frontend logic: split(/[。！？.!?]/).filter((s) => s.trim())
    
    # We need to map boundary_char to sentence index
    sentences_raw = SENTENCE_SPLIT_PATTERN.split(text)
    sentences = []
    current_char_count = 0
    boundary_sentence_index = None
    
    # Reconstruct sentences by appending delimiters to previous part
    temp_sentences = []
    current_sent = ""
    
    for part in sentences_raw:
        if SENTENCE_SPLIT_PATTERN.match(part):
            current_sent += part
            temp_sentences.append(current_sent)
            current_sent = ""
        else:
            if current_sent:
                temp_sentences.append(current_sent)
            current_sent = part
    if current_sent:
        temp_sentences.append(current_sent)
        
    # Filter empty like frontend
    final_sentences = [s for s in temp_sentences if s.strip()]
    
    # Now find which sentence contains the boundary_char
    running_char_count = 0
    for idx, sent in enumerate(final_sentences):
        sent_len = len(sent)
        if boundary_char is not None:
            if running_char_count <= boundary_char < running_char_count + sent_len:
                boundary_sentence_index = idx
        running_char_count += sent_len

    # Construct sentence details
    sentence_results = []
    for i, sent in enumerate(final_sentences):
        is_ai = False
        if result_type == "ai":
            is_ai = True
        elif result_type == "human":
            is_ai = False
        elif result_type == "mixed":
            if boundary_sentence_index is not None:
                if i >= boundary_sentence_index:
                    is_ai = True
            else:
                 # Fallback if mixed but no boundary (shouldn't happen given logic above, but safe fallback)
                 is_ai = True if ai_percentage > 50 else False
        
        # Add some randomness to sentence confidence to look realistic (since we don't have per-sentence confidence from model easily without running it N times)
        # Or we could just use the global confidence
        sent_conf = confidence # Simple approximation
        
        sentence_results.append(SentenceResult(
            text=sent,
            isAI=is_ai,
            confidence=sent_conf
        ))

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

# AI续写和润色API端点
class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[dict]
    temperature: float = 0.7
    max_tokens: int = 1000


def resolve_api_key(authorization_header: str | None) -> str | None:
    """Resolve provider key from env first, then optional Authorization header."""
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
async def chat_completions(
    request: ChatRequest,
    authorization: str | None = Header(default=None),
):
    """OpenAI兼容的聊天接口，用于AI续写和润色"""

    # 从环境变量读取API密钥
    api_key = resolve_api_key(authorization)
    api_base = os.getenv("OPENAI_BASE_URL", "https://api.hotaruapi.top/v1")
    model_name = request.model or DEFAULT_CHAT_MODEL

    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set")

    try:
        response = requests.post(
            f"{api_base}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model_name,
                "messages": request.messages,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
            },
            timeout=60
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        
        result = response.json()
        return result
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="API请求超时")
    except requests.exceptions.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"API返回格式错误: {response.text[:200]}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API调用失败: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
