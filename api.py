import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertForTokenClassification
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import re
import time

# --- HybridTextDetector Class (Adapted for API) ---
class HybridTextDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading models on {self.device}...")
        
        # Load classification model
        try:
            self.classifier_tokenizer = BertTokenizer.from_pretrained('models/bert_v2_with_sep')
            self.classifier = BertForSequenceClassification.from_pretrained('models/bert_v2_with_sep').to(self.device)
            self.classifier.eval()
            print("Classifier loaded.")
        except Exception as e:
            print(f"Error loading classifier: {e}")
            raise e

        # Load span detection model
        try:
            self.span_tokenizer = BertTokenizer.from_pretrained('models/bert_span_detector')
            self.span_detector = BertForTokenClassification.from_pretrained('models/bert_span_detector').to(self.device)
            self.span_detector.eval()
            print("Span detector loaded.")
        except Exception as e:
            print(f"Error loading span detector: {e}")
            # We might want to continue if only span detector fails, but for now let's raise
            raise e
    
    def classify(self, text):
        """Determine if text is AI-generated"""
        encoding = self.classifier_tokenizer(
            text, max_length=512, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        
        with torch.no_grad():
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            outputs = self.classifier(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits[0], dim=0)
            pred = torch.argmax(outputs.logits[0]).item()
        
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

@app.post("/api/detect", response_model=DetectionResponse)
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
    
    if ai_percentage > 80:
        result_type = "ai"
    elif human_percentage > 80:
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
    sentences_raw = re.split(r'([。！？.!?])', text)
    sentences = []
    current_char_count = 0
    boundary_sentence_index = None
    
    # Reconstruct sentences by appending delimiters to previous part
    temp_sentences = []
    current_sent = ""
    
    for part in sentences_raw:
        if re.match(r'[。！？.!?]', part):
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
    
    return DetectionResponse(
        type=result_type,
        confidence=confidence,
        humanPercentage=human_percentage,
        aiPercentage=ai_percentage,
        boundary=boundary_sentence_index,
        sentences=sentence_results,
        processingTime=processing_time
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
