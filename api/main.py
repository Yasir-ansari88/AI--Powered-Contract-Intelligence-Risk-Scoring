import shutil
import sys
import uuid
from pathlib import Path

import torch
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models" / "clause_classifier_model"
UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

sys.path.append(str(BASE_DIR / "ingestion"))
from ocr_pipline import ingest_pdf

CONFIDENCE_THRESHOLD = 0.60
CHUNK_WORD_SIZE = 150

app = FastAPI(title="Contract Intelligence & Risk Scoring API")

jobs = {}

tokenizer = None
model = None

@app.on_event("startup")
def  load_model():
    global tokenizer, model
    print(f"Loading fine-tuned model from {MODEL_DIR} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.eval()
    print("Model loaded.")

def chunk_text(text: str, max_words: int = CHUNK_WORD_SIZE) -> list:
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def classify_chunk(text: str) -> tuple:
    inputs = tokenizer(text, truncation=True, padding=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    confidence, pred_id = torch.max(probs, dim=-1)
    label = model.config.id2label[pred_id.item()]
    return label, confidence.item()

