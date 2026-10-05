import logging
import os
import re
import shutil
import sys
import time
import uuid
from pathlib import Path

import spacy
import torch
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]       
SRC_DIR = PROJECT_ROOT / "src"                            
MODEL_DIR = SRC_DIR / "models" / "clause_classifier_model"
UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

sys.path.append(str(SRC_DIR / "ingestion"))
from ocr_pipline import ingest_pdf 

CONFIDENCE_THRESHOLD = 0.60
CHUNK_WORD_SIZE = 150 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "app.log"),
    ],
)
logger = logging.getLogger("contract_intelligence")

app = FastAPI(title="Contract Intelligence & Risk Scoring API")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000
    logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({duration_ms:.0f}ms)")
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs = {}

tokenizer = None
model = None
nlp = None
embedder = None
pinecone_index = None

INDEX_NAME = "contract-clauses"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ENTITY_LABELS = {"ORG", "DATE", "MONEY", "GPE", "LAW"}


@app.on_event("startup")
def load_model():
    global tokenizer, model, nlp
    logger.info(f"Loading fine-tuned model from {MODEL_DIR} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.eval()
    logger.info("Model loaded.")
    logger.info("Loading spaCy model for entity extraction ...")
    nlp = spacy.load("en_core_web_sm")
    logger.info("spaCy loaded. Startup complete.")


def get_search_components():
    global embedder, pinecone_index
    if embedder is None or pinecone_index is None:
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise HTTPException(status_code=503, detail="PINECONE_API_KEY environment variable is not set.")
        from pinecone import Pinecone
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(EMBEDDING_MODEL)
        pinecone_index = Pinecone(api_key=api_key).Index(INDEX_NAME)
    return embedder, pinecone_index


def extract_entities(text: str) -> list:
    doc = nlp(text[:100_000])
    seen, entities = set(), []
    for ent in doc.ents:
        key = (ent.text, ent.label_)
        if ent.label_ in ENTITY_LABELS and key not in seen and len(ent.text) <= 60:
            seen.add(key)
            entities.append({"text": ent.text, "label": ent.label_})
    return entities


def clean_clause_label(raw_label: str) -> str:
    match = re.search(r'"([^"]+)"', raw_label)
    return match.group(1) if match else raw_label


def chunk_text(text: str, max_words: int = CHUNK_WORD_SIZE) -> list:
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def classify_chunk(text: str) -> tuple:
    inputs = tokenizer(text, truncation=True, padding=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    confidence, pred_id = torch.max(probs, dim=-1)
    raw_label = model.config.id2label[pred_id.item()]
    return clean_clause_label(raw_label), confidence.item()


def process_document(job_id: str, pdf_path: str):
    jobs[job_id]["status"] = "processing"
    logger.info(f"[{job_id}] Processing started")
    start = time.time()
    try:
        ingestion_result = ingest_pdf(pdf_path)
        text = ingestion_result["text"]
        logger.info(f"[{job_id}] Text extracted via '{ingestion_result['extraction_method']}' ({len(text)} chars)")

        chunks = [c for c in chunk_text(text) if len(c.strip()) >= 20]
        needs_review_count = 0
        clauses = []
        for chunk in chunks:
            label, confidence = classify_chunk(chunk)
            status = "ok" if confidence >= CONFIDENCE_THRESHOLD else "needs_review"
            if status == "needs_review":
                needs_review_count += 1
            clauses.append({
                "text_preview": chunk[:300],
                "predicted_clause_type": label,
                "confidence": round(confidence, 4),
                "status": status,
            })

        jobs[job_id].update({
            "status": "done",
            "extraction_method": ingestion_result["extraction_method"],
            "num_chunks": len(chunks),
            "entities": extract_entities(text),
            "clauses": clauses,
        })
        duration_s = time.time() - start
        logger.info(
            f"[{job_id}] Done in {duration_s:.1f}s — {len(chunks)} clauses, "
            f"{needs_review_count} flagged needs_review"
        )
    except Exception as e:
        jobs[job_id].update({"status": "failed", "error": str(e)})
        logger.error(f"[{job_id}] Processing failed: {e}", exc_info=True)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/contracts/upload")
async def upload_contract(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported right now.")

    job_id = str(uuid.uuid4())
    dest_path = UPLOAD_DIR / f"{job_id}.pdf"
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    jobs[job_id] = {"status": "queued", "filename": file.filename}
    background_tasks.add_task(process_document, job_id, str(dest_path))
    logger.info(f"[{job_id}] Upload received: {file.filename}")

    return {"job_id": job_id, "status": "queued"}


@app.get("/search")
def search_clauses(q: str, top_k: int = 5):
    logger.info(f"Search query: {q!r} (top_k={top_k})")
    emb, index = get_search_components()
    query_vector = emb.encode([q]).tolist()[0]
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    return [
        {
            "score": round(m["score"], 4),
            "clause_type": clean_clause_label(m["metadata"]["clause_type"]),
            "text": m["metadata"]["text"],
        }
        for m in results["matches"]
    ]


@app.get("/contracts/{job_id}")
def get_contract_result(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="No job found with that id.")
    return jobs[job_id]
