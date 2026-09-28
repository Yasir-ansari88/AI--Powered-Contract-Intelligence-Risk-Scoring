import os
import shutil
import sys
import uuid
from pathlib import Path

import spacy
import torch
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]  
SRC_DIR = PROJECT_ROOT / "src"                             
MODEL_DIR = SRC_DIR / "models" / "clause_classifier_model"
UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

sys.path.append(str(SRC_DIR / "ingestion"))
from ocr_pipline import ingest_pdf  

CONFIDENCE_THRESHOLD = 0.60
CHUNK_WORD_SIZE = 150  

app = FastAPI(title="Contract Intelligence & Risk Scoring API")

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
    print(f"Loading fine-tuned model from {MODEL_DIR} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.eval()
    print("Model loaded.")
    print("Loading spaCy model for entity extraction ...")
    nlp = spacy.load("en_core_web_sm")
    print("spaCy loaded.")

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
        if ent.label_ in ENTITY_LABELS and key not in seen:
            seen.add(key)
            entities.append({"text": ent.text, "label": ent.label_})
    return entities

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

def process_document(job_id: str, pdf_path: str):
    jobs[job_id]["status"] = "processing"
    try:
        ingestion_result = ingest_pdf(pdf_path)
        text = ingestion_result["text"]

        chunks = [c for c in chunk_text(text) if len(c.strip()) >= 20]
        clauses = []
        for chunk in chunks:
            label, confidence = classify_chunk(chunk)
            clauses.append({
                "text_preview": chunk[:300],
                "predicted_clause_type": label,
                "confidence": round(confidence, 4),
                "status": "ok" if confidence >= CONFIDENCE_THRESHOLD else "needs_review",
            })

        jobs[job_id].update({
            "status": "done",
            "extraction_method": ingestion_result["extraction_method"],
            "num_chunks": len(chunks),
            "entities": extract_entities(text),
            "clauses": clauses,
        })
    except Exception as e:
        jobs[job_id].update({"status": "failed", "error": str(e)})


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

    return {"job_id": job_id, "status": "queued"}

@app.get("/search")
def search_clauses(q: str, top_k: int = 5):
    emb, index = get_search_components()
    query_vector = emb.encode([q]).tolist()[0]
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    return [
        {
            "score": round(m["score"], 4),
            "clause_type": m["metadata"]["clause_type"],
            "text": m["metadata"]["text"],
        }
        for m in results["matches"]
    ]

@app.get("/contracts/{job_id}")
def get_contract_result(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="No job found with that id.")
    return jobs[job_id]