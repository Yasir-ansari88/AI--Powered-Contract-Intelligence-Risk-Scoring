import shutil
import sys
import uuid
from pathlib import path

import torch
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models" / "clause_classifier_model"
UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

sys.path.append(str(BASE_DIR / "ingestion"))
from ocr_pipeline import ingest_pdf

CONFIDENCE_THRESHOLD = 0.60
CHUNK_WORD_SIZE = 150

app = FastAPI(title="Contract Intelligence & Risk Scoring API")