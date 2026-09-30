FROM python:3.11-slim

# System dependencies needed for OCR (pdf2image + pytesseract)
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .


RUN pip install --no-cache-dir --default-timeout=200 --retries 10 \
    torch --index-url https://download.pytorch.org/whl/cpu


RUN pip install --no-cache-dir --default-timeout=200 --retries 10 -r requirements.txt

RUN python -m spacy download en_core_web_sm

COPY api/ ./api/
COPY src/ ./src/

WORKDIR /app/api
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]