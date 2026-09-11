# AI-Powered Contract Intelligence & Risk Scoring (NLP)

A sophisticated NLP platform designed for legal and compliance teams. The
system ingests lengthy legal contracts (PDF/Word documents), automatically
extracts key entities (dates, parties, jurisdictions), identifies specific
clause types (e.g., termination, confidentiality), and flags anomalous or
high-risk language using a fine-tuned Large Language Model (LLM) and Named
Entity Recognition (NER).

## Why this project

Legal and compliance teams spend enormous manual effort reviewing contracts
for hidden liability clauses and unfavorable auto-renewal terms. This
platform drastically reduces that manual review time and helps ensure
high-volume contract environments don't miss critical risk language.

## Data source

- **Primary source:** CUAD (Contract Understanding Atticus Dataset)
- **Characteristics:** 500+ commercial contracts, expertly annotated across
  41 distinct legal clause categories
- Loaded via Hugging Face Datasets (`datasets.load_dataset("cuad")`)

## Tech stack

| Layer                 | Tools |
|------------------------|-------|
| Language               | Python |
| NLP & ML                | Hugging Face Transformers (BERT/RoBERTa), spaCy, PyTorch |
| Information retrieval   | Pinecone / Milvus (vector DB), LangChain |
| Backend / API           | FastAPI, Uvicorn, Celery (async document processing) |
| Deployment              | Docker, AWS EC2 |