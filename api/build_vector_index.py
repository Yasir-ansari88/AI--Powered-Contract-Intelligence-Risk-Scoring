import json
import getpass
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone ServerlessSpec

TRAIN_PATH = "/content/AI--Powered-Contract-Intelligence-Risk-Scoring/clause_train.json"
TEST_PATH = "/content/AI--Powered-Contract-Intelligence-Risk-Scoring/clause_test.json"

INDEX_NAME = "contract-clauses"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
BATCH_SIZE = 100

def load_all_clauses():
    with open (TRAIN_PATH, "r", encoding = "utf-8") as f:
        train_rows = json.load(f)
    with open (TEST_PATH, "r", encoding="utf-8") as f:
        test_rows = json.load(f)
    return train_rows + test_rows

def main():
    api_key = getpass.getpass("Enter you pinecone api key :")
    pc = Pinecone(api_key = api_key)

    if INDEX_NAME not in [idx["name"] for idx in pc.list_indexes()]:
        print(f"Creating index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    else:
        print(f"Index '{INDEX_NAME}' already exists, reusing it.")