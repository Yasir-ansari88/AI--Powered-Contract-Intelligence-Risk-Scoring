import json
import getpass
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

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

    index = pc.Index(INDEX_NAME)

    print("Loading clauses...")
    clauses = load_all_clauses()
    print(f"Total clauses to embed: {len(clauses)}")

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Embedding and upserting to Pinecone...")
    for i in range(0, len(clauses), BATCH_SIZE):
        batch = clauses[i:i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        vectors = []
        for j, (clause, embedding) in enumerate(zip(batch, embeddings)):
            vectors.append({
                "id": f"clause-{i + j}",
                "values": embedding,
                "metadata": {
                    "clause_type": clause["label"],
                    "text": clause["text"][:1000],
                },
            })

        index.upsert(vectors=vectors)
        print(f"  Upserted {min(i + BATCH_SIZE, len(clauses))}/{len(clauses)}")

    print("\nDone. Index stats:")
    print(index.describe_index_stats())

def search_example(query_text: str, top_k: int = 5):
    api_key = getpass.getpass("Enter your Pinecone API key: ")
    pc = Pinecone(api_key=api_key)
    index = pc.Index(INDEX_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL)

    query_embedding = model.encode([query_text]).tolist()[0]
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

    for match in results["matches"]:
        print(f"\nScore: {match['score']:.4f}")
        print(f"Clause type: {match['metadata']['clause_type']}")
        print(f"Text: {match['metadata']['text'][:200]}...")


if __name__ == "__main__":
    main()