"""Ingest corpus into Qdrant using sentence-transformers.

Reads data/corpus.txt, chunks by paragraph, computes embeddings,
then creates/recreates the 'rag_collection' collection in Qdrant.

Usage:
    python utils/ingestion.py

Environment variables (all optional — defaults work with docker-compose):
    QDRANT_HOST          (default: localhost)
    QDRANT_PORT          (default: 6333)
    QDRANT_COLLECTION    (default: rag_collection)
    EMBEDDING_MODEL      (default: all-MiniLM-L6-v2)
    VECTOR_DIMENSION     (default: 384)
"""
import os
from typing import List

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_collection")  # matches retriever default
VECTOR_SIZE = int(os.getenv("VECTOR_DIMENSION", 384))

CORPUS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "corpus.txt")


def load_corpus(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Corpus not found at: {path}")
    text = open(path, "r", encoding="utf-8").read()
    chunks = [p.strip() for p in text.split("\n\n") if p.strip()]
    return chunks


def main():
    print(f"Loading corpus from: {CORPUS_PATH}")
    chunks = load_corpus(CORPUS_PATH)
    print(f"Loaded {len(chunks)} document chunks")

    print(f"Loading embedding model: {MODEL}")
    model = SentenceTransformer(MODEL)
    embeddings = model.encode(chunks, convert_to_tensor=False, show_progress_bar=True)
    print(f"Generated {len(embeddings)} embeddings (dim={VECTOR_SIZE})")

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Drop existing collection so we start fresh on re-ingestion
    try:
        client.delete_collection(collection_name=COLLECTION)
        print(f"Deleted existing collection '{COLLECTION}'")
    except Exception:
        pass

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
    )
    print(f"Created collection '{COLLECTION}'")

    points = [
        models.PointStruct(
            id=i + 1,
            vector=emb.tolist(),
            payload={"document": chunk},  # 'document' key matches retriever expectation
        )
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]

    client.upsert(collection_name=COLLECTION, points=points, wait=True)
    print(f"Upserted {len(points)} points into '{COLLECTION}'")
    print("Ingestion complete.")


if __name__ == "__main__":
    main()
