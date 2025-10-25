"""Ingest sample corpus into Qdrant using sentence-transformers.

This script will:
- read data/corpus.txt
- chunk it (simple split by paragraphs)
- compute embeddings with all-MiniLM-L6-v2
- create or recreate a Qdrant collection named 'rag_knowledge'
- upsert points with payload {"document": <chunk_text>}

Usage: python utils/ingestion.py
"""
import os
import json
from typing import List

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION = os.getenv("QDRANT_COLLECTION_NAME", "rag_knowledge")
VECTOR_SIZE = int(os.getenv("VECTOR_DIMENSION", 384))


def load_corpus(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    text = open(path, "r", encoding="utf-8").read()
    # Simple paragraph splitting
    chunks = [p.strip() for p in text.split("\n\n") if p.strip()]
    return chunks


def embed_chunks(chunks: List[str]):
    model = SentenceTransformer(MODEL)
    embeddings = model.encode(chunks, convert_to_tensor=False)
    return embeddings


def main():
    corpus = load_corpus("data/corpus.txt")
    print(f"Loaded {len(corpus)} chunks from data/corpus.txt")
    embeddings = embed_chunks(corpus)

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    # Recreate collection (safe for dev only)
    try:
        client.recreate_collection(collection_name=COLLECTION, vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE))
    except Exception:
        # If recreate fails, try create
        client.create_collection(collection_name=COLLECTION, vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE))

    points = []
    for i, (chunk, emb) in enumerate(zip(corpus, embeddings), start=1):
        points.append(models.PointStruct(id=i, vector=emb.tolist(), payload={"document": chunk}))

    client.upsert(collection_name=COLLECTION, points=points, wait=True)
    print(f"Upserted {len(points)} points into Qdrant collection '{COLLECTION}'")


if __name__ == '__main__':
    main()
import os
import time
import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

# Load .env variables
load_dotenv()

# Configuration from .env
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = os.getenv("QDRANT_PORT")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
VECTOR_DIMENSION = 384 # For all-MiniLM-L6-v2

DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/corpus.txt')

def load_data(file_path):
    """Loads and chunks the text data."""
    with open(file_path, 'r') as f:
        full_text = f.read()
    
    # Simple fixed-size chunking for hackathon speed
    chunk_size = 500 
    overlap = 50
    chunks = []
    
    i = 0
    while i < len(full_text):
        chunk = full_text[i:i + chunk_size]
        chunks.append(chunk)
        i += chunk_size - overlap
    
    return chunks

def run_ingestion():
    """Initializes Qdrant and uploads vectors in batches."""
    print("--- Starting Qdrant Ingestion Pipeline ---")
    
    # 1. Initialize Clients and Model
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=10)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL, device='cpu')
    
    # 2. Load and Chunk Data
    chunks = load_data(DATA_PATH)
    print(f"Loaded {len(chunks)} chunks for indexing.")
    
    # 3. Create or Recreate Collection
    qdrant_client.recreate_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=models.VectorParams(size=VECTOR_DIMENSION, distance=models.Distance.COSINE)
    )
    print(f"Collection '{QDRANT_COLLECTION_NAME}' created/recreated.")
    
    # 4. Generate Embeddings and Upsert Points
    # Generate all embeddings at once for simplicity, then structure points
    vectors = embedding_model.encode(chunks, show_progress_bar=True).tolist()
    
    points = []
    for i, (vector, chunk) in enumerate(zip(vectors, chunks)):
        points.append(
            models.PointStruct(
                id=i,
                vector=vector,
                # Store the original text (chunk) as payload for retrieval
                payload={"text": chunk, "source_id": f"doc_{i}"}
            )
        )

    # Upsert data in a single batch (or multiple batches for larger data)
    qdrant_client.upsert(
        collection_name=QDRANT_COLLECTION_NAME,
        points=points,
        wait=True
    )
    
    print(f"Ingestion complete. Total points in Qdrant: {len(points)}")

if __name__ == "__main__":
    # Ensure you have a placeholder corpus.txt file in the data/ directory before running.
    run_ingestion()