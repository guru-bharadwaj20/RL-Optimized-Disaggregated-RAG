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