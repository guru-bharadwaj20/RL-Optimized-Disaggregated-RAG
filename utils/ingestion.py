"""
Corpus ingestion utility — chunks, embeds, and stores documents into Qdrant.

Uses the same RecursiveCharacterTextSplitter logic as the encoder service
so that query embeddings and document embeddings are produced by the same
model with the same chunking strategy.

Usage:
    python utils/ingestion.py [path/to/corpus.txt]

Environment variables (all optional — defaults work with docker-compose):
    QDRANT_HOST          (default: localhost)
    QDRANT_PORT          (default: 6333)
    QDRANT_COLLECTION    (default: rag_collection)
    EMBEDDING_MODEL      (default: all-MiniLM-L6-v2)
    VECTOR_DIMENSION     (default: 384)
    CHUNK_SIZE           (default: 500)
    CHUNK_OVERLAP        (default: 100)
"""
import os
import sys
import hashlib
import logging
from typing import List

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("ingestion")

MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_collection")
VECTOR_SIZE = int(os.getenv("VECTOR_DIMENSION", 384))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))

DEFAULT_CORPUS = os.path.join(os.path.dirname(__file__), "..", "data", "corpus.txt")


def recursive_text_splitter(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> List[str]:
    """Recursive character text splitter — matches encoder service implementation."""
    separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

    def _split(text: str, seps: List[str]) -> List[str]:
        if len(text) <= chunk_size:
            return [text.strip()] if text.strip() else []

        sep = seps[0]
        rest = seps[1:]
        parts = text.split(sep) if sep else list(text)

        chunks: List[str] = []
        current = ""
        for part in parts:
            candidate = (current + sep + part) if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current.strip())
                if len(part) > chunk_size and rest:
                    sub = _split(part, rest)
                    if sub:
                        chunks.extend(sub[:-1])
                        current = sub[-1]
                    else:
                        current = ""
                elif len(part) > chunk_size:
                    for i in range(0, len(part), chunk_size - overlap):
                        seg = part[i : i + chunk_size]
                        if seg.strip():
                            chunks.append(seg.strip())
                    current = ""
                else:
                    current = part
        if current.strip():
            chunks.append(current.strip())
        return [c for c in chunks if c]

    raw = _split(text, separators)
    if overlap <= 0 or len(raw) <= 1:
        return raw
    result = [raw[0]]
    for i in range(1, len(raw)):
        tail = raw[i - 1][-overlap:]
        result.append((tail + " " + raw[i]).strip())
    return result


def load_corpus(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Corpus not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main(corpus_path: str = DEFAULT_CORPUS):
    from sentence_transformers import SentenceTransformer
    from qdrant_client import QdrantClient
    from qdrant_client import models as qm

    logger.info(f"Loading corpus: {corpus_path}")
    text = load_corpus(corpus_path)
    source = os.path.basename(corpus_path)

    chunks = recursive_text_splitter(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    logger.info(f"Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    logger.info(f"Loading embedding model: {MODEL}")
    model = SentenceTransformer(MODEL)
    embeddings = model.encode(
        chunks,
        convert_to_tensor=False,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32,
    )
    logger.info(f"Generated {len(embeddings)} embeddings (dim={VECTOR_SIZE})")

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Drop + recreate for a clean slate
    try:
        client.delete_collection(COLLECTION)
        logger.info(f"Deleted existing collection '{COLLECTION}'")
    except Exception:
        pass

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=qm.VectorParams(size=VECTOR_SIZE, distance=qm.Distance.COSINE),
    )
    logger.info(f"Created collection '{COLLECTION}'")

    points = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        raw_id = int(hashlib.sha256(f"{source}::{i}".encode()).hexdigest(), 16)
        point_id = raw_id % (2**63 - 1)
        points.append(
            qm.PointStruct(
                id=point_id,
                vector=emb.tolist(),
                payload={
                    "document": chunk,
                    "source": source,
                    "chunk_index": i,
                },
            )
        )

    client.upsert(collection_name=COLLECTION, points=points, wait=True)
    logger.info(f"Upserted {len(points)} points into '{COLLECTION}' — ingestion complete.")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CORPUS
    main(path)
