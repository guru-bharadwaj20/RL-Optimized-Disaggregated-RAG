import os
import time
import json
import asyncio
import hashlib
import logging
from typing import Optional
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("encoder")

load_dotenv()

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
FALLBACK_MODEL = os.getenv("EMBEDDING_MODEL_FALLBACK", "BAAI/bge-small-en-v1.5")
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", 384))
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_collection")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))

app = FastAPI(title="Encoder Service", version="2.0")

_model = None


# ── Models ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    text: str


class BatchEncodeRequest(BaseModel):
    texts: list[str]


class VectorResponse(BaseModel):
    vector: list[float]
    dim: int = VECTOR_DIMENSION


class BatchVectorResponse(BaseModel):
    vectors: list[list[float]]
    count: int


class IngestRequest(BaseModel):
    text: str
    source: str = "uploaded_document"


class IngestResponse(BaseModel):
    chunks_added: int
    source: str
    processing_time_ms: float


# ── Model Management ──────────────────────────────────────────────────────────

def load_model():
    global _model
    if _model is not None:
        return _model
    for model_name in [MODEL_NAME, FALLBACK_MODEL]:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {model_name}")
            _model = SentenceTransformer(model_name, device=EMBEDDING_DEVICE)
            logger.info(f"Model '{model_name}' loaded on {EMBEDDING_DEVICE}")
            return _model
        except Exception as e:
            logger.warning(f"Failed to load {model_name}: {e}")
    logger.error("Could not load any embedding model")
    return None


def get_redis_client():
    try:
        import redis
        return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)
    except Exception:
        return None


def get_qdrant_client():
    from qdrant_client import QdrantClient
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


# ── Chunking ──────────────────────────────────────────────────────────────────

def recursive_text_splitter(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Recursive character text splitter similar to LangChain's RecursiveCharacterTextSplitter.
    Tries separators in order: paragraph → newline → sentence → space → character.
    """
    separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

    def _split(text: str, seps: list[str]) -> list[str]:
        if len(text) <= chunk_size:
            return [text.strip()] if text.strip() else []

        sep = seps[0]
        rest = seps[1:]

        parts = text.split(sep) if sep else list(text)
        chunks: list[str] = []
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
                    # Hard split by character
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

    raw_chunks = _split(text, separators)

    # Apply overlap: prepend tail of previous chunk to each subsequent chunk
    if overlap <= 0 or len(raw_chunks) <= 1:
        return raw_chunks

    result = [raw_chunks[0]]
    for i in range(1, len(raw_chunks)):
        tail = raw_chunks[i - 1][-overlap:]
        result.append((tail + " " + raw_chunks[i]).strip())
    return result


def _ensure_collection(client) -> None:
    from qdrant_client import models as qm
    try:
        client.get_collection(QDRANT_COLLECTION)
    except Exception:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=qm.VectorParams(size=VECTOR_DIMENSION, distance=qm.Distance.COSINE),
        )
        logger.info(f"Created Qdrant collection '{QDRANT_COLLECTION}'")


# ── HTTP Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    m = load_model()
    if m is None:
        raise HTTPException(status_code=503, detail="Embedding model not available")
    return {"status": "ok", "model": MODEL_NAME, "device": EMBEDDING_DEVICE}


@app.post("/encode", response_model=VectorResponse)
async def encode_query(request: QueryRequest):
    """Encode a single text string into an embedding vector."""
    m = load_model()
    if m is None:
        raise HTTPException(status_code=503, detail="Model unavailable")
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")
    try:
        embedding = m.encode(request.text, convert_to_tensor=False, normalize_embeddings=True)
        dim = int(getattr(embedding, "shape", (len(embedding),))[0])
        return VectorResponse(vector=embedding.tolist(), dim=dim)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Encoding failed: {exc}")


@app.post("/encode_batch", response_model=BatchVectorResponse)
async def encode_batch(request: BatchEncodeRequest):
    """Encode multiple texts in a single batched call (more efficient than repeated /encode)."""
    m = load_model()
    if m is None:
        raise HTTPException(status_code=503, detail="Model unavailable")
    if not request.texts:
        raise HTTPException(status_code=400, detail="texts list cannot be empty")
    try:
        embeddings = m.encode(
            request.texts,
            convert_to_tensor=False,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
        return BatchVectorResponse(
            vectors=[emb.tolist() for emb in embeddings],
            count=len(embeddings),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Batch encoding failed: {exc}")


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(request: IngestRequest):
    """
    Chunk text, embed each chunk, and upsert into Qdrant.
    Accepts documents from PDF/TXT/MD upload via the API Gateway.
    Each chunk gets metadata: source filename, chunk_index.
    """
    t0 = time.time()
    m = load_model()
    if m is None:
        raise HTTPException(status_code=503, detail="Embedding model unavailable")

    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="No text content provided")

    # 1. Chunk
    chunks = recursive_text_splitter(request.text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    if not chunks:
        raise HTTPException(status_code=400, detail="Text produced no usable chunks")

    logger.info(f"Ingesting '{request.source}': {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    # 2. Embed
    try:
        embeddings = m.encode(
            chunks,
            convert_to_tensor=False,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    # 3. Build Qdrant points with stable deterministic IDs
    from qdrant_client import models as qm

    points = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        # Stable ID: hash of (source + chunk_index) so re-ingesting same file overwrites
        raw_id = int(hashlib.sha256(f"{request.source}::{i}".encode()).hexdigest(), 16)
        point_id = raw_id % (2**63 - 1)
        points.append(
            qm.PointStruct(
                id=point_id,
                vector=emb.tolist(),
                payload={
                    "document": chunk,
                    "source": request.source,
                    "chunk_index": i,
                },
            )
        )

    # 4. Upsert into Qdrant
    try:
        client = get_qdrant_client()
        _ensure_collection(client)
        client.upsert(collection_name=QDRANT_COLLECTION, points=points, wait=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Qdrant upsert failed: {e}")

    processing_ms = (time.time() - t0) * 1000
    logger.info(f"Ingested {len(points)} chunks from '{request.source}' in {processing_ms:.0f}ms")
    return IngestResponse(
        chunks_added=len(points),
        source=request.source,
        processing_time_ms=processing_ms,
    )


# ── Background Worker (Redis Queue) ──────────────────────────────────────────

async def encoder_worker_loop(poll_interval: float = 1.0):
    """BRPOP from job:encoder_in → embed → LPUSH to job:retriever_in."""
    r = get_redis_client()
    if r is None:
        logger.error("Encoder worker: Redis unavailable, worker will not start")
        return

    model = load_model()
    if model is None:
        logger.warning("Encoder worker: model not loaded at startup; will retry per job")

    while True:
        try:
            item = r.brpop("job:encoder_in", timeout=5)
            if not item:
                await asyncio.sleep(poll_interval)
                continue

            _, raw = item
            try:
                job = json.loads(raw)
            except Exception:
                try:
                    job = json.loads(raw.decode())
                except Exception:
                    logger.error("Encoder worker: invalid job payload, skipping")
                    continue

            text = job.get("text") or job.get("query")
            job_id = job.get("job_id")
            if not text or not job_id:
                logger.warning("Encoder worker: missing text or job_id, skipping")
                continue

            if model is None:
                model = load_model()
                if model is None:
                    r.set(
                        f"job:completion:{job_id}",
                        json.dumps({"error": "embedding model unavailable"}),
                        ex=3600,
                    )
                    continue

            vec = model.encode(text, convert_to_tensor=False, normalize_embeddings=True)
            payload = {
                "job_id": job_id,
                "vector": vec.tolist(),
                "query": text,
                "selected_replica_index": job.get("selected_replica_index"),
            }
            r.lpush("job:retriever_in", json.dumps(payload))
            logger.info(f"[{job_id}] Encoded → forwarded to retriever")

        except Exception as exc:
            logger.exception(f"Encoder worker loop error: {exc}")
            await asyncio.sleep(5)


@app.on_event("startup")
async def startup_event():
    load_model()
    asyncio.create_task(encoder_worker_loop())
