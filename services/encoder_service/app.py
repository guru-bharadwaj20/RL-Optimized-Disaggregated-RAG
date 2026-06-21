import os
import time
import json
import asyncio
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", 384))
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

app = FastAPI(title="Encoder Service")

_model = None


class QueryRequest(BaseModel):
    text: str


class VectorResponse(BaseModel):
    vector: list[float]
    dim: int = VECTOR_DIMENSION


def get_redis_client():
    try:
        import redis
        return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)
    except Exception:
        return None


def load_model():
    global _model
    if _model is not None:
        return _model
    try:
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME, device=EMBEDDING_DEVICE)
        print(f"Model loaded successfully on {EMBEDDING_DEVICE}")
        return _model
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None


@app.get("/health")
async def health_check():
    m = load_model()
    if m is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/encode", response_model=VectorResponse)
async def encode_query(request: QueryRequest):
    m = load_model()
    if m is None:
        raise HTTPException(status_code=503, detail="Model unavailable")
    if not request.text or request.text.strip() == "":
        raise HTTPException(status_code=400, detail="Input text cannot be empty")
    try:
        embedding = m.encode(request.text, convert_to_tensor=False)
        dim = getattr(embedding, "shape", (len(embedding),))[0]
        return VectorResponse(vector=embedding.tolist(), dim=dim)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate embedding: {exc}")


async def encoder_worker_loop(poll_interval: float = 1.0):
    """Background loop: BRPOP from job:encoder_in, encode text, push to job:retriever_in."""
    redis = get_redis_client()
    if redis is None:
        print("Encoder worker: redis client unavailable, exiting worker loop")
        return

    model = load_model()
    if model is None:
        print("Encoder worker: model could not be loaded at startup; will retry per job")

    while True:
        try:
            item = redis.brpop("job:encoder_in", timeout=5)
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
                    print("Encoder worker: invalid job payload, skipping")
                    continue

            text = job.get("text") or job.get("query")
            job_id = job.get("job_id")
            if not text or not job_id:
                print("Encoder worker: job missing text or job_id, skipping")
                continue

            if model is None:
                model = load_model()
                if model is None:
                    redis.set(
                        f"job:completion:{job_id}",
                        json.dumps({"error": "embedding model unavailable"}),
                        ex=3600,
                    )
                    continue

            vec = model.encode(text, convert_to_tensor=False)
            payload = {
                "job_id": job_id,
                "vector": vec.tolist(),
                "query": text,
                # Correctly forward the replica index chosen by the API Gateway
                "selected_replica_index": job.get("selected_replica_index"),
            }
            redis.lpush("job:retriever_in", json.dumps(payload))
            print(f"[{job_id}] Encoded and forwarded to retriever")
        except Exception as exc:
            print(f"Encoder worker loop error: {exc}")
            await asyncio.sleep(5)


@app.on_event("startup")
async def startup_event():
    load_model()  # Pre-load model so first request is fast
    asyncio.create_task(encoder_worker_loop())
