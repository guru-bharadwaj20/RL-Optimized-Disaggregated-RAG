import os
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import logging

import os
import time
import json
import asyncio
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Any, Dict

# We'll lazily import heavy libs inside functions to avoid startup crashes in constrained
# environments. This module runs both an HTTP /encode endpoint and a background
# Redis-backed worker that listens on `job:encoder_in` and writes to `job:retriever_in`.

load_dotenv()
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", 384))
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

app = FastAPI(title="Encoder Service")


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
    # Lazy import model
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(MODEL_NAME, device=EMBEDDING_DEVICE)
        return model
    except Exception:
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
        return VectorResponse(vector=embedding.tolist(), dim=getattr(embedding, "shape", (len(embedding),))[0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate embedding: {exc}")


async def encoder_worker_loop(poll_interval: float = 1.0):
    """Background loop: BRPOP from job:encoder_in, encode text, push to job:retriever_in."""
    redis = get_redis_client()
    if redis is None:
        print("Encoder worker: redis client unavailable, exiting worker loop")
        return

    # Load model once
    model = load_model()
    if model is None:
        print("Encoder worker: model could not be loaded; worker will retry every 10s")
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
                # if raw bytes already, try decode
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

            # Ensure model is loaded
            if model is None:
                model = load_model()
                if model is None:
                    # push an error status to completion and skip
                    redis.set(f"job:completion:{job_id}", json.dumps({"error": "model unavailable"}))
                    continue

            vec = model.encode(text, convert_to_tensor=False)
            # Use the key `query` for the original text so downstream services (retriever)
            # which expect `query` (per the RAGJob model) can validate/parse the payload.
            payload = {"job_id": job_id, "vector": vec.tolist(), "query": text, "selected_replica": job.get("selected_replica")}
            redis.lpush("job:retriever_in", json.dumps(payload))
        except Exception as exc:
            print(f"Encoder worker loop error: {exc}")
            await asyncio.sleep(5)


@app.on_event("startup")
async def startup_event():
    # Start background encoder worker loop
    asyncio.create_task(encoder_worker_loop())

