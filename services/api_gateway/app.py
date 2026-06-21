import os
import time
import json
import asyncio
import uuid
import hashlib
import logging
from typing import Optional, Tuple

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from redis import Redis
from dotenv import load_dotenv
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("api_gateway")

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
ENCODER_SERVICE_URL = os.getenv("ENCODER_SERVICE_URL", "http://encoder_service:8000")
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", 384))

# Cache settings
EXACT_CACHE_ENABLED = os.getenv("EXACT_CACHE_ENABLED", "true").lower() in ("1", "true", "yes")
EXACT_CACHE_TTL = int(os.getenv("EXACT_CACHE_TTL", 3600))
SEMANTIC_CACHE_ENABLED = os.getenv("SEMANTIC_CACHE_ENABLED", "true").lower() in ("1", "true", "yes")
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.95"))
SEMANTIC_CACHE_COLLECTION = "semantic_cache"

LLM_REPLICAS = ["replica-1", "replica-2", "replica-3"]
REDIS_LOAD_KEY_PREFIX = "llm_load_"
REDIS_LATENCY_KEY_PREFIX = "llm_latency_ms_"

# ── Redis ──────────────────────────────────────────────────────────────────────
redis_client: Optional[Redis] = None
for _attempt in range(30):
    try:
        redis_client = Redis(host=REDIS_HOST, port=6379, decode_responses=True, socket_connect_timeout=2)
        redis_client.ping()
        logger.info(f"Connected to Redis on attempt {_attempt + 1}")
        break
    except Exception as e:
        logger.warning(f"Redis attempt {_attempt + 1}/30: {e}")
        if _attempt < 29:
            time.sleep(1)
        else:
            redis_client = None

# ── Qdrant (for semantic cache) ────────────────────────────────────────────────
_qdrant_client = None
try:
    from qdrant_client import QdrantClient, models as qm
    _qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    _qdrant_client.get_collections()
    logger.info(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
except Exception as e:
    logger.warning(f"Qdrant unavailable ({e}) — semantic cache disabled")
    _qdrant_client = None

_rr_counter = 0
_semantic_cache_ready = False

app = FastAPI(title="RL-Optimized RAG API Gateway", version="2.0")


# ── Prometheus Metrics ─────────────────────────────────────────────────────────
query_counter = Counter("api_gateway_queries_total", "Total queries", ["status"])
query_latency = Histogram(
    "api_gateway_query_latency_seconds", "Query latency",
    buckets=[0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60],
)
active_jobs_gauge = Gauge("api_gateway_active_jobs", "Active pipeline jobs")
replica_load_gauge = Gauge("api_gateway_replica_load", "Load per replica", ["replica_index"])

cache_hits_counter = Counter("cache_hits_total", "Exact cache hits")
cache_misses_counter = Counter("cache_misses_total", "Exact cache misses")
semantic_cache_hits_counter = Counter("semantic_cache_hits_total", "Semantic cache hits")
semantic_cache_misses_counter = Counter("semantic_cache_misses_total", "Semantic cache misses")

worker_selections_counter = Counter("worker_selection_total", "Worker selections", ["replica_index"])
worker_avg_latency_gauge = Gauge("worker_avg_latency_ms", "Worker EMA latency", ["replica_index"])
ingestion_docs_counter = Counter("ingestion_documents_total", "Documents ingested")
ingestion_chunks_counter = Counter("ingestion_chunks_total", "Chunks ingested")


# ── Pydantic Models ────────────────────────────────────────────────────────────

class QueryInput(BaseModel):
    query: str


class IngestRequest(BaseModel):
    text: str
    source: str = "api_upload"


class FinalResponse(BaseModel):
    job_id: str
    answer: str
    latency_ms: float
    selected_replica: str
    cache_status: str = "miss"
    retrieved_chunks: list[dict] = []
    llm_backend: str = "unknown"


class IngestResponse(BaseModel):
    chunks_added: int
    source: str
    processing_time_ms: float


# ── Semantic Cache ─────────────────────────────────────────────────────────────

def _ensure_semantic_cache_collection() -> bool:
    global _semantic_cache_ready
    if _semantic_cache_ready:
        return True
    if _qdrant_client is None:
        return False
    try:
        _qdrant_client.get_collection(SEMANTIC_CACHE_COLLECTION)
        _semantic_cache_ready = True
        return True
    except Exception:
        try:
            _qdrant_client.create_collection(
                collection_name=SEMANTIC_CACHE_COLLECTION,
                vectors_config=qm.VectorParams(size=VECTOR_DIMENSION, distance=qm.Distance.COSINE),
            )
            _semantic_cache_ready = True
            logger.info(f"Created semantic cache collection '{SEMANTIC_CACHE_COLLECTION}'")
            return True
        except Exception as e:
            logger.warning(f"Could not create semantic cache collection: {e}")
            return False


def _get_query_embedding(question: str) -> Optional[list]:
    try:
        resp = requests.post(
            f"{ENCODER_SERVICE_URL}/encode",
            json={"text": question},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()["vector"]
    except Exception as e:
        logger.warning(f"Encoder service call failed: {e}")
        return None


def _check_semantic_cache(embedding: list) -> Optional[dict]:
    if not _qdrant_client or not embedding:
        return None
    try:
        results = _qdrant_client.search(
            collection_name=SEMANTIC_CACHE_COLLECTION,
            query_vector=embedding,
            limit=1,
            score_threshold=SEMANTIC_CACHE_THRESHOLD,
            with_payload=True,
        )
        if results:
            return results[0].payload
    except Exception as e:
        logger.warning(f"Semantic cache lookup failed: {e}")
    return None


def _store_semantic_cache(embedding: list, payload: dict) -> None:
    if not _qdrant_client or not embedding:
        return
    try:
        raw_id = int(uuid.uuid4().int) % (2**63 - 1)
        _qdrant_client.upsert(
            collection_name=SEMANTIC_CACHE_COLLECTION,
            points=[qm.PointStruct(id=raw_id, vector=embedding, payload=payload)],
            wait=False,
        )
    except Exception as e:
        logger.warning(f"Semantic cache store failed: {e}")


# ── Cache Helpers ──────────────────────────────────────────────────────────────

def _normalize(q: str) -> str:
    return " ".join(q.lower().strip().split())


def _exact_cache_key(question: str) -> str:
    return f"cache:exact:{hashlib.md5(_normalize(question).encode()).hexdigest()}"


# ── Adaptive Load Balancer ─────────────────────────────────────────────────────

def _get_best_replica() -> Tuple[str, int]:
    """
    Adaptive LB: score = active_connections + (ema_latency_ms / 1000).
    Falls back to round-robin when Redis is unavailable.
    """
    best_replica, best_index, best_score = LLM_REPLICAS[0], 0, float("inf")

    for idx, replica in enumerate(LLM_REPLICAS):
        try:
            load = int(redis_client.get(f"{REDIS_LOAD_KEY_PREFIX}{idx}") or 0) if redis_client else 0
            latency = float(redis_client.get(f"{REDIS_LATENCY_KEY_PREFIX}{idx}") or 200) if redis_client else 200
        except Exception:
            load, latency = 0, 200

        score = load + latency / 1000.0
        if score < best_score:
            best_score, best_replica, best_index = score, replica, idx

    return best_replica, best_index


def _update_replica_latency(idx: int, latency_ms: float) -> None:
    if not redis_client:
        return
    try:
        key = f"{REDIS_LATENCY_KEY_PREFIX}{idx}"
        prev = float(redis_client.get(key) or latency_ms)
        ema = 0.2 * latency_ms + 0.8 * prev
        redis_client.set(key, ema, ex=3600)
        worker_avg_latency_gauge.labels(replica_index=str(idx)).set(ema)
    except Exception:
        pass


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    redis_ok, qdrant_ok = False, False
    try:
        if redis_client:
            redis_client.ping()
            redis_ok = True
    except Exception:
        pass
    try:
        if _qdrant_client:
            _qdrant_client.get_collections()
            qdrant_ok = True
    except Exception:
        pass
    return {
        "status": "ok",
        "redis": "connected" if redis_ok else "disconnected",
        "qdrant": "connected" if qdrant_ok else "disconnected",
        "exact_cache": "enabled" if EXACT_CACHE_ENABLED else "disabled",
        "semantic_cache": "enabled" if (SEMANTIC_CACHE_ENABLED and qdrant_ok) else "disabled",
        "encoder_url": ENCODER_SERVICE_URL,
    }


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest):
    """Proxy document ingestion to the encoder service."""
    t0 = time.time()
    try:
        resp = requests.post(
            f"{ENCODER_SERVICE_URL}/ingest",
            json={"text": request.text, "source": request.source},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        ingestion_docs_counter.inc()
        ingestion_chunks_counter.inc(data.get("chunks_added", 0))
        return IngestResponse(
            chunks_added=data["chunks_added"],
            source=data["source"],
            processing_time_ms=(time.time() - t0) * 1000,
        )
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Encoder service error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


@app.post("/query", response_model=FinalResponse)
async def process_rag_query(input: QueryInput):
    """
    Full RAG pipeline with:
      1. Exact Redis cache (hash-based, O(1))
      2. Semantic cache (Qdrant cosine search)
      3. Adaptive least-loaded + lowest-latency worker selection
      4. Redis queue pipeline: encode → retrieve → generate
    """
    global redis_client
    start_time = time.time()
    active_jobs_gauge.inc()
    query_embedding: Optional[list] = None

    try:
        # Reconnect Redis if needed
        if not redis_client:
            try:
                redis_client = Redis(host=REDIS_HOST, port=6379, decode_responses=True, socket_connect_timeout=2)
                redis_client.ping()
                logger.info("Reconnected to Redis")
            except Exception:
                redis_client = None

        job_id = str(uuid.uuid4())

        # ── Step 1: Exact Cache ───────────────────────────────────────────────
        if EXACT_CACHE_ENABLED and redis_client:
            ck = _exact_cache_key(input.query)
            raw_cached = redis_client.get(ck)
            if raw_cached:
                cache_hits_counter.inc()
                query_counter.labels(status="cache_hit").inc()
                cached = json.loads(raw_cached)
                cached.update({"job_id": job_id, "cache_status": "hit"})
                return FinalResponse(**cached)
            cache_misses_counter.inc()

        # ── Step 2: Semantic Cache ────────────────────────────────────────────
        if SEMANTIC_CACHE_ENABLED and _qdrant_client and _ensure_semantic_cache_collection():
            query_embedding = _get_query_embedding(input.query)
            if query_embedding:
                sem_hit = _check_semantic_cache(query_embedding)
                if sem_hit:
                    semantic_cache_hits_counter.inc()
                    query_counter.labels(status="semantic_hit").inc()
                    if EXACT_CACHE_ENABLED and redis_client:
                        redis_client.setex(
                            _exact_cache_key(input.query), EXACT_CACHE_TTL, json.dumps(sem_hit)
                        )
                    sem_hit.update({"job_id": job_id, "cache_status": "semantic_hit"})
                    return FinalResponse(**sem_hit)
                semantic_cache_misses_counter.inc()

        # ── Step 3: Adaptive Worker Selection ─────────────────────────────────
        if not redis_client:
            global _rr_counter
            replica_idx = _rr_counter % len(LLM_REPLICAS)
            _rr_counter += 1
            load_key = None
        else:
            _, replica_idx = _get_best_replica()
            load_key = f"{REDIS_LOAD_KEY_PREFIX}{replica_idx}"
            try:
                redis_client.incr(load_key)
                replica_load_gauge.labels(replica_index=str(replica_idx)).set(
                    int(redis_client.get(load_key) or 0)
                )
                worker_selections_counter.labels(replica_index=str(replica_idx)).inc()
            except Exception:
                load_key = None

        # ── Step 4: Publish to Pipeline ───────────────────────────────────────
        job_payload = {
            "job_id": job_id,
            "text": input.query,
            "selected_replica_index": replica_idx,
        }
        try:
            if not redis_client:
                raise RuntimeError("Redis unavailable")
            redis_client.lpush("job:encoder_in", json.dumps(job_payload))
        except Exception as e:
            if load_key and redis_client:
                try:
                    redis_client.decr(load_key)
                except Exception:
                    pass
            query_counter.labels(status="error").inc()
            raise HTTPException(status_code=503, detail=f"Failed to queue job: {e}")

        # ── Step 5: Poll for Completion ───────────────────────────────────────
        completion_key = f"job:completion:{job_id}"
        result = None
        timeout, poll_interval, elapsed = 60.0, 0.25, 0.0
        try:
            while elapsed < timeout:
                raw = redis_client.get(completion_key)
                if raw:
                    try:
                        result = json.loads(raw)
                    except Exception:
                        result = {"answer": str(raw)}
                    break
                await asyncio.sleep(poll_interval)
                elapsed = time.time() - start_time
        finally:
            if load_key and redis_client:
                try:
                    new_val = redis_client.decr(load_key)
                    if new_val is not None and int(new_val) < 0:
                        redis_client.set(load_key, 0)
                    replica_load_gauge.labels(replica_index=str(replica_idx)).set(max(0, int(new_val or 0)))
                except Exception:
                    pass

        if not result:
            query_counter.labels(status="timeout").inc()
            raise HTTPException(status_code=504, detail="Job timed out — no response within 60s")

        end_time = time.time()
        e2e_ms = (end_time - start_time) * 1000

        query_counter.labels(status="success").inc()
        query_latency.observe(end_time - start_time)
        _update_replica_latency(replica_idx, result.get("llm_latency_ms", e2e_ms))

        # ── Step 6: Populate Caches ───────────────────────────────────────────
        cache_payload = {
            "answer": result.get("answer", ""),
            "latency_ms": e2e_ms,
            "selected_replica": f"replica-{replica_idx + 1}",
            "retrieved_chunks": result.get("retrieved_chunks", []),
            "llm_backend": result.get("llm_backend", "unknown"),
            "cache_status": "miss",
        }

        if EXACT_CACHE_ENABLED and redis_client:
            try:
                redis_client.setex(_exact_cache_key(input.query), EXACT_CACHE_TTL, json.dumps(cache_payload))
            except Exception as e:
                logger.warning(f"Exact cache write failed: {e}")

        if SEMANTIC_CACHE_ENABLED and _qdrant_client and query_embedding:
            _store_semantic_cache(query_embedding, cache_payload)

        return FinalResponse(job_id=job_id, **cache_payload)

    finally:
        active_jobs_gauge.dec()
