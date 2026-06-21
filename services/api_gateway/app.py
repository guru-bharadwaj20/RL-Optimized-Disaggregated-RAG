import os
import time
import json
import asyncio
import uuid
from typing import Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from redis import Redis
from dotenv import load_dotenv
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "redis")

# LLM replica identifiers used for load-tracking only (not actual service URLs)
LLM_REPLICAS = ["replica-1", "replica-2", "replica-3"]
REDIS_LOAD_KEY_PREFIX = "llm_load_"

# --- Redis client with retry ---
redis_client = None
for attempt in range(30):
    try:
        redis_client = Redis(host=REDIS_HOST, port=6379, decode_responses=True, socket_connect_timeout=2)
        redis_client.ping()
        print(f"Connected to Redis on attempt {attempt + 1}")
        break
    except Exception as e:
        print(f"Attempt {attempt + 1}/30: Could not connect to Redis: {e}")
        if attempt < 29:
            time.sleep(1)
        else:
            print("Warning: Could not connect to Redis after 30 attempts")
            redis_client = None

_rr_counter = 0

app = FastAPI(title="RL-Optimized RAG API Gateway", version="1.0")

# --- Prometheus Metrics ---
query_counter = Counter("api_gateway_queries_total", "Total queries processed", ["status"])
query_latency = Histogram(
    "api_gateway_query_latency_seconds",
    "Query latency in seconds",
    buckets=[0.1, 0.5, 1, 5, 10, 30, 60],
)
active_jobs = Gauge("api_gateway_active_jobs", "Number of active jobs")
replica_load = Gauge("api_gateway_replica_load", "Load on each replica", ["replica_index"])


class QueryInput(BaseModel):
    query: str


class FinalResponse(BaseModel):
    job_id: str
    answer: str
    latency_ms: float
    selected_replica: str


def get_least_loaded_replica() -> Tuple[str, int]:
    """Least-connections heuristic: pick the replica with the fewest active jobs."""
    min_load = float("inf")
    best_replica = LLM_REPLICAS[0]
    best_index = 0

    for idx, replica in enumerate(LLM_REPLICAS):
        load_key = f"{REDIS_LOAD_KEY_PREFIX}{idx}"
        try:
            current_load = int(redis_client.get(load_key) or 0) if redis_client else 0
        except Exception:
            current_load = 0
        if current_load < min_load:
            min_load = current_load
            best_replica = replica
            best_index = idx

    return best_replica, best_index


@app.get("/health")
async def health_check():
    """Service liveness probe."""
    try:
        if redis_client:
            redis_client.ping()
        return {"status": "ok", "redis": "connected" if redis_client else "disconnected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Redis unavailable: {e}")


@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/query", response_model=FinalResponse)
async def process_rag_query(input: QueryInput):
    """Orchestrate the full RAG pipeline: Encode -> Retrieve -> Generate."""
    global redis_client
    start_time = time.time()
    active_jobs.inc()

    try:
        # Reconnect Redis if needed
        if not redis_client:
            try:
                redis_client = Redis(host=REDIS_HOST, port=6379, decode_responses=True, socket_connect_timeout=2)
                redis_client.ping()
                print("Reconnected to Redis")
            except Exception as e:
                print(f"Could not reconnect to Redis: {e}")
                redis_client = None

        job_id = str(uuid.uuid4())

        # Pick least-loaded replica and increment its counter
        if not redis_client:
            global _rr_counter
            replica_idx = _rr_counter % len(LLM_REPLICAS)
            _rr_counter += 1
            replica_name = LLM_REPLICAS[replica_idx]
            load_key = None
        else:
            replica_name, replica_idx = get_least_loaded_replica()
            load_key = f"{REDIS_LOAD_KEY_PREFIX}{replica_idx}"
            try:
                redis_client.incr(load_key)
                replica_load.labels(replica_index=str(replica_idx)).set(
                    int(redis_client.get(load_key) or 0)
                )
            except Exception:
                load_key = None

        # Publish job to encoder queue
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

        # Poll for completion (written by LLM worker)
        completion_key = f"job:completion:{job_id}"
        result = None
        timeout = 60.0
        poll_interval = 0.25
        elapsed = 0.0
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
                    replica_load.labels(replica_index=str(replica_idx)).set(
                        max(0, int(new_val or 0))
                    )
                except Exception:
                    pass

        end_time = time.time()
        e2e_latency_ms = (end_time - start_time) * 1000

        if not result:
            query_counter.labels(status="timeout").inc()
            raise HTTPException(status_code=504, detail="Job timed out — no response within 60s")

        query_counter.labels(status="success").inc()
        query_latency.observe(end_time - start_time)

        return FinalResponse(
            job_id=job_id,
            answer=result.get("answer", ""),
            latency_ms=e2e_latency_ms,
            selected_replica=f"replica-{replica_idx + 1}",
        )

    finally:
        active_jobs.dec()
