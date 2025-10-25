import os
import time
import json
import asyncio
import uuid
import requests
from typing import List, Dict, Any, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from redis import Redis
from dotenv import load_dotenv
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# --- Configuration & Initialization ---
load_dotenv()

# Service URLs (Uses Docker service names from .env)
ENCODER_URL = f"http://{os.getenv('ENCODER_HOST', 'encoder_service')}:{os.getenv('API_PORT', '8000')}"
REDIS_HOST = os.getenv("REDIS_HOST", "redis")

# List of LLM Generator Replicas (Manual list for hackathon scaling demo)
# In a real system, Kubernetes service discovery would provide this list.
# These are for LOAD BALANCING ONLY - actual LLM requests go to mock_llm
LLM_REPLICAS = [
    "replica-1",  # Just identifiers for load tracking
    "replica-2",  # Not actual URLs - don't call these
    "replica-3",  # LLM service URL is configured separately
]
# Redis keys for tracking load
REDIS_LOAD_KEY_PREFIX = "llm_load_" 

# --- Clients ---
# Initialize Redis client (used for Heuristic load tracking)
redis_client = None
for attempt in range(30):
    try:
        redis_client = Redis(host=REDIS_HOST, port=6379, decode_responses=True, socket_connect_timeout=2)
        redis_client.ping()
        print(f"✓ Connected to Redis on attempt {attempt + 1}")
        break
    except Exception as e:
        print(f"Attempt {attempt + 1}/30: Could not connect to Redis: {e}")
        if attempt < 29:
            time.sleep(1)
        else:
            print("Warning: Could not connect to Redis after 30 attempts, will retry at runtime")
            redis_client = None

# Simple in-process round-robin counter used when Redis is unavailable
_rr_counter = 0

# --- FastAPI Setup & Models ---
app = FastAPI(title="API Gateway (RL Heuristic)", version="1.0")

# --- Prometheus Metrics ---
query_counter = Counter('api_gateway_queries_total', 'Total queries processed', ['status'])
query_latency = Histogram('api_gateway_query_latency_seconds', 'Query latency in seconds', buckets=[0.1, 0.5, 1, 5, 10, 30, 60])
active_jobs = Gauge('api_gateway_active_jobs', 'Number of active jobs')
replica_load = Gauge('api_gateway_replica_load', 'Load on each replica', ['replica_index'])

class QueryInput(BaseModel):
    """External user query input."""
    query: str

class FinalResponse(BaseModel):
    """Final, aggregated response to the user."""
    job_id: str
    answer: str
    latency_ms: float
    selected_replica: str

# --- Load Balancing Heuristic (CRITICAL LOGIC) ---
def get_least_loaded_replica() -> Tuple[str, int]:
    """
    Simulates the RL Agent's decision: finds the replica with the fewest active jobs (Least Connections).
    In a full RL system, this would be replaced by the agent's complex policy network.
    """
    min_load = float('inf')
    best_replica_url = LLM_REPLICAS[0]
    best_index = 0

    for idx, replica_url in enumerate(LLM_REPLICAS):
        # Use index-based keys to avoid special characters in Redis keys
        load_key = f"{REDIS_LOAD_KEY_PREFIX}{idx}"
        try:
            current_load = int(redis_client.get(load_key) or 0) if redis_client else 0
        except Exception:
            current_load = 0

        # Choose the replica with the minimum load
        if current_load < min_load:
            min_load = current_load
            best_replica_url = replica_url
            best_index = idx

    return best_replica_url, best_index

# --- Prometheus Metrics Endpoint ---
@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

# --- Main Orchestration Endpoint ---
@app.post("/query", response_model=FinalResponse)
async def process_rag_query(input: QueryInput):
    """Orchestrates the full RAG pipeline: Encode -> Retrieve (Simulated) -> Generate."""
    global redis_client
    start_time = time.time()
    active_jobs.inc()
    
    try:
        # Ensure Redis is connected (retry if needed)
        if not redis_client:
            try:
                redis_client = Redis(host=REDIS_HOST, port=6379, decode_responses=True, socket_connect_timeout=2)
                redis_client.ping()
                print("✓ Reconnected to Redis")
            except Exception as e:
                print(f"Could not reconnect to Redis: {e}")
                redis_client = None
        
        # Generate a unique job id server-side
        job_id = str(uuid.uuid4())

        # Choose LLM replica and increment its load counter (before publishing the job)
        # 2. HEURISTIC & LOAD TRACKING: pick replica and publish job to encoder queue
        if not redis_client:
            global _rr_counter
            replica_idx = _rr_counter % len(LLM_REPLICAS)
            _rr_counter += 1
            replica_url = LLM_REPLICAS[replica_idx]
            load_key = None
        else:
            replica_url, replica_idx = get_least_loaded_replica()
            load_key = f"{REDIS_LOAD_KEY_PREFIX}{replica_idx}"
            try:
                redis_client.incr(load_key)
                replica_load.labels(replica_index=str(replica_idx)).set(int(redis_client.get(load_key) or 0))
            except Exception:
                load_key = None

        # Publish the job to the encoder queue and include selected replica metadata
        job_payload = {"job_id": job_id, "text": input.query, "selected_replica_index": replica_idx}
        try:
            if not redis_client:
                raise RuntimeError("Redis not available to publish job")
            redis_client.lpush("job:encoder_in", json.dumps(job_payload))
        except Exception as e:
            # If we failed to publish, decrement and raise
            if load_key and redis_client:
                try:
                    redis_client.decr(load_key)
                except Exception:
                    pass
            query_counter.labels(status='error').inc()
            raise HTTPException(status_code=503, detail=f"Failed to publish job to encoder queue: {e}")

        # 3. Poll for completion key written by the LLM generator worker
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
            # Decrement load counter regardless of success/timeout
            if load_key and redis_client:
                try:
                    new_val = redis_client.decr(load_key)
                    if new_val is not None and int(new_val) < 0:
                        redis_client.set(load_key, 0)
                    replica_load.labels(replica_index=str(replica_idx)).set(max(0, int(new_val or 0)))
                except Exception:
                    pass

        end_time = time.time()
        e2e_latency = (end_time - start_time) * 1000
        latency_seconds = (end_time - start_time)

        if not result:
            query_counter.labels(status='timeout').inc()
            raise HTTPException(status_code=504, detail="Job timed out before completion")

        query_counter.labels(status='success').inc()
        query_latency.observe(latency_seconds)

        return FinalResponse(job_id=job_id, answer=result.get("answer", ""), latency_ms=e2e_latency, selected_replica=f"replica-{replica_idx + 1}")
    
    finally:
        active_jobs.dec()