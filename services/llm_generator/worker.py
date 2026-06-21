import os
import time
import json
import logging
import requests
import redis
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("llm_worker")

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
QUEUE_IN = os.getenv("LLM_IN_QUEUE", "job:llm_in")
MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
INITIAL_BACKOFF = float(os.getenv("LLM_INITIAL_BACKOFF", "0.5"))

# Ollama settings (primary LLM backend)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "")          # empty = disabled
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))

# Mock LLM fallback
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://mock_llm:8000")
LLM_API_PATH = os.getenv("LLM_API_PATH", "/generate")
MOCK_TARGET_URL = f"{LLM_SERVICE_URL}{LLM_API_PATH}"

USE_MOCK_FALLBACK = os.getenv("USE_MOCK_FALLBACK", "true").lower() in ("1", "true", "yes")


def call_ollama(prompt: str) -> str:
    """Call Ollama /api/generate endpoint."""
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"
    resp = requests.post(
        url,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 512,
                "stop": ["USER QUESTION:", "Context:"],
            },
        },
        timeout=OLLAMA_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    return (data.get("response") or "").strip()


def call_mock_llm(prompt: str, job_id: str) -> str:
    """Call mock LLM service with retry."""
    backoff = INITIAL_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                MOCK_TARGET_URL,
                json={"prompt": prompt, "job_id": job_id},
                timeout=30,
            )
            resp.raise_for_status()
            out = resp.json()
            return out.get("answer") or out.get("response") or str(out)
        except requests.exceptions.RequestException as e:
            logger.warning(f"Mock LLM attempt {attempt}/{MAX_RETRIES}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(backoff)
                backoff *= 2
    return None


def generate_answer(prompt: str, job_id: str) -> tuple[str, str]:
    """
    Try Ollama first (if configured), fall back to mock LLM.
    Returns (answer, backend_used).
    """
    # --- Ollama (primary) ---
    if OLLAMA_BASE_URL:
        try:
            logger.info(f"[{job_id}] Calling Ollama ({OLLAMA_MODEL}) at {OLLAMA_BASE_URL}")
            answer = call_ollama(prompt)
            if answer:
                return answer, f"ollama:{OLLAMA_MODEL}"
            logger.warning(f"[{job_id}] Ollama returned empty response")
        except Exception as e:
            logger.warning(f"[{job_id}] Ollama failed: {e}. Falling back to mock LLM.")

    # --- Mock LLM (fallback / default when Ollama not configured) ---
    if USE_MOCK_FALLBACK or not OLLAMA_BASE_URL:
        answer = call_mock_llm(prompt, job_id)
        if answer:
            return answer, "mock_llm"

    return "The system was unable to generate a response. Please try again.", "none"


def worker_loop():
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    mode = f"Ollama({OLLAMA_MODEL})" if OLLAMA_BASE_URL else "mock_llm"
    logger.info(f"LLM generator worker listening on: {QUEUE_IN}")
    logger.info(f"LLM backend: {mode}")

    while True:
        item = r.blpop(QUEUE_IN, timeout=5)
        if not item:
            time.sleep(0.1)
            continue

        _, raw = item
        try:
            job = json.loads(raw)
        except Exception:
            logger.error("Invalid JSON payload, skipping")
            continue

        job_id = job.get("job_id")
        prompt = job.get("augmented_prompt") or job.get("prompt")
        selected_replica_index = job.get("selected_replica_index")
        retrieved_chunks = job.get("retrieved_chunks", [])
        retrieval_latency_ms = job.get("retrieval_latency_ms", 0.0)

        if not job_id or not prompt:
            logger.warning(f"Job missing job_id or prompt, skipping: {job_id}")
            continue

        logger.info(
            f"[{job_id}] Processing (replica={selected_replica_index}, "
            f"prompt_len={len(prompt)}, chunks={len(retrieved_chunks)})"
        )
        start = time.time()

        answer, backend = generate_answer(prompt, job_id)
        llm_latency_ms = (time.time() - start) * 1000
        total_latency_ms = llm_latency_ms + retrieval_latency_ms

        logger.info(f"[{job_id}] Done via {backend} in {llm_latency_ms:.0f}ms (LLM)")

        completion_payload = {
            "job_id": job_id,
            "answer": answer,
            "latency_ms": total_latency_ms,
            "llm_latency_ms": llm_latency_ms,
            "retrieval_latency_ms": retrieval_latency_ms,
            "selected_replica_index": selected_replica_index,
            "retrieved_chunks": retrieved_chunks,
            "llm_backend": backend,
        }

        try:
            r.set(f"job:completion:{job_id}", json.dumps(completion_payload), ex=3600)
            logger.info(f"[{job_id}] Completion written (total {total_latency_ms:.0f}ms)")
        except Exception as e:
            logger.error(f"[{job_id}] Failed to write completion: {e}")


if __name__ == "__main__":
    try:
        worker_loop()
    except KeyboardInterrupt:
        logger.info("LLM generator worker shutting down")
