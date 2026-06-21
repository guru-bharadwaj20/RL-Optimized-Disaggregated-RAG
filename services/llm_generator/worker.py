import os
import time
import json
import requests
import redis
from dotenv import load_dotenv

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://mock_llm:8000")
LLM_API_PATH = os.getenv("LLM_API_PATH", "/generate")
QUEUE_IN = os.getenv("LLM_IN_QUEUE", "job:llm_in")
MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
INITIAL_BACKOFF = float(os.getenv("LLM_INITIAL_BACKOFF", "0.5"))

TARGET_URL = f"{LLM_SERVICE_URL}{LLM_API_PATH}"


def worker_loop():
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    print(f"LLM generator worker listening on queue: {QUEUE_IN}")
    print(f"Forwarding requests to: {TARGET_URL}")

    while True:
        item = r.blpop(QUEUE_IN, timeout=5)
        if not item:
            time.sleep(0.1)
            continue

        _, raw = item
        try:
            job = json.loads(raw)
        except Exception:
            print("LLM worker: invalid JSON payload, skipping")
            continue

        job_id = job.get("job_id")
        prompt = job.get("augmented_prompt") or job.get("prompt")
        selected_replica_index = job.get("selected_replica_index")

        if not job_id or not prompt:
            print(f"LLM worker: missing job_id or prompt, skipping")
            continue

        print(f"[{job_id}] Processing (replica={selected_replica_index}, prompt_len={len(prompt)})")
        start = time.time()

        answer = None
        backoff = INITIAL_BACKOFF
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    TARGET_URL,
                    json={"prompt": prompt, "job_id": job_id},
                    timeout=30,
                )
                resp.raise_for_status()
                out = resp.json()
                answer = out.get("answer") or out.get("response") or str(out)
                print(f"[{job_id}] Got response on attempt {attempt}")
                break
            except requests.exceptions.RequestException as e:
                print(f"[{job_id}] Attempt {attempt}/{MAX_RETRIES} failed: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(backoff)
                    backoff *= 2

        if answer is None:
            answer = "The system was unable to generate a response. Please try again."
            print(f"[{job_id}] All retries exhausted, using fallback answer")

        latency_ms = (time.time() - start) * 1000
        completion_payload = {
            "job_id": job_id,
            "answer": answer,
            "latency_ms": latency_ms,
            "selected_replica_index": selected_replica_index,
        }

        try:
            r.set(f"job:completion:{job_id}", json.dumps(completion_payload), ex=3600)
            print(f"[{job_id}] Completion written in {latency_ms:.0f}ms")
        except Exception as e:
            print(f"[{job_id}] Failed to write completion: {e}")


if __name__ == "__main__":
    try:
        worker_loop()
    except KeyboardInterrupt:
        print("LLM generator worker shutting down")
