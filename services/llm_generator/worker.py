import os
import time
import json
import requests
import redis
from dotenv import load_dotenv

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
LLM_API_PATH = os.getenv("LLM_API_PATH", "/generate")
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", f"http://localhost:8000")

QUEUE_IN = os.getenv("LLM_IN_QUEUE", "job:llm_in")

# Retry and fallback configuration
MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "5"))
INITIAL_BACKOFF = float(os.getenv("LLM_INITIAL_BACKOFF", "0.5"))
ENABLE_FALLBACK = os.getenv("LLM_ENABLE_FALLBACK", "true").lower() in ("1", "true", "yes")


def worker_loop():
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    print(f"LLM generator worker listening on {QUEUE_IN}")

    while True:
        item = r.blpop(QUEUE_IN, timeout=5)
        if not item:
            time.sleep(0.1)
            continue

        _, raw = item
        try:
            job = json.loads(raw)
        except Exception:
            try:
                job = json.loads(raw.decode())
            except Exception:
                print("LLM worker: invalid payload, skipping")
                continue

        job_id = job.get("job_id")
        prompt = job.get("augmented_prompt") or job.get("prompt")
        if not job_id or not prompt:
            print(f"LLM worker: job missing job_id or prompt, skipping. job_id={job_id}, prompt_field={prompt}. Full job: {job}")
            continue

        print(f"[{job_id}] Processing LLM job. Prompt length: {len(prompt) if prompt else 0}")

        start = time.time()

        # Prefer selected_replica passed through the pipeline; otherwise use configured service URL
        selected = job.get("selected_replica")
        primary_target = (selected.rstrip('/') + LLM_API_PATH) if selected else f"{LLM_SERVICE_URL}{LLM_API_PATH}"
        
        print(f"[{job_id}] Calling {primary_target} with prompt of {len(prompt)} chars")

        # Try the primary target with retries, then fall back to local if configured
        answer = None
        backoff = INITIAL_BACKOFF
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.post(primary_target, json={"prompt": prompt, "job_id": job_id}, timeout=10)
                resp.raise_for_status()
                out = resp.json()
                answer = out.get("answer") or out.get("response") or out
                print(f"[{job_id}] Got response: {answer}")
                break
            except requests.exceptions.RequestException as e:
                print(f"[{job_id}] Attempt {attempt}/{MAX_RETRIES} failed to call {primary_target}: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                else:
                    # Last attempt failed — will try fallback
                    last_exc = e

        # Fallback: try local configured URL if primary failed and a different local service is available
        if answer is None and ENABLE_FALLBACK and LLM_SERVICE_URL:
            try:
                fallback_target = f"{LLM_SERVICE_URL}{LLM_API_PATH}"
                print(f"Primary target failed; trying fallback {fallback_target}")
                resp = requests.post(fallback_target, json={"prompt": prompt, "job_id": job_id}, timeout=10)
                resp.raise_for_status()
                out = resp.json()
                answer = out.get("answer") or out.get("response") or out
                print(f"[{job_id}] Fallback response: {answer}")
            except Exception as e:
                answer = f"LLM call failed: {last_exc if 'last_exc' in locals() else e}"
                print(f"[{job_id}] Fallback also failed: {answer}")

        latency_ms = (time.time() - start) * 1000

        completion_key = f"job:completion:{job_id}"
        payload = {"job_id": job_id, "answer": answer, "latency_ms": latency_ms}
        try:
            r.set(completion_key, json.dumps(payload), ex=3600)
            print(f"[{job_id}] ✓ Completion written: {payload}")
        except Exception as e:
            print(f"Failed to write completion key for {job_id}: {e}")


if __name__ == '__main__':
    try:
        worker_loop()
    except KeyboardInterrupt:
        print("LLM generator worker exiting")
