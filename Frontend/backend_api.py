# backend_api.py
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse
import time
from prometheus_client import start_http_server, Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import CollectorRegistry
from prometheus_client import multiprocess
import threading

app = FastAPI()

# Prometheus metrics
REQUEST_COUNT = Counter("backend_requests_total", "Total number of backend requests", ["endpoint", "method", "status"])
LATENCY = Histogram("backend_request_latency_seconds", "Latency for backend requests", ["endpoint"])

# Optional: start a /metrics server on port 8001 (separate from FastAPI)
# We'll expose FastAPI /metrics route using prometheus_client.generate_latest
# but also start a separate http server as fallback (not strictly necessary).
def start_prometheus_server():
    start_http_server(8001)  # exposes metrics on http://localhost:8001/

threading.Thread(target=start_prometheus_server, daemon=True).start()

@app.post("/api/query")
async def query_json(payload: dict):
    start = time.time()
    try:
        # --- Placeholder for backend processing ---
        # Example:
        # text = payload.get("text")
        # file_b64 = payload.get("file_b64")   # if you send base64 encoded files
        # image_b64 = payload.get("image_b64")
        # video_b64 = payload.get("video_b64")
        # Here you would run your RAG optimization, RL, model calls, etc.
        # -----------------------------------------------------------
        # For demo, just echo fields and pretend processing took some time:
        time.sleep(0.05)  # simulate processing 50ms
        response = {"status": "ok", "received": payload, "note": "backend processing placeholder"}
        status_code = 200
        return JSONResponse(content=response, status_code=status_code)
    finally:
        elapsed = time.time() - start
        REQUEST_COUNT.labels(endpoint="/api/query", method="POST", status="200").inc()
        LATENCY.labels(endpoint="/api/query").observe(elapsed)

@app.post("/api/query-multipart")
async def query_multipart(
    text: str = Form(None),
    upload_file: UploadFile = File(None),
    image_file: UploadFile = File(None),
    video_file: UploadFile = File(None),
):
    start = time.time()
    try:
        # --- Placeholder for actual backend processing with uploaded files ---
        # You can access file content with await upload_file.read()
        meta = {}
        if text:
            meta["text"] = text
        if upload_file:
            meta["upload_filename"] = upload_file.filename
            # content = await upload_file.read()   # careful with large files
        if image_file:
            meta["image_filename"] = image_file.filename
        if video_file:
            meta["video_filename"] = video_file.filename

        # simulate some work
        time.sleep(0.08)

        response = {"status": "ok", "meta": meta, "note": "multipart backend placeholder"}
        status_code = 200
        return JSONResponse(content=response, status_code=status_code)
    finally:
        elapsed = time.time() - start
        REQUEST_COUNT.labels(endpoint="/api/query-multipart", method="POST", status="200").inc()
        LATENCY.labels(endpoint="/api/query-multipart").observe(elapsed)

@app.get("/metrics")
def metrics():
    # Expose prometheus metrics for Prometheus to scrape
    from prometheus_client import REGISTRY, generate_latest
    data = generate_latest()
    return JSONResponse(content=data.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)
