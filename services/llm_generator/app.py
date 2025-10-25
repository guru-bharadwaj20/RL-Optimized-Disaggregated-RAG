import os
import json
import time
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
# LLM service details (from .env/docker-compose)
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://mock_llm:8000")
LLM_MOCK_FALLBACK = os.getenv("LLM_MOCK_FALLBACK", "false").lower() in ("1", "true", "yes")

# --- FastAPI Setup ---
app = FastAPI(title="LLM Generator Service", version="1.0")

# --- Pydantic Data Models ---
class GeneratorRequest(BaseModel):
    """Input model received from the Redis queue/API Gateway."""
    prompt: str
    job_id: str
    
class GeneratorResponse(BaseModel):
    """Output model for the final answer."""
    job_id: str
    answer: str
    latency_ms: float

# --- Health Check ---
@app.get("/health")
def health_check():
    """Checks service status and connectivity to the LLM service."""
    try:
        response = requests.post(
            f"{LLM_SERVICE_URL}/generate",
            json={"query": "Test"},
            timeout=5
        )
        if response.status_code == 200:
            return {"status": "ok", "llm_service": LLM_SERVICE_URL}
        else:
            return {"status": "degraded", "detail": f"LLM service returned status {response.status_code}"}
    except requests.exceptions.RequestException:
        raise HTTPException(status_code=503, detail="Cannot reach LLM service.")

# --- Main Inference Endpoint ---
@app.post("/generate", response_model=GeneratorResponse)
async def generate_response(request: GeneratorRequest):
    """
    Receives an augmented prompt, calls the LLM service, and returns the final answer 
    with measured latency (CRITICAL for P99 measurement).
    """
    start_time = time.time()
    
    # 1. Prepare the payload for the LLM service
    payload = {
        "query": request.prompt
    }
    
    try:
        # 2. Call the LLM service
        response = requests.post(f"{LLM_SERVICE_URL}/generate", json=payload, timeout=60)
        response.raise_for_status()

        # 3. Parse the generated text
        data = response.json()
        generated_text = data.get("answer", data.get("response", "Error: No response text found.")).strip()

    except requests.exceptions.RequestException as e:
        print(f"LLM Service Request Error for Job {request.job_id}: {e}")
        if LLM_MOCK_FALLBACK:
            mock_text = f"[MOCK LLM] Echo: {request.prompt[:300]}"
            generated_text = mock_text
        else:
            raise HTTPException(status_code=500, detail=f"Failed to get response from LLM service: {e}")

    # 4. Measure Latency (The key measurement for the RL scaling demo)
    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000

    # 5. Return the final result
    return GeneratorResponse(
        job_id=request.job_id,
        answer=generated_text,
        latency_ms=latency_ms
    )
