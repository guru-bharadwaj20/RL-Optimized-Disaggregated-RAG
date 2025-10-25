import os
import json
import time
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
# Ollama service details (from .env/docker-compose)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "ollama_server")
OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral:7b-instruct")

# The URL for the Ollama API endpoint
OLLAMA_API_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate"

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
    """Checks service status and connectivity to the Ollama model."""
    try:
        # Check if Ollama is reachable by sending a dummy request (optional but robust)
        response = requests.post(
            OLLAMA_API_URL,
            json={"model": LLM_MODEL, "prompt": "Test", "stream": False, "options": {"num_predict": 1}},
            timeout=5
        )
        if response.status_code == 200:
            return {"status": "ok", "model": LLM_MODEL, "ollama_status": "ready"}
        else:
            return {"status": "degraded", "detail": f"Ollama returned status {response.status_code}"}
    except requests.exceptions.RequestException:
        raise HTTPException(status_code=503, detail="Cannot reach Ollama server.")

# --- Main Inference Endpoint ---
@app.post("/generate", response_model=GeneratorResponse)
async def generate_response(request: GeneratorRequest):
    """
    Receives an augmented prompt, calls Ollama, and returns the final answer 
    with measured latency (CRITICAL for P99 measurement).
    """
    start_time = time.time()
    
    # 1. Prepare the payload for Ollama
    payload = {
        "model": LLM_MODEL,
        "prompt": request.prompt,
        "stream": False,  # We want the full response at once
        "options": {
            "temperature": 0.1,
            # Limit the output length to save compute time in the hackathon
            "num_predict": 256 
        }
    }
    
    try:
        # 2. Call the local Ollama API
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=60)
        response.raise_for_status() # Raise exception for 4xx or 5xx errors

        # 3. Parse the generated text
        data = response.json()
        
        # Ollama response structure usually contains a 'response' field
        generated_text = data.get("response", "Error: No response text found.").strip()

    except requests.exceptions.RequestException as e:
        print(f"Ollama Request Error for Job {request.job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get response from LLM: {e}")

    # 4. Measure Latency (The key measurement for the RL scaling demo)
    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000

    # 5. Return the final result
    return GeneratorResponse(
        job_id=request.job_id,
        answer=generated_text,
        latency_ms=latency_ms
    )