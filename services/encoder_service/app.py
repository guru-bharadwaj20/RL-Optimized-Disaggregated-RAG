import os
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
VECTOR_DIMENSION = 384 



try:
    # Set device to 'cpu' for standard microservice deployment
    model = SentenceTransformer(MODEL_NAME, device='cpu')
except Exception as e:
    model = None 

# --- FastAPI Setup ---
app = FastAPI(title="Encoder Service")

# --- Data Models ---
class QueryRequest(BaseModel):
    text: str

class VectorResponse(BaseModel):
    vector: list[float]
    dim: int = VECTOR_DIMENSION

# --- Health Check Endpoint ---
@app.get("/health")
def health_check():
    """Checks service and model status for auto-scaling/monitoring."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return {"status": "ok", "model": MODEL_NAME}

# --- Main Encoding Endpoint ---
@app.post("/encode", response_model=VectorResponse)
async def encode_query(request: QueryRequest):
    """Generates the 384-dimensional embedding for the input text."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model unavailable.")
    
    if not request.text or request.text.strip() == "":
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    try:
        # Encode text and convert NumPy array to Python list for JSON serialization
        embedding: np.ndarray = model.encode(request.text, convert_to_tensor=False)
        
        return VectorResponse(vector=embedding.tolist(), dim=embedding.shape[0])

    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to generate embedding.")

