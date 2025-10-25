import os
import uuid
import time
import json
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- Pydantic Models for Data Validation ---

class QueryRequest(BaseModel):
    """Schema for the incoming user query."""
    query: str

class StatusResponse(BaseModel):
    """Schema for the status check response."""
    session_id: str
    status: str
    message: str
    timestamp: float
    # result is optional, only present when status is COMPLETED
    result: str | None = None
    status_endpoint: str # Adding this back for convenience

# --- Configuration ---
# Load environment variables for the message queue (MQ)
MQ_HOST = os.getenv("MQ_HOST", "localhost")
MQ_PORT = int(os.getenv("MQ_PORT", 5672))
QUERY_QUEUE = os.getenv("QUERY_QUEUE", "rag_query_topic")

# Initialize FastAPI App
app = FastAPI(
    title="RAG API Gateway",
    description="Receives user queries, queues them for async RAG processing, and provides status tracking.",
    version="1.0.0"
)

# --- Mock Message Queue Client ---
# This mock class simulates the publishing step.
class MessageQueueClient:
    """
    Mocks the client for the Asynchronous Message Queue (e.g., RabbitMQ, Kafka).
    In production, this would use a library like Pika or Confluent Kafka.
    """
    def __init__(self, host, port):
        self.host = host
        self.port = port
        print(f"MQ Client initialized for {self.host}:{self.port}")
        # self.connection = pika.BlockingConnection(...) # Real connection goes here

    def publish_query(self, session_id: str, user_query: str) -> bool:
        """
        Simulates publishing the RAG request (State A) to the message queue.
        The message contains the session ID for tracking and the user's query.
        """
        message = {
            "session_id": session_id,
            "query": user_query,
            "timestamp": time.time(),
            "status": "QUEUED" # Initial status
        }
        
        # --- MOCK ACTION ---
        # In a real app, this would be: self.channel.basic_publish(...)
        # For simplicity, we just log the action:
        print(f"[{session_id}] RAG Job published to MQ: '{user_query}'")
        return True

# Initialize MQ Client globally 
try:
    mq_client = MessageQueueClient(MQ_HOST, MQ_PORT)
except Exception as e:
    print(f"ERROR: Failed to initialize MQ client: {e}")
    # In a real microservice, you might add retry logic or fail fast here.

# --- API Endpoints ---

@app.post("/query", response_model=StatusResponse, status_code=202)
async def handle_query(request_body: QueryRequest):
    """
    Receives a user query, generates a session ID, and publishes the job to the MQ.
    Returns the session ID and status endpoint for client tracking.
    """
    # 1. Generate unique session ID (the key to track the async job)
    session_id = str(uuid.uuid4())
    user_query = request_body.query
    
    # 2. Publish the job to the Asynchronous Message Queue
    try:
        mq_client.publish_query(session_id, user_query)
    except Exception as e:
        # If queuing fails, return a 503 Service Unavailable
        raise HTTPException(
            status_code=503,
            detail=f"Failed to queue job: Message Queue unavailable or error: {e}"
        )

    # 3. Return the session ID to the client (202 Accepted)
    return StatusResponse(
        message="RAG job successfully queued.",
        session_id=session_id,
        status="QUEUED",
        timestamp=time.time(),
        result=None,
        status_endpoint=f"/status/{session_id}"
    )

@app.get("/status/{session_id}", response_model=StatusResponse)
async def get_status(session_id: str):
    """
    Allows the user to poll for the status of their RAG job using the session ID.
    In a real system, this would query a cache (like Redis) or database (like Firestore)
    where the final result/status is stored by the LLM Generation service.
    """
    # --- MOCK STATUS RETRIEVAL ---
    # Replace this with actual database/cache lookup in production
    mock_statuses = {
        "1234": "COMPLETED",
        "5678": "PROCESSING: Vector Search Completed",
    }
    
    # Simple check for the demo, otherwise assume it's still being processed
    if session_id in mock_statuses:
        status = mock_statuses[session_id]
        if status == "COMPLETED":
            result = f"The official answer for known ID {session_id} is 'Apollo 11'."
        else:
            result = None
    else:
        # Simulate different stages based on a simple hash of the ID for demonstration
        # This makes different UUIDs give different mock responses
        hash_val = sum(ord(c) for c in session_id)
        
        if hash_val % 3 == 0:
            status = "COMPLETED"
            result = f"The answer for session {session_id} is: Placeholder data successfully generated after RAG completion."
        elif hash_val % 3 == 1:
            status = "PROCESSING: Retrieval and Vector Search Completed"
            result = None
        else:
            status = "QUEUED: Awaiting LLM Processing"
            result = None


    message = f"Processing status: {status}"
    if status.startswith("COMPLETED"):
        message = "Processing complete. Result available."
    elif status.startswith("QUEUED"):
        message = "Job is in the queue, waiting for an available worker."
    
    return StatusResponse(
        session_id=session_id,
        status=status,
        message=message,
        timestamp=time.time(),
        result=result,
        status_endpoint=f"/status/{session_id}"
    )

@app.get('/health')
async def health_check():
    """Basic health check endpoint."""
    return {"status": "ok", "service": "API Gateway", "framework": "FastAPI"}

# --- Entry Point (for local development with Uvicorn) ---
if __name__ == '__main__':
    # Running locally with 'python app.py'
    print("Running API Gateway locally with Uvicorn on http://127.0.0.1:8000")
    # Note: In production, the Dockerfile/deployment config would use a Uvicorn worker
    # running under Gunicorn (e.g., `gunicorn -k uvicorn.workers.UvicornWorker`).
    uvicorn.run(app, host='0.0.0.0', port=8000)
