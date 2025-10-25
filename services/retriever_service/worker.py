import os
import time
import json
import redis
import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from pydantic import BaseModel, Field

# Load environment variables from .env file (if it exists)
load_dotenv()

# --- Pydantic Models ---
class RAGJob(BaseModel):
    """Schema for the RAG job data received from the Encoder Service."""
    job_id: str = Field(..., description="Unique ID for tracking the job.")
    query: str = Field(..., description="The original user query.")
    vector: list[float] = Field(..., description="The encoded query vector.")
    selected_replica_index: int | None = Field(None, description="Index of selected LLM replica (for load tracking only).")
    timestamp: float = Field(default_factory=time.time, description="Time job was queued.")

class AugmentedPromptJob(BaseModel):
    """Schema for the job data published to the LLM Generator Service."""
    job_id: str = Field(..., description="Unique ID for tracking the job.")
    augmented_prompt: str = Field(..., description="The final prompt including retrieved context.")
    retrieval_time: float = Field(default_factory=time.time, description="Time retrieval was completed.")

# --- Configuration ---
# Environment Variables
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333)) # Qdrant gRPC default port is 6334, REST default is 6333

# Queue names (align with encoder/llm naming)
ENCODER_QUEUE = os.getenv("ENCODER_QUEUE", "job:retriever_in") # Input queue from encoder
LLM_QUEUE = os.getenv("LLM_QUEUE", "job:llm_in")           # Output queue to LLM generator

# RAG configuration
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_collection")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", 384)) # Match all-MiniLM-L6-v2 vector size
K_HITS = int(os.getenv("K_HITS", 3)) # Number of chunks to retrieve

# --- Initialize Clients ---
def initialize_clients():
    """Initializes Redis and Qdrant connections."""
    print("Initializing clients...")
    
    # Redis Client (Message Queue)
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    r.ping()
    print(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")

    # Qdrant Client (Note: using REST API port)
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    # Check if collection exists and ensure a mock setup for testing
    try:
        qdrant_client.get_collection(collection_name=QDRANT_COLLECTION)
        print(f"Connected to Qdrant Collection: {QDRANT_COLLECTION}")
    except Exception as e:
        print(f"Qdrant Collection '{QDRANT_COLLECTION}' not found. Creating mock data.")
        
        # Create a basic collection for the RAG data
        qdrant_client.recreate_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE)
        )
        
        # Add mock documents (vectors are random, only for successful execution)
        points = [
            models.PointStruct(id=1, vector=np.random.rand(VECTOR_SIZE).tolist(), payload={"document": "The RAG pipeline uses an Asynchronous Message Queue for decoupling."}),
            models.PointStruct(id=2, vector=np.random.rand(VECTOR_SIZE).tolist(), payload={"document": "Qdrant is a powerful, distributed vector database."}),
            models.PointStruct(id=3, vector=np.random.rand(VECTOR_SIZE).tolist(), payload={"document": "The LLM Generator service is the final step in the inference process."})
        ]
        qdrant_client.upsert(collection_name=QDRANT_COLLECTION, points=points, wait=True)
        print("Mock Qdrant data added successfully.")
        
    return r, qdrant_client

# --- Main Logic ---

def create_augmented_prompt(query: str, retrieved_context: list[str]) -> str:
    """
    Constructs the final prompt with context and instructions for the LLM.
    """
    context_text = "\n---\n".join(retrieved_context)
    
    prompt = (
        "You are an expert RAG system. Use the provided context to answer the user's question. "
        "If the context does not contain the answer, state that you cannot answer based on the provided documents.\n\n"
        f"CONTEXT:\n{context_text}\n\n"
        f"USER QUESTION: {query}"
    )
    return prompt

def worker_loop(r: redis.Redis, qdrant_client: QdrantClient):
    """
    The main loop that listens to the Redis queue for new jobs.
    """
    print(f"Retriever Service is listening on Redis queue: {ENCODER_QUEUE}")
    
    while True:
        # Blocking List POP (BLPOP) to wait for new messages
        job_data = r.blpop(ENCODER_QUEUE, timeout=1)
        
        if job_data:
            queue_name, serialized_job = job_data
            
            try:
                # 1. Parse the incoming job data
                job_dict = json.loads(serialized_job)
                job = RAGJob(**job_dict)
                job_id = job.job_id

                print(f"[{job_id}] Job received. Starting retrieval for query: '{job.query[:50]}...'")

                # 2. Perform Vector Search (the core retrieval step using Qdrant)
                start_time = time.time()
                
                search_result = qdrant_client.search(
                    collection_name=QDRANT_COLLECTION,
                    query_vector=job.vector, # The encoded vector from the encoder service
                    limit=K_HITS,
                    with_payload=True, # Ensure we get the document text
                    score_threshold=0.5 # Optional: Filter out low-relevance results
                )
                
                # Extract the document text from the payload
                retrieved_docs = [hit.payload['document'] for hit in search_result]
                
                end_time = time.time()
                
                print(f"[{job_id}] Retrieval complete in {end_time - start_time:.4f}s. Found {len(retrieved_docs)} chunks.")

                # 3. Augment the Prompt
                augmented_prompt = create_augmented_prompt(job.query, retrieved_docs)

                # 4. Publish the Augmented Prompt to the next queue (LLM Generator)
                output_job = AugmentedPromptJob(
                    job_id=job_id,
                    augmented_prompt=augmented_prompt,
                    retrieval_time=time.time()
                )
                
                # Push the job to the LLM queue
                r.rpush(LLM_QUEUE, output_job.model_dump_json())
                print(f"[{job_id}] Augmented prompt published to queue: {LLM_QUEUE}")

            except redis.exceptions.ConnectionError as e:
                print(f"ERROR: Redis connection lost. Retrying in 5s. {e}")
                time.sleep(5)
            except Exception as e:
                print(f"ERROR: Failed to process job: {e}")
                # Log faulty job content and continue
                print(f"Faulty job content: {serialized_job if 'serialized_job' in locals() else 'N/A'}")
        
        # A small non-blocking sleep is fine to yield control.
        time.sleep(0.1)


if __name__ == '__main__':
    try:
        redis_client, qdrant_client = initialize_clients()
        worker_loop(redis_client, qdrant_client)
    except Exception as e:
        print(f"FATAL ERROR: Service initialization failed: {e}")
