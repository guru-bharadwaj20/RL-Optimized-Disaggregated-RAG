import os
import time
import json
import redis
import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from pydantic import BaseModel, Field

load_dotenv()

# --- Pydantic Models ---
class RAGJob(BaseModel):
    job_id: str = Field(..., description="Unique job ID.")
    query: str = Field(..., description="Original user query.")
    vector: list[float] = Field(..., description="Encoded query vector.")
    selected_replica_index: int | None = Field(None, description="Load-balancing replica index.")
    timestamp: float = Field(default_factory=time.time)


class AugmentedPromptJob(BaseModel):
    job_id: str = Field(...)
    augmented_prompt: str = Field(...)
    selected_replica_index: int | None = Field(None)
    retrieval_time: float = Field(default_factory=time.time)


# --- Configuration ---
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
ENCODER_QUEUE = os.getenv("ENCODER_QUEUE", "job:retriever_in")
LLM_QUEUE = os.getenv("LLM_QUEUE", "job:llm_in")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_collection")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", 384))
K_HITS = int(os.getenv("K_HITS", 3))


def initialize_clients():
    print("Initializing clients...")

    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    r.ping()
    print(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")

    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    try:
        qdrant_client.get_collection(collection_name=QDRANT_COLLECTION)
        print(f"Connected to Qdrant collection: '{QDRANT_COLLECTION}'")
    except Exception:
        print(f"Collection '{QDRANT_COLLECTION}' not found — creating with seed data.")
        # Delete if exists in broken state, then create fresh
        try:
            qdrant_client.delete_collection(collection_name=QDRANT_COLLECTION)
        except Exception:
            pass
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
        )
        seed_docs = [
            "RAG (Retrieval-Augmented Generation) combines information retrieval with text generation to produce accurate, grounded responses.",
            "Qdrant is a high-performance vector database designed for semantic search and AI applications.",
            "The LLM Generator service is the final step in the RAG inference pipeline, producing the answer.",
            "Microservices architecture enables independent scaling and deployment of each component in the pipeline.",
            "Redis message queues decouple services, allowing asynchronous processing and improved throughput.",
            "Load balancing distributes requests across replicas using strategies like least-connections and round-robin.",
            "Caching stores frequently accessed data in fast memory to reduce latency and backend load.",
            "FastAPI is a modern Python framework for building high-performance APIs with automatic documentation.",
        ]
        points = [
            models.PointStruct(
                id=i + 1,
                vector=np.random.rand(VECTOR_SIZE).tolist(),
                payload={"document": doc},
            )
            for i, doc in enumerate(seed_docs)
        ]
        qdrant_client.upsert(collection_name=QDRANT_COLLECTION, points=points, wait=True)
        print(f"Seeded {len(points)} mock documents into '{QDRANT_COLLECTION}'.")

    return r, qdrant_client


def create_augmented_prompt(query: str, retrieved_context: list[str]) -> str:
    if retrieved_context:
        context_text = "\n---\n".join(retrieved_context)
    else:
        context_text = "No relevant documents were found."

    return (
        "You are an expert assistant. Use the provided context to answer the user's question.\n"
        "If the context does not contain the answer, say so clearly.\n\n"
        f"CONTEXT:\n{context_text}\n\n"
        f"USER QUESTION: {query}"
    )


def worker_loop(r: redis.Redis, qdrant_client: QdrantClient):
    print(f"Retriever listening on queue: {ENCODER_QUEUE}")

    while True:
        job_data = r.blpop(ENCODER_QUEUE, timeout=1)

        if job_data:
            _, serialized_job = job_data
            try:
                job_dict = json.loads(serialized_job)
                job = RAGJob(**job_dict)
                job_id = job.job_id

                print(f"[{job_id}] Retrieving for: '{job.query[:60]}'")
                start_time = time.time()

                search_result = qdrant_client.search(
                    collection_name=QDRANT_COLLECTION,
                    query_vector=job.vector,
                    limit=K_HITS,
                    with_payload=True,
                    # No score_threshold — with seed random vectors, scores may be low
                )

                retrieved_docs = []
                for hit in search_result:
                    # Support both 'document' (standard) and 'text' (legacy ingestion key)
                    doc = hit.payload.get("document") or hit.payload.get("text", "")
                    if doc:
                        retrieved_docs.append(doc)

                elapsed = time.time() - start_time
                print(f"[{job_id}] Found {len(retrieved_docs)} chunks in {elapsed:.4f}s")

                augmented_prompt = create_augmented_prompt(job.query, retrieved_docs)

                output_job = AugmentedPromptJob(
                    job_id=job_id,
                    augmented_prompt=augmented_prompt,
                    selected_replica_index=job.selected_replica_index,
                )

                r.rpush(LLM_QUEUE, output_job.model_dump_json())
                print(f"[{job_id}] Published augmented prompt to LLM queue")

            except redis.exceptions.ConnectionError as e:
                print(f"Redis connection lost: {e}. Retrying in 5s.")
                time.sleep(5)
            except Exception as e:
                print(f"Failed to process job: {e}")

        time.sleep(0.05)


if __name__ == "__main__":
    try:
        redis_client, qdrant_client = initialize_clients()
        worker_loop(redis_client, qdrant_client)
    except Exception as e:
        print(f"FATAL: Service initialization failed: {e}")
        raise
