import os
import time
import json
import logging
import redis
import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("retriever")

load_dotenv()

# --- Pydantic Models ---
class RAGJob(BaseModel):
    job_id: str = Field(...)
    query: str = Field(...)
    vector: list[float] = Field(...)
    selected_replica_index: int | None = Field(None)
    timestamp: float = Field(default_factory=time.time)


class RetrievedChunk(BaseModel):
    text: str
    score: float
    source: str = "corpus"
    chunk_index: int = 0


class AugmentedPromptJob(BaseModel):
    job_id: str = Field(...)
    augmented_prompt: str = Field(...)
    selected_replica_index: int | None = Field(None)
    retrieved_chunks: list[dict] = Field(default_factory=list)
    retrieval_latency_ms: float = Field(0.0)
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
    logger.info("Initializing clients...")

    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    r.ping()
    logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")

    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    try:
        qdrant_client.get_collection(collection_name=QDRANT_COLLECTION)
        logger.info(f"Connected to Qdrant collection: '{QDRANT_COLLECTION}'")
    except Exception:
        logger.warning(f"Collection '{QDRANT_COLLECTION}' not found — creating with seed data.")
        try:
            qdrant_client.delete_collection(collection_name=QDRANT_COLLECTION)
        except Exception:
            pass
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
        )
        seed_docs = [
            ("RAG (Retrieval-Augmented Generation) combines information retrieval with text generation to produce accurate, grounded responses.", "seed"),
            ("Qdrant is a high-performance vector database designed for semantic search and AI applications.", "seed"),
            ("The LLM Generator service is the final step in the RAG inference pipeline, producing the answer.", "seed"),
            ("Microservices architecture enables independent scaling and deployment of each component in the pipeline.", "seed"),
            ("Redis message queues decouple services, allowing asynchronous processing and improved throughput.", "seed"),
            ("Load balancing distributes requests across replicas using strategies like least-connections and round-robin.", "seed"),
            ("Caching stores frequently accessed data in fast memory to reduce latency and backend load.", "seed"),
            ("FastAPI is a modern Python framework for building high-performance APIs with automatic documentation.", "seed"),
        ]
        points = [
            models.PointStruct(
                id=i + 1,
                vector=np.random.rand(VECTOR_SIZE).tolist(),
                payload={"document": doc, "source": src, "chunk_index": i},
            )
            for i, (doc, src) in enumerate(seed_docs)
        ]
        qdrant_client.upsert(collection_name=QDRANT_COLLECTION, points=points, wait=True)
        logger.info(f"Seeded {len(points)} documents into '{QDRANT_COLLECTION}'.")

    return r, qdrant_client


def create_augmented_prompt(query: str, retrieved_context: list[str]) -> str:
    if retrieved_context:
        context_text = "\n---\n".join(retrieved_context)
    else:
        context_text = "No relevant documents were found."

    return (
        "You are a helpful assistant.\n"
        "Use ONLY the supplied context to answer the question.\n"
        "If the context is insufficient, clearly say so.\n\n"
        f"Context:\n{context_text}\n\n"
        f"USER QUESTION: {query}\n\n"
        "Answer:"
    )


def worker_loop(r: redis.Redis, qdrant_client: QdrantClient):
    logger.info(f"Retriever listening on queue: {ENCODER_QUEUE}")

    while True:
        job_data = r.blpop(ENCODER_QUEUE, timeout=1)

        if job_data:
            _, serialized_job = job_data
            try:
                job_dict = json.loads(serialized_job)
                job = RAGJob(**job_dict)
                job_id = job.job_id

                logger.info(f"[{job_id}] Retrieving for: '{job.query[:60]}'")
                t0 = time.time()

                search_result = qdrant_client.search(
                    collection_name=QDRANT_COLLECTION,
                    query_vector=job.vector,
                    limit=K_HITS,
                    with_payload=True,
                )

                retrieved_docs: list[str] = []
                retrieved_chunks: list[dict] = []

                for hit in search_result:
                    doc = hit.payload.get("document") or hit.payload.get("text", "")
                    if doc:
                        retrieved_docs.append(doc)
                        retrieved_chunks.append({
                            "text": doc[:300],  # truncate for transport
                            "score": round(float(hit.score), 4),
                            "source": hit.payload.get("source", "corpus"),
                            "chunk_index": hit.payload.get("chunk_index", 0),
                        })

                retrieval_latency_ms = (time.time() - t0) * 1000
                logger.info(
                    f"[{job_id}] Found {len(retrieved_docs)} chunks in {retrieval_latency_ms:.1f}ms"
                )

                augmented_prompt = create_augmented_prompt(job.query, retrieved_docs)

                output_job = AugmentedPromptJob(
                    job_id=job_id,
                    augmented_prompt=augmented_prompt,
                    selected_replica_index=job.selected_replica_index,
                    retrieved_chunks=retrieved_chunks,
                    retrieval_latency_ms=retrieval_latency_ms,
                )

                r.rpush(LLM_QUEUE, output_job.model_dump_json())
                logger.info(f"[{job_id}] Published to LLM queue")

            except redis.exceptions.ConnectionError as e:
                logger.error(f"Redis connection lost: {e}. Retrying in 5s.")
                time.sleep(5)
            except Exception as e:
                logger.exception(f"Failed to process job: {e}")

        time.sleep(0.05)


if __name__ == "__main__":
    try:
        redis_client, qdrant_client = initialize_clients()
        worker_loop(redis_client, qdrant_client)
    except Exception as e:
        logger.critical(f"FATAL: Service initialization failed: {e}")
        raise
