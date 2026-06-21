"""
Mock LLM service for local testing.
Simulates LLM responses with deterministic keyword-based answers.
Matches keywords against the USER QUESTION section of the prompt,
not the full augmented context, so answers are relevant to the actual query.
"""
import os
import re
import time
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Mock LLM Service", version="1.0")


class GenerateRequest(BaseModel):
    prompt: str = None
    query: str = None
    job_id: str = None


class GenerateResponse(BaseModel):
    job_id: str
    answer: str
    latency_ms: float


MOCK_RESPONSES = {
    "recursion": (
        "Recursion is a programming technique where a function calls itself to solve smaller "
        "sub-problems. Every recursive function needs a base case (where it stops) and a recursive "
        "case (where it calls itself). Classic examples include computing factorial (n! = n × (n-1)!), "
        "Fibonacci numbers, and tree/graph traversals."
    ),
    "microservice": (
        "Microservices is an architectural style where an application is built as a collection of "
        "small, independently deployable services. Each service owns a specific business capability "
        "and communicates over APIs or message queues. Benefits include independent scaling, "
        "technology flexibility, fault isolation, and faster deployments."
    ),
    "rag": (
        "RAG (Retrieval-Augmented Generation) enhances LLM responses by first retrieving relevant "
        "documents from a knowledge base, then injecting that context into the prompt before "
        "generation. This grounds responses in real data, reduces hallucinations, and allows the "
        "model to answer questions about information not seen during training."
    ),
    "kubernetes": (
        "Kubernetes (K8s) is an open-source container orchestration platform that automates "
        "deployment, scaling, and management of containerized applications. Key features include "
        "self-healing (auto-restart failed pods), horizontal scaling, rolling updates with "
        "zero downtime, and service discovery via internal DNS."
    ),
    "distributed": (
        "Distributed systems consist of multiple independent computers working together toward "
        "a common goal. They enable horizontal scalability and fault tolerance but introduce "
        "challenges around consistency (CAP theorem), network latency, and partial failures. "
        "The CAP theorem states you can only guarantee two of: Consistency, Availability, Partition tolerance."
    ),
    "cach": (
        "Caching stores frequently accessed data in fast memory (RAM) to avoid repeated expensive "
        "computations or database lookups. Common strategies: LRU (evict least-recently used), "
        "LFU (evict least-frequently used), and TTL (expire after a time). Effective caching can "
        "reduce backend load by over 90% and cut response latency dramatically."
    ),
    "load balanc": (
        "Load balancing distributes incoming requests across multiple backend servers to maximize "
        "throughput and prevent overload. Algorithms include round-robin (sequential distribution), "
        "least-connections (route to the server with fewest active requests), and IP-hash (sticky "
        "sessions). The least-connections algorithm is most effective for variable-duration workloads."
    ),
    "api": (
        "An API (Application Programming Interface) defines how software components communicate. "
        "REST APIs use HTTP methods (GET, POST, PUT, DELETE) and are stateless and cacheable. "
        "FastAPI is a modern Python framework that auto-generates OpenAPI documentation and uses "
        "Python type hints for request validation, making it one of the fastest Python API frameworks."
    ),
    "database": (
        "Databases are organized collections of structured data. Relational (SQL) databases like "
        "PostgreSQL enforce schemas and ACID transactions. NoSQL databases (document stores like "
        "MongoDB, key-value stores like Redis, graph databases like Neo4j) sacrifice strict "
        "consistency for flexibility and horizontal scalability. Vector databases like Qdrant "
        "specialize in storing and searching high-dimensional embedding vectors."
    ),
    "vector": (
        "Vector databases store high-dimensional embedding vectors and enable semantic similarity "
        "search using distance metrics (cosine, dot-product, Euclidean). Unlike keyword search, "
        "vector search captures semantic meaning — finding conceptually similar content even when "
        "exact words differ. Qdrant is a purpose-built vector database optimized for RAG pipelines."
    ),
    "embed": (
        "Embeddings are dense numerical vector representations of text (or images, audio) learned "
        "by neural networks. The all-MiniLM-L6-v2 model produces 384-dimensional embeddings that "
        "capture semantic meaning. Similar sentences cluster near each other in embedding space, "
        "enabling semantic search, clustering, and retrieval tasks."
    ),
    "redis": (
        "Redis is an in-memory data structure store supporting strings, hashes, lists, sets, and "
        "sorted sets. It excels as a message broker using LPUSH/BRPOP for reliable queue-based "
        "communication between microservices. Redis also supports pub/sub patterns, atomic counters "
        "for load tracking, and TTL-based key expiration."
    ),
    "docker": (
        "Docker packages applications and their dependencies into portable containers that run "
        "identically across environments. Docker Compose orchestrates multi-container applications "
        "in a single YAML file, defining services, networks, volumes, and dependencies — making "
        "local development of microservice architectures straightforward."
    ),
    "message queue": (
        "Message queues decouple producers from consumers in distributed systems. A producer pushes "
        "a job and immediately returns; consumers process it asynchronously at their own pace. This "
        "enables horizontal scaling (add more consumers), handles traffic spikes (queue buffers "
        "bursts), and improves fault tolerance (jobs aren't lost if a consumer crashes)."
    ),
    "prometheus": (
        "Prometheus is an open-source monitoring system that scrapes metrics from HTTP /metrics "
        "endpoints. Applications expose counters (total requests), gauges (active jobs), and "
        "histograms (latency distributions). PromQL lets you query metrics like P95 latency: "
        "histogram_quantile(0.95, rate(api_gateway_query_latency_seconds_bucket[30s]))."
    ),
    "fastapi": (
        "FastAPI is a modern, high-performance Python web framework for building APIs. It uses "
        "Python type hints for automatic request validation via Pydantic, auto-generates Swagger "
        "and ReDoc documentation, and supports async/await for non-blocking I/O. FastAPI "
        "benchmarks comparably to NodeJS and Go for I/O-bound workloads."
    ),
}

_DEFAULT_ANSWER = (
    "Based on the retrieved context, this question relates to modern distributed systems and "
    "AI infrastructure. The system uses a microservices architecture with Redis-based message "
    "queues, Qdrant vector search, and an LLM generator to retrieve relevant information and "
    "synthesize accurate answers. For more specific information, please refine your question."
)


def _extract_user_question(text: str) -> str:
    """Extract the USER QUESTION section from an augmented RAG prompt."""
    match = re.search(r"USER QUESTION:\s*(.+)$", text, re.MULTILINE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "mock_llm"}


@app.post("/generate", response_model=GenerateResponse)
async def generate_response(request: GenerateRequest):
    start_time = time.time()

    full_text = request.prompt or request.query or ""

    # Match keywords against the user's actual question, not the retrieved context
    search_text = _extract_user_question(full_text).lower()

    answer = _DEFAULT_ANSWER
    for keyword, response in MOCK_RESPONSES.items():
        if keyword in search_text:
            answer = response
            break

    # Simulate realistic processing time
    time.sleep(0.05 + (len(answer) % 80) * 0.001)

    latency_ms = (time.time() - start_time) * 1000
    return GenerateResponse(
        job_id=request.job_id or "",
        answer=answer,
        latency_ms=latency_ms,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
