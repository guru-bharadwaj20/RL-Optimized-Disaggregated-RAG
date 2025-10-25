"""
Quick mock LLM service for local testing.
Simulates real LLM response with deterministic answers.
"""
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time

app = FastAPI(title="Mock LLM Service", version="1.0")

class GenerateRequest(BaseModel):
    prompt: str = None
    query: str = None  # Accept both prompt and query
    job_id: str = None

class GenerateResponse(BaseModel):
    job_id: str
    answer: str
    latency_ms: float

# Mock knowledge base - deterministic responses based on keywords
MOCK_RESPONSES = {
    "recursion": "Recursion is a programming technique where a function calls itself to solve smaller instances of the same problem. It typically has a base case (where recursion stops) and a recursive case (where it calls itself). Classic examples include factorial, Fibonacci, and tree traversal.",
    "microservices": "Microservices is an architectural approach where an application is built as a collection of small, loosely coupled, independently deployable services. Each service handles a specific business function and communicates with others via APIs. Benefits include scalability, flexibility, and easier maintenance.",
    "rag": "RAG (Retrieval-Augmented Generation) is an AI technique that combines information retrieval with text generation. It retrieves relevant documents or context first, then uses that context to generate more accurate and informed responses. This reduces hallucinations and grounds responses in real data.",
    "kubernetes": "Kubernetes (K8s) is an open-source container orchestration platform that automates the deployment, scaling, and management of containerized applications. It provides features like load balancing, self-healing, rolling updates, and resource management.",
    "distributed systems": "Distributed systems consist of multiple independent computers that communicate and coordinate to achieve a common goal. They enable scalability and fault tolerance but introduce challenges like consistency, latency, and network failures.",
    "caching": "Caching stores frequently accessed data in fast-access storage (like memory) to reduce latency and load on slower storage. Common caching strategies include LRU (Least Recently Used), LFU (Least Frequently Used), and TTL (Time To Live).",
    "load balancing": "Load balancing distributes incoming requests across multiple servers to optimize resource use, maximize throughput, minimize response time, and avoid overload on any single server. Methods include round-robin, least-connections, and IP-hash.",
    "api": "An API (Application Programming Interface) defines how software components communicate. REST APIs use HTTP methods (GET, POST, PUT, DELETE) and are widely used for web services. APIs enable integration between different systems and services.",
    "database": "A database is an organized collection of structured data stored and accessed electronically. Types include relational (SQL), NoSQL (document, key-value, graph), and time-series databases. Each has different trade-offs for consistency, availability, and scalability.",
    "default": "That's an interesting question. Based on the context provided and my knowledge, I can offer the following insights: The topic you're asking about involves several key concepts related to modern software architecture and distributed systems. I'd recommend exploring the specific components and their interactions to gain a deeper understanding."
}

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "mock_llm"}

@app.post("/generate", response_model=GenerateResponse)
async def generate_response(request: GenerateRequest):
    """
    Generate a response by matching keywords in the prompt
    against a mock knowledge base.
    """
    start_time = time.time()
    
    # Support both 'prompt' and 'query' fields
    text = request.prompt or request.query or ""
    
    # Find best matching response based on keywords
    text_lower = text.lower()
    answer = MOCK_RESPONSES["default"]
    
    for keyword, response in MOCK_RESPONSES.items():
        if keyword in text_lower:
            answer = response
            break
    
    # Simulate some processing time (50-200ms to be realistic)
    time.sleep(0.05 + (len(answer) % 100) * 0.001)
    
    latency_ms = (time.time() - start_time) * 1000
    
    return GenerateResponse(
        job_id=request.job_id or "",
        answer=answer,
        latency_ms=latency_ms
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
