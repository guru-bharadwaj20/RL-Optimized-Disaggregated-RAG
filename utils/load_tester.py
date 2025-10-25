"""Simple load tester: fire N concurrent requests at /query and compute P99 latency."""
import os
import time
import asyncio
import statistics
import httpx

# API Gateway URL (port 8000 is the external API gateway port in docker-compose.clean.yml)
API_URL = os.getenv("API_URL", "http://localhost:8000/query")
CONCURRENCY = int(os.getenv("LT_CONCURRENCY", 50))
SAMPLE_QUERIES = [
    "What is recursion in computer science?",
    "Explain the difference between REST and GraphQL.",
    "How does machine learning improve RAG systems?",
    "What are the benefits of microservices architecture?",
    "Describe the role of embeddings in vector databases."
]


async def send_once(client: httpx.AsyncClient, query: str):
    start = time.time()
    try:
        r = await client.post(API_URL, json={"query": query}, timeout=60.0)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return None, time.time() - start, str(e)
    return data, time.time() - start, None


async def run_test(concurrency: int = CONCURRENCY):
    async with httpx.AsyncClient() as client:
        tasks = [send_once(client, SAMPLE_QUERIES[i % len(SAMPLE_QUERIES)]) for i in range(concurrency)]
        results = await asyncio.gather(*tasks)

    latencies = [r[1] for r in results if r[1] is not None]
    errors = [r[2] for r in results if r[2] is not None]

    if not latencies:
        print("No successful requests")
        return

    lat_ms = [l * 1000 for l in latencies]
    p99 = statistics.quantiles(lat_ms, n=100)[98] if len(lat_ms) >= 100 else max(lat_ms)
    p95 = statistics.quantiles(lat_ms, n=100)[94] if len(lat_ms) >= 100 else max(lat_ms)
    p50 = statistics.median(lat_ms)
    
    print(f"\n--- Load Test Results ---")
    print(f"Total Requests: {len(lat_ms)}")
    print(f"Successful: {len(lat_ms)}, Errors: {len(errors)}")
    print(f"P50 (Median) latency (ms): {p50:.2f}")
    print(f"P95 latency (ms): {p95:.2f}")
    print(f"P99 latency (ms): {p99:.2f}")
    print(f"Min latency (ms): {min(lat_ms):.2f}")
    print(f"Max latency (ms): {max(lat_ms):.2f}")
    if errors:
        print(f"\nSample errors: {errors[:3]}")
    print(f"API URL: {API_URL}")


if __name__ == '__main__':
    asyncio.run(run_test())