# RL-Optimized Disaggregated RAG

A production-style **Retrieval-Augmented Generation (RAG)** system built as independent microservices connected by Redis queues, with semantic search via Qdrant, least-connections load balancing across LLM worker replicas, real-time Prometheus metrics, and an interactive Streamlit chat UI.

---

## Architecture

```
                        ┌─────────────────────────────┐
                        │       Streamlit UI           │
                        │       (port 8501)            │
                        └───────────┬─────────────────┘
                                    │ HTTP POST /query
                                    ▼
                        ┌─────────────────────────────┐
                        │       API Gateway            │  ◄── /metrics (Prometheus)
                        │       (port 8000)            │
                        │  Least-connections LB        │
                        └──────────┬──────────────────┘
                                   │ Redis LPUSH
                                   ▼
                         job:encoder_in queue
                                   │
                                   ▼
                        ┌─────────────────────────────┐
                        │     Encoder Service          │
                        │  all-MiniLM-L6-v2 (384-dim) │
                        └──────────┬──────────────────┘
                                   │ Redis LPUSH (vector)
                                   ▼
                         job:retriever_in queue
                                   │
                                   ▼
                        ┌─────────────────────────────┐
                        │    Retriever Service         │
                        │    Qdrant cosine search      │
                        └──────────┬──────────────────┘
                                   │ Redis RPUSH (augmented prompt)
                                   ▼
                          job:llm_in queue
                                   │
                     ┌─────────────┴─────────────┐
                     ▼                           ▼
           ┌──────────────────┐      ┌──────────────────┐
           │  LLM Worker 1    │      │  LLM Worker 2    │
           └────────┬─────────┘      └────────┬─────────┘
                    │                          │
                    └──────────┬───────────────┘
                               │ HTTP POST /generate
                               ▼
                    ┌──────────────────────┐
                    │     Mock LLM         │
                    │   (port 8888)        │
                    └──────────────────────┘
                               │
                    Redis SET job:completion:{id}
                               │
                               ▼
                        API Gateway polls → FinalResponse
```

### Services

| Service | Port | Role |
|---------|------|------|
| API Gateway | 8000 | Entry point; least-connections load balancing; Prometheus metrics |
| Encoder Service | — | Encodes query text to 384-dim vectors via sentence-transformers |
| Retriever Service | — | Searches Qdrant for top-K relevant document chunks |
| LLM Worker ×2 | — | Calls Mock LLM with augmented prompt; writes completion to Redis |
| Mock LLM | 8888 | Fast deterministic keyword-based responses (no real LLM cost) |
| Streamlit UI | 8501 | Chat interface with per-query latency and replica metrics |
| Qdrant | 6333 | Vector database for semantic search |
| Redis | 6379 | Message queues + completion key storage |
| Prometheus | 9090 | Metrics scraping and time-series storage |

---

## Prerequisites

- **Docker** and **Docker Compose** (Docker Desktop ≥ 4.x)
- **8 GB RAM** recommended (sentence-transformer model + Qdrant)
- **Python 3.11+** — only for running ingestion locally (optional)

---

## Quick Start

### 1. Start all services

```bash
docker-compose up -d --build
```

Wait ~30–60 s for the encoder to download and cache the `all-MiniLM-L6-v2` model (~90 MB) on first run.

### 2. Check service health

```bash
docker-compose ps
```

All services should show `Up` or `Up (healthy)`.

### 3. (Optional) Ingest your own documents

The retriever auto-seeds a small document set on startup.  
To replace it with the included corpus (22 topic-rich paragraphs):

```bash
# Requires Python 3.11+ with qdrant-client and sentence-transformers
pip install sentence-transformers qdrant-client
python utils/ingestion.py
```

### 4. Open the chat UI

```
http://localhost:8501
```

Type any question and watch it flow through the pipeline in real time.

---

## Querying the API directly

**curl:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is load balancing?"}'
```

**Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/query",
    json={"query": "How does caching reduce latency?"}
)
print(response.json())
```

**Response:**
```json
{
  "job_id": "a3f1c2d4-...",
  "answer": "Caching stores frequently accessed data in fast memory...",
  "latency_ms": 312.4,
  "selected_replica": "replica-1"
}
```

**Sample questions to try:**
- `What is RAG?`
- `How does vector search work?`
- `Explain Redis message queues`
- `What is the CAP theorem?`
- `How does Kubernetes scale applications?`
- `What is recursion?`

---

## Monitoring

### Prometheus

Open **http://localhost:9090** and run PromQL queries:

| Query | Description |
|-------|-------------|
| `api_gateway_queries_total` | Total queries by status (success/error/timeout) |
| `rate(api_gateway_queries_total[30s])` | Requests per second |
| `histogram_quantile(0.95, rate(api_gateway_query_latency_seconds_bucket[30s]))` | P95 latency |
| `api_gateway_active_jobs` | Currently in-flight jobs |
| `api_gateway_replica_load` | Active load on each replica |

### Raw metrics endpoint

```bash
curl http://localhost:8000/metrics
```

---

## Load Testing

### Built-in async tester

```bash
pip install httpx
python utils/load_tester.py
```

Output shows P50, P95, P99 latencies across 50 concurrent requests.

### Locust (web UI)

```bash
pip install locust
cd rag-optimizer/locust
locust -f locustfile.py --host=http://localhost:8000
```

Open **http://localhost:8089**, set users and spawn rate, click **Start**.

---

## Configuration

### Environment variables

All services read from environment variables (set in `docker-compose.yml`):

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_HOST` | `redis` | Redis hostname |
| `QDRANT_HOST` | `qdrant` | Qdrant hostname |
| `QDRANT_COLLECTION` | `rag_collection` | Qdrant collection name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `VECTOR_DIMENSION` | `384` | Embedding dimensionality |
| `K_HITS` | `3` | Number of documents to retrieve |
| `LLM_SERVICE_URL` | `http://mock_llm:8000` | LLM backend URL |
| `LLM_MAX_RETRIES` | `3` | Retries on LLM call failure |

### Scaling LLM workers

Add a third worker in `docker-compose.yml` and update `LLM_REPLICAS` in the API Gateway:

```yaml
# docker-compose.yml
llm_generator_3:
  build:
    context: ./services/llm_generator
    dockerfile: Dockerfile
  environment:
    - REDIS_HOST=redis
    - LLM_SERVICE_URL=http://mock_llm:8000
  depends_on:
    - redis
    - mock_llm
  restart: unless-stopped
```

```python
# services/api_gateway/app.py
LLM_REPLICAS = ["replica-1", "replica-2", "replica-3"]
```

---

## Project Structure

```
RL-Optimized-Disaggregated-RAG/
├── docker-compose.yml              # Full stack orchestration
├── data/
│   └── corpus.txt                 # Knowledge base (22 topic paragraphs)
├── services/
│   ├── api_gateway/               # FastAPI + Prometheus + load balancing
│   ├── encoder_service/           # Sentence-transformer embedding worker
│   ├── retriever_service/         # Qdrant vector search worker
│   ├── llm_generator/             # LLM inference worker (HTTP + Redis)
│   └── mock_llm/                  # Fast deterministic mock LLM service
├── rag-optimizer/
│   ├── streamlit_app/             # Interactive chat UI (port 8501)
│   │   ├── app.py
│   │   └── Dockerfile
│   ├── prometheus/
│   │   └── prometheus.yml         # Prometheus scrape config
│   └── locust/
│       └── locustfile.py          # Load test scenarios
└── utils/
    ├── ingestion.py               # Corpus → Qdrant ingestion script
    └── load_tester.py             # Async P50/P95/P99 latency tester
```

---

## Troubleshooting

**Services won't start**
```bash
docker-compose logs -f <service_name>
docker-compose down && docker-compose up -d --build
```

**Encoder slow on first query**  
The model downloads ~90 MB from Hugging Face on first startup. Subsequent starts use the Docker layer cache. Wait ~60 s after `docker-compose up`.

**Queries time out (504)**  
Check that all five pipeline services are running:
```bash
docker-compose ps
docker-compose logs encoder_service
docker-compose logs retriever_service
docker-compose logs llm_generator_1
```

**Redis connection errors**
```bash
docker exec -it <project>-redis-1 redis-cli ping
# Expected: PONG
```

**Prometheus shows no targets**
```bash
# Targets page: http://localhost:9090/targets
# api_gateway should show state=UP
docker-compose restart prometheus
```

---

## Performance

Measured with `utils/load_tester.py` (50 concurrent requests) on a mid-range laptop:

| Metric | Value |
|--------|-------|
| Throughput | 3–6 req/s |
| P50 latency | 200–350 ms |
| P95 latency | 400–600 ms |
| P99 latency | 600–900 ms |
| Success rate | 100% |

*Bottleneck: sentence-transformer encoding on CPU. Swap `EMBEDDING_DEVICE=cuda` with a GPU for 10–20× speedup.*

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API framework | FastAPI + Uvicorn + Gunicorn |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector DB | Qdrant |
| Message queue | Redis |
| Monitoring | Prometheus |
| UI | Streamlit |
| Load testing | Locust + httpx |
| Containerization | Docker + Docker Compose |

---

## License

MIT License — open source, use freely.
