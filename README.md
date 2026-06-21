# RL-Optimized Disaggregated RAG

A production-grade, **fully disaggregated Retrieval-Augmented Generation** system with real embeddings, optional Ollama LLM, dual-layer caching, adaptive load balancing, live document ingestion, and a Streamlit transparency dashboard.

---

## Architecture

```
                          ┌───────────────────────────────────────┐
                          │           Streamlit UI  :8501          │
                          │  Chat │ Upload Docs │ Metrics          │
                          └──────────────┬────────────────────────┘
                                         │ HTTP
                          ┌──────────────▼────────────────────────┐
                          │        API Gateway  :8000              │
                          │                                        │
                          │  1. Exact Cache  (Redis MD5)           │
                          │  2. Semantic Cache (Qdrant cos>=0.95)  │
                          │  3. Adaptive Load Balancer (EMA)       │
                          │  4. Prometheus /metrics                │
                          └───┬─────────────┬──────────────┬──────┘
                              │             │              │
               ┌──────────────▼──┐   ┌──────▼──────┐  ┌──▼────────────────┐
               │ Encoder Service │   │  Retriever  │  │  LLM Generator    │
               │   :8001 (HTTP)  │   │   Worker    │  │  Worker x2        │
               │                 │   │             │  │                   │
               │ sentence-transf │   │ Qdrant ANN  │  │  Ollama (opt)     │
               │ MiniLM-L6-v2   │   │ k=3 cosine  │  │  -> Mock LLM      │
               │ /encode         │   │             │  │                   │
               │ /encode_batch   │   └──────┬──────┘  └──────────────────┘
               │ /ingest         │          │
               └────────┬────────┘          │
                        │                   │
               ┌────────▼───────────────────▼──────────────────────┐
               │                   Redis  :6379                     │
               │  Queues: encoder_in / retriever_in / llm_in        │
               │  Exact cache: cache:exact:{md5}  (TTL 3600s)       │
               │  LB state: llm_load_{idx}, llm_latency_ms_{idx}    │
               └───────────────────────────────────────────────────┘
               ┌───────────────────────────────────────────────────┐
               │                   Qdrant  :6333                   │
               │  rag_collection     — document vectors (dim 384)   │
               │  semantic_cache     — cached query vectors          │
               └───────────────────────────────────────────────────┘
               ┌───────────────────────────────────────────────────┐
               │               Prometheus  :9090                   │
               └───────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# 1. Clone
git clone <repo-url> && cd RL-Optimized-Disaggregated-RAG

# 2. Start all services
docker-compose up -d --build

# 3. Open the UI
open http://localhost:8501      # Mac
start http://localhost:8501     # Windows
```

First startup downloads the `all-MiniLM-L6-v2` embedding model (~80 MB) and ingests the seed corpus. Allow ~60 seconds for the encoder service to become healthy.

---

## Ollama (Real LLM)

By default the system uses the **Mock LLM** (keyword-based fast responses). To use a real local LLM:

### Option A — Docker profile (recommended)

```bash
# Start with Ollama sidecar
docker-compose --profile ollama up -d --build

# Pull the model inside the container (one-time, ~2 GB)
docker exec -it $(docker ps -qf name=ollama) ollama pull llama3.2

# Enable in the generator workers — edit docker-compose.yml, uncomment:
#   - OLLAMA_BASE_URL=http://ollama:11434
#   - OLLAMA_MODEL=llama3.2

# Recreate workers
docker-compose up -d --no-deps llm_generator_1 llm_generator_2
```

### Option B — Host Ollama

```bash
# Install Ollama: https://ollama.ai/download
ollama pull llama3.2

# Point workers at host
# In docker-compose.yml, set:
#   - OLLAMA_BASE_URL=http://host.docker.internal:11434
```

The workers **always fall back to Mock LLM** if Ollama is unreachable — zero downtime.

---

## Environment Variables

| Service | Variable | Default | Description |
|---------|----------|---------|-------------|
| `api_gateway` | `EXACT_CACHE_ENABLED` | `true` | Enable Redis exact cache |
| `api_gateway` | `EXACT_CACHE_TTL` | `3600` | Cache TTL in seconds |
| `api_gateway` | `SEMANTIC_CACHE_ENABLED` | `true` | Enable Qdrant semantic cache |
| `api_gateway` | `SEMANTIC_CACHE_THRESHOLD` | `0.95` | Cosine similarity threshold |
| `api_gateway` | `ENCODER_SERVICE_URL` | `http://encoder_service:8000` | Encoder HTTP endpoint |
| `encoder_service` | `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace model ID |
| `encoder_service` | `CHUNK_SIZE` | `500` | Characters per chunk |
| `encoder_service` | `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `llm_generator_*` | `OLLAMA_BASE_URL` | _(empty)_ | Ollama URL (disabled if empty) |
| `llm_generator_*` | `OLLAMA_MODEL` | `llama3.2` | Model to use |
| `llm_generator_*` | `USE_MOCK_FALLBACK` | `true` | Fallback to Mock LLM |
| All | `QDRANT_HOST` | `qdrant` | Qdrant hostname |
| All | `QDRANT_COLLECTION` | `rag_collection` | Main vector collection |

---

## Caching

### Exact Cache (Redis)

Every incoming question is normalized (lowercase, stripped) and hashed with MD5.
The hash is used as a Redis key (`cache:exact:{md5}`) with a 3600-second TTL.
An identical question asked twice never touches the pipeline.

```
Question -> normalize -> MD5 -> GET cache:exact:{hash}
    hit  -> return immediately  (cache_status: "hit")
    miss -> continue to pipeline
```

### Semantic Cache (Qdrant)

On a cache miss the question is embedded (MiniLM) and searched in the `semantic_cache`
Qdrant collection (cosine similarity). If score >= 0.95 the stored answer is returned
without invoking the full RAG pipeline.

```
Question -> embed -> Qdrant search semantic_cache (threshold 0.95)
    hit  -> return stored answer  (cache_status: "semantic_hit")
    miss -> full RAG pipeline     (cache_status: "miss")
```

Both caches are populated on every pipeline completion.

---

## Document Upload

Upload **PDF**, **TXT**, or **Markdown** files via the Streamlit **Upload Documents** tab.
Files are chunked with a RecursiveCharacterTextSplitter (500 chars, 100 overlap),
embedded with MiniLM, and upserted into Qdrant using deterministic SHA256 IDs — safe
to re-upload the same file.

Supported: `.pdf`, `.txt`, `.md`

The seed corpus (`data/corpus.txt`) covers RAG, vector search, Redis, Qdrant, FastAPI,
Prometheus, Kubernetes, CAP theorem, and more — enough to get meaningful answers immediately.

---

## Smart Chunking

Documents are split using a recursive character splitter that tries separators in order:

```
"\n\n"  ->  "\n"  ->  ". "  ->  "! "  ->  "? "  ->  "; "  ->  ", "  ->  " "  ->  ""
```

Each chunk is at most 500 characters with a 100-character overlap from the previous
chunk prepended. Qdrant point IDs are deterministic (`SHA256(source::chunk_index) % 2^63`)
so re-ingesting the same document overwrites existing chunks rather than duplicating them.

---

## Adaptive Load Balancer

The API Gateway selects which LLM worker replica to dispatch to using:

```
score(replica) = active_connections + (EMA_latency_ms / 1000)
```

EMA (Exponential Moving Average) latency is updated after every completed job with alpha=0.2.
The replica with the lowest score is selected. Load counters (`llm_load_{idx}`) and
latency (`llm_latency_ms_{idx}`) are stored in Redis for cross-process visibility.

---

## Prometheus Metrics

Open `http://localhost:9090` and query:

| Metric | PromQL |
|--------|--------|
| Request rate | `rate(api_gateway_queries_total[30s])` |
| P95 latency | `histogram_quantile(0.95, rate(api_gateway_query_latency_seconds_bucket[30s]))` |
| Exact cache hit rate | `cache_hits_total / (cache_hits_total + cache_misses_total)` |
| Semantic cache hits | `rate(semantic_cache_hits_total[1m])` |
| Active jobs | `api_gateway_active_jobs` |
| Worker load | `api_gateway_replica_load` |
| Worker avg latency | `worker_avg_latency_ms` |
| Ingested chunks | `ingestion_chunks_total` |

---

## Benchmarks

Run the benchmark suite against a live stack:

```bash
# Default: 20 questions x 3 scenarios x 3 worker counts, concurrency=4
python benchmarks/run_benchmark.py

# Custom
python benchmarks/run_benchmark.py \
    --url http://localhost:8000 \
    --requests 50 \
    --concurrency 8 \
    --workers 1,2,4

# If /admin/cache endpoint not present, skip toggle
python benchmarks/run_benchmark.py --skip-cache-toggle
```

Results are written to `benchmarks/results/`:

| File | Contents |
|------|----------|
| `report.md` | Markdown table — P50/P95/P99, RPS, cache hit % |
| `results.csv` | Per-request raw data |
| `summary.json` | Machine-readable aggregate metrics |

Scenarios compared:
- **Exact Cache OFF** — every request runs the full pipeline
- **Exact Cache ON** — repeated questions served from Redis
- **Semantic Cache ON** — near-duplicate questions served from Qdrant (threshold 0.95)

---

## Streamlit Dashboard

`http://localhost:8501`

| Tab | Features |
|-----|---------|
| **Chat** | Chat UI, chunk transparency (source/score/index), 4-metric row per answer |
| **Upload Documents** | PDF/TXT/MD uploader, source label, text preview, balloons on success |
| **Metrics** | Prometheus PromQL reference, /metrics fetcher, /health JSON viewer |
| **Sidebar** | Redis/Qdrant status, cache toggles status, session hit rate, replica distribution |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit 1.x |
| API | FastAPI + Uvicorn |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` (dim 384) |
| Vector DB | Qdrant (ANN cosine search) |
| Queues | Redis List queues |
| Exact Cache | Redis (MD5 key, TTL 3600s) |
| Semantic Cache | Qdrant `semantic_cache` collection |
| LLM (optional) | Ollama `llama3.2` |
| LLM (fallback) | Mock LLM (keyword-based FastAPI service) |
| Monitoring | Prometheus + prometheus-client |
| Containerisation | Docker Compose |

---

## Service Ports

| Service | Port | URL |
|---------|------|-----|
| Streamlit UI | 8501 | `http://localhost:8501` |
| API Gateway | 8000 | `http://localhost:8000` |
| Encoder Service | 8001 | `http://localhost:8001` |
| Mock LLM | 8888 | `http://localhost:8888` |
| Qdrant | 6333 | `http://localhost:6333` |
| Redis | 6379 | `redis://localhost:6379` |
| Prometheus | 9090 | `http://localhost:9090` |
| Ollama (opt) | 11434 | `http://localhost:11434` |

---

## API Reference

```bash
# Ask a question
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?"}' | jq .

# Ingest a document
curl -s -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "Your document text here.", "source": "my-doc.txt"}' | jq .

# Health check
curl -s http://localhost:8000/health | jq .

# Prometheus metrics
curl -s http://localhost:8000/metrics | grep -v "^#"
```

Response schema for `/query`:

```json
{
  "job_id": "abc123",
  "query": "What is RAG?",
  "answer": "Retrieval-Augmented Generation...",
  "latency_ms": 142.7,
  "selected_replica": "replica-1",
  "cache_status": "hit",
  "retrieved_chunks": [
    {"text": "...", "score": 0.921, "source": "corpus.txt", "chunk_index": 3}
  ],
  "llm_backend": "ollama:llama3.2"
}
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Encoder takes 60s to start | First-run model download — wait for health check to pass |
| All answers are generic | Corpus not ingested yet — check encoder logs for `/ingest` |
| `cache_status` always `"miss"` | Exact/semantic cache env vars may be `false` — check `docker-compose.yml` |
| Ollama answers not appearing | Check `OLLAMA_BASE_URL` is set and `ollama pull llama3.2` ran |
| `503 Service Unavailable` | Workers not yet ready — `docker-compose ps` to check health |
| Qdrant collection not found | Encoder startup runs collection creation — check encoder logs |
| Prometheus shows no data | Check `rag-optimizer/prometheus/prometheus.yml` has `api_gateway:8000` |

```bash
# View live logs
docker-compose logs -f api_gateway encoder_service llm_generator_1

# Check all service health
docker-compose ps

# Reset everything
docker-compose down -v && docker-compose up -d --build
```

---

## Project Structure

```
.
├── data/corpus.txt                     # Seed knowledge base
├── utils/ingestion.py                  # Standalone corpus ingestion script
├── benchmarks/
│   ├── run_benchmark.py                # Benchmark suite
│   └── results/                        # Generated output (CSV/JSON/MD)
├── services/
│   ├── api_gateway/app.py              # Main entry point + caching + LB
│   ├── encoder_service/app.py          # Embeddings + /ingest + chunking
│   ├── retriever_service/worker.py     # Qdrant ANN search worker
│   ├── llm_generator/worker.py         # Ollama + Mock LLM worker
│   └── mock_llm/app.py                 # Keyword-based fallback LLM
└── rag-optimizer/
    ├── streamlit_app/app.py            # 3-tab Streamlit UI
    └── prometheus/prometheus.yml       # Scrape config
```
