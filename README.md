# Cache-Me-If-You-Can

A high-performance **Retrieval-Augmented Generation (RAG)** system with microservices architecture, Redis-based message queuing, and built-in monitoring using Prometheus and load testing with Locust.

## 🚀 Features

- **Microservices Architecture**: Modular design with independent services (API Gateway, Encoder, Retriever, LLM Generator)
- **Vector Search**: Qdrant-powered semantic search for document retrieval
- **Message Queue**: Redis-based asynchronous job processing
- **Load Balancing**: Round-robin distribution across multiple LLM worker replicas
- **Monitoring**: Real-time metrics collection with Prometheus
- **Load Testing**: Locust integration for performance testing
- **Containerized**: Full Docker Compose orchestration

## 📋 Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       ▼
┌─────────────────┐     ┌─────────────┐
│  API Gateway    │────▶│   Redis     │
│  (Port 8000)    │     │  (Queue)    │
└─────────────────┘     └──────┬──────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   Encoder    │      │  Retriever   │      │ LLM Worker 1 │
│   Service    │      │   Service    │      │ LLM Worker 2 │
└──────────────┘      └──────┬───────┘      └──────┬───────┘
                             │                      │
                             ▼                      ▼
                      ┌──────────────┐      ┌──────────────┐
                      │   Qdrant     │      │  Mock LLM    │
                      │  (Vectors)   │      │  Service     │
                      └──────────────┘      └──────────────┘
```

## 🛠️ Prerequisites

- Docker & Docker Compose
- Python 3.11+ (for local development)
- 8GB+ RAM recommended

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Seaweed-Boi/Cache-Me-If-You-Can.git
cd Cache-Me-If-You-Can
```

### 2. Start Services
```bash
docker-compose up -d
```

This will start:
- **Redis** (port 6379) - Message queue
- **Qdrant** (port 6333) - Vector database
- **API Gateway** (port 8000) - Main entry point
- **Encoder Service** - Text encoding
- **Retriever Service** - Vector search
- **LLM Generator** (2 replicas) - Answer generation
- **Mock LLM** (port 8888) - Simulated LLM
- **Prometheus** (port 9090) - Metrics collection

### 3. Verify Services
```bash
docker-compose ps
```

All services should show status `Up`.

### 4. Initialize Vector Database
```bash
python utils/ingestion.py
```

This ingests documents from `data/corpus.txt` into Qdrant.

## 🎯 Usage

### Query the RAG System

**Using curl:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is microservices?"}'
```

**Using Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/query",
    json={"question": "What is caching?"}
)

print(response.json())
```

**Response Format:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "answer": "Caching is a technique used to store frequently accessed data...",
  "retrieved_docs": ["doc1.txt", "doc2.txt"],
  "latency_ms": 267
}
```

## 📊 Monitoring & Testing

### Prometheus Metrics

**Access Prometheus UI:**
```bash
open http://localhost:9090
```

**Key Metrics Queries:**

**Total Queries:**
```promql
api_gateway_queries_total
```

**Requests Per Second:**
```promql
rate(api_gateway_queries_total[30s])
```

**95th Percentile Latency:**
```promql
histogram_quantile(0.95, rate(api_gateway_query_latency_seconds_bucket[30s]))
```

**Average Latency:**
```promql
rate(api_gateway_query_latency_seconds_sum[30s]) / rate(api_gateway_query_latency_seconds_count[30s])
```

### Load Testing with Locust

**1. Set up Python environment:**
```bash
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
pip install locust
```

**2. Start Locust:**
```bash
cd rag-optimizer/locust
locust -f locustfile.py --host=http://localhost:8000 --web-port 8090
```

**3. Open Locust UI:**
```bash
open http://localhost:8090
```

**4. Configure Test:**
- Number of users: `20`
- Spawn rate: `5` users/second
- Click "Start swarming"

**5. Monitor in Real-Time:**
- Watch metrics in Locust dashboard
- Query Prometheus for system metrics
- Observe performance under load

## 🔧 Configuration

### Environment Variables

Create `.env` file in project root (optional):
```bash
# Redis
REDIS_HOST=redis
REDIS_PORT=6379

# Qdrant
QDRANT_HOST=qdrant
QDRANT_PORT=6333

# LLM Service
LLM_SERVICE_URL=http://mock_llm:8888
```

### Scaling Workers

Edit `docker-compose.yml` to add more LLM workers:
```yaml
llm_generator_3:
  build: ./services/llm_generator
  environment:
    - REPLICA_ID=replica-3
  # ... rest of config
```

Update `LLM_REPLICAS` in `services/api_gateway/app.py`:
```python
LLM_REPLICAS = ["replica-1", "replica-2", "replica-3"]
```

## 📁 Project Structure

```
Cache-Me-If-You-Can/
├── docker-compose.yml           # Main orchestration
├── data/
│   └── corpus.txt              # RAG knowledge base
├── services/
│   ├── api_gateway/           # Entry point service
│   ├── encoder_service/       # Text encoding
│   ├── retriever_service/     # Vector search
│   ├── llm_generator/         # Answer generation workers
│   └── mock_llm/              # Simulated LLM
├── rag-optimizer/
│   ├── locust/
│   │   └── locustfile.py      # Load test scenarios
│   └── prometheus/
│       └── prometheus.yml      # Metrics config
└── utils/
    ├── ingestion.py           # Data ingestion script
    └── load_tester.py         # Testing utilities
```

## 🐛 Troubleshooting

### Services Won't Start
```bash
# Check logs
docker-compose logs -f [service_name]

# Rebuild containers
docker-compose down
docker-compose up -d --build
```

### Redis Connection Issues
```bash
# Check Redis is running
docker-compose ps redis

# Test connection
docker exec -it cache-me-if-you-can-redis-1 redis-cli ping
```

### Empty Responses
```bash
# Check if data is ingested
python utils/ingestion.py

# Verify Qdrant has vectors
curl http://localhost:6333/collections/documents
```

### Prometheus Not Scraping
```bash
# Restart Prometheus
docker-compose restart prometheus

# Check targets
curl http://localhost:9090/api/v1/targets | jq
```

## 📈 Performance Benchmarks

Tested on MacBook Air M2 (8GB RAM):

| Metric | Value |
|--------|-------|
| Throughput | 4-5 req/s |
| Avg Latency | 260-280ms |
| P95 Latency | 270-300ms |
| P99 Latency | 500-550ms |
| Success Rate | 100% |

*Load test: 10 concurrent users, 60s duration*

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Qdrant** - Vector database
- **Redis** - Message queue
- **Prometheus** - Monitoring
- **Locust** - Load testing
- **FastAPI** - Web framework

## 📞 Contact

**Project Repository:** [Cache-Me-If-You-Can](https://github.com/Seaweed-Boi/Cache-Me-If-You-Can)

---

**Built with ❤️ for high-performance RAG systems**