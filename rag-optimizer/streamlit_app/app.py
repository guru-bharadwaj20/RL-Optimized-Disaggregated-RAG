import os
import time
import json
import requests
import streamlit as st

API_GATEWAY_URL = os.getenv("API_GATEWAY_URL", "http://localhost:8000")

st.set_page_config(
    page_title="RL-Optimized RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Session State Defaults ────────────────────────────────────────────────────
for key, default in {
    "messages": [],
    "total_queries": 0,
    "total_latency_ms": 0.0,
    "replica_counts": {},
    "cache_hit_count": 0,
    "semantic_hit_count": 0,
    "health": None,
    "health_ts": 0.0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def get_health(backend_url: str, force: bool = False) -> dict:
    now = time.time()
    if not force and st.session_state.health and (now - st.session_state.health_ts) < 10:
        return st.session_state.health
    try:
        r = requests.get(f"{backend_url}/health", timeout=3)
        st.session_state.health = r.json() if r.status_code == 200 else {"status": f"HTTP {r.status_code}"}
    except Exception as e:
        st.session_state.health = {"status": "unreachable", "error": str(e)}
    st.session_state.health_ts = now
    return st.session_state.health


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 RL-Optimized RAG")
    st.caption("Production-grade disaggregated pipeline")

    backend_url = st.text_input("API Gateway URL", value=API_GATEWAY_URL, key="backend_url_input")

    # Health status
    health = get_health(backend_url)
    redis_ok = health.get("redis") == "connected"
    qdrant_ok = health.get("qdrant") == "connected"
    sem_cache_on = health.get("semantic_cache") == "enabled"
    exact_cache_on = health.get("exact_cache") == "enabled"

    st.subheader("System Status")
    col1, col2 = st.columns(2)
    col1.metric("Redis", "✅" if redis_ok else "❌")
    col2.metric("Qdrant", "✅" if qdrant_ok else "❌")
    col1.metric("Exact Cache", "ON" if exact_cache_on else "OFF")
    col2.metric("Semantic Cache", "ON" if sem_cache_on else "OFF")

    if st.button("Refresh status"):
        get_health(backend_url, force=True)
        st.rerun()

    st.divider()
    st.subheader("Session Stats")

    n = st.session_state.total_queries
    avg_lat = st.session_state.total_latency_ms / n if n > 0 else 0
    hit_rate = (st.session_state.cache_hit_count + st.session_state.semantic_hit_count) / n if n > 0 else 0

    m1, m2 = st.columns(2)
    m1.metric("Queries", n)
    m2.metric("Avg Latency", f"{avg_lat:.0f} ms")
    m1.metric("Cache Hits", st.session_state.cache_hit_count)
    m2.metric("Semantic Hits", st.session_state.semantic_hit_count)
    if n > 0:
        st.metric("Cache Hit Rate", f"{hit_rate:.0%}")

    if st.session_state.replica_counts:
        st.markdown("**Replica distribution**")
        for replica, count in sorted(st.session_state.replica_counts.items()):
            pct = count / max(n, 1)
            st.progress(pct, text=f"{replica}: {count} ({pct:.0%})")

    st.divider()
    st.subheader("Pipeline Architecture")
    st.markdown(
        """
        ```
        Query
          ↓ Exact Cache (Redis)
          ↓ Semantic Cache (Qdrant)
          ↓ Encoder (MiniLM-L6-v2)
          ↓ Retriever (Qdrant search)
          ↓ LLM Generator
          ↓ Ollama / Mock LLM
          ↓ Answer
        ```
        """
    )

    st.divider()
    if st.button("Clear chat history"):
        for k in ["messages", "total_queries", "total_latency_ms",
                  "replica_counts", "cache_hit_count", "semantic_hit_count"]:
            st.session_state[k] = [] if k == "messages" else ({} if k == "replica_counts" else 0.0 if k == "total_latency_ms" else 0)
        st.rerun()

    st.subheader("Try asking:")
    st.markdown(
        "- What is RAG?\n"
        "- How does vector search work?\n"
        "- Explain Redis message queues\n"
        "- What is load balancing?\n"
        "- How does caching reduce latency?\n"
        "- What is the CAP theorem?\n"
        "- Explain Kubernetes\n"
        "- What are embeddings?"
    )


# ── Main Area ─────────────────────────────────────────────────────────────────
tab_chat, tab_upload, tab_metrics = st.tabs(["💬 Chat", "📄 Upload Documents", "📊 Metrics"])


# ── TAB 1: Chat ───────────────────────────────────────────────────────────────
with tab_chat:
    st.header("RAG Chat Interface")

    for msg in st.session_state.messages:
        avatar = "🧑" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

            if msg["role"] == "assistant" and "meta" in msg:
                meta = msg["meta"]
                cs = meta.get("cache_status", "miss")
                cache_badge = "✅ Cache Hit" if cs == "hit" else ("🧠 Semantic Hit" if cs == "semantic_hit" else "❌ Cache Miss")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Latency", f"{meta.get('latency_ms', 0):.0f} ms")
                c2.metric("Replica", meta.get("selected_replica", "—"))
                c3.metric("Cache", cache_badge)
                c4.metric("LLM", meta.get("llm_backend", "—"))

                chunks = meta.get("retrieved_chunks", [])
                if chunks:
                    with st.expander(f"📚 Retrieved chunks ({len(chunks)})"):
                        for i, chunk in enumerate(chunks):
                            score = chunk.get("score", 0)
                            source = chunk.get("source", "corpus")
                            idx = chunk.get("chunk_index", 0)
                            text = chunk.get("text", "")
                            st.markdown(
                                f"**Chunk {i+1}** — source: `{source}` | chunk #{idx} | "
                                f"similarity: `{score:.4f}`"
                            )
                            st.caption(text)
                            if i < len(chunks) - 1:
                                st.divider()

    query = st.chat_input("Ask something about distributed systems, RAG, caching…")

    if query:
        with st.chat_message("user", avatar="🧑"):
            st.markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Running through RAG pipeline…"):
                try:
                    t0 = time.time()
                    resp = requests.post(
                        f"{backend_url}/query",
                        json={"query": query},
                        timeout=70,
                    )
                    client_ms = (time.time() - t0) * 1000

                    if resp.status_code == 200:
                        data = resp.json()
                        answer = data.get("answer", "No answer returned.")
                        latency_ms = data.get("latency_ms", client_ms)
                        replica = data.get("selected_replica", "unknown")
                        job_id = data.get("job_id", "")
                        cache_status = data.get("cache_status", "miss")
                        chunks = data.get("retrieved_chunks", [])
                        llm_backend = data.get("llm_backend", "unknown")

                        st.markdown(answer)

                        cs_badge = "✅ Cache Hit" if cache_status == "hit" else (
                            "🧠 Semantic Hit" if cache_status == "semantic_hit" else "❌ Cache Miss"
                        )
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Latency", f"{latency_ms:.0f} ms")
                        c2.metric("Replica", replica)
                        c3.metric("Cache", cs_badge)
                        c4.metric("LLM", llm_backend)

                        if chunks:
                            with st.expander(f"📚 Retrieved chunks ({len(chunks)})"):
                                for i, chunk in enumerate(chunks):
                                    score = chunk.get("score", 0)
                                    source = chunk.get("source", "corpus")
                                    idx = chunk.get("chunk_index", 0)
                                    text = chunk.get("text", "")
                                    st.markdown(
                                        f"**Chunk {i+1}** — `{source}` | chunk #{idx} | sim: `{score:.4f}`"
                                    )
                                    st.caption(text)
                                    if i < len(chunks) - 1:
                                        st.divider()

                        # Update session stats
                        st.session_state.total_queries += 1
                        st.session_state.total_latency_ms += latency_ms
                        st.session_state.replica_counts[replica] = st.session_state.replica_counts.get(replica, 0) + 1
                        if cache_status == "hit":
                            st.session_state.cache_hit_count += 1
                        elif cache_status == "semantic_hit":
                            st.session_state.semantic_hit_count += 1

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "meta": {
                                "latency_ms": latency_ms,
                                "selected_replica": replica,
                                "job_id": job_id,
                                "cache_status": cache_status,
                                "retrieved_chunks": chunks,
                                "llm_backend": llm_backend,
                            },
                        })

                    elif resp.status_code == 504:
                        st.error("Pipeline timed out (60s). Check `docker-compose ps` for unhealthy services.")
                        st.session_state.messages.append({"role": "assistant", "content": "Pipeline timed out."})
                    else:
                        detail = resp.json().get("detail", resp.text)
                        st.error(f"API error {resp.status_code}: {detail}")
                        st.session_state.messages.append({"role": "assistant", "content": f"Error: {detail}"})

                except requests.exceptions.ConnectionError:
                    msg = f"Cannot reach {backend_url}. Run `docker-compose up -d` first."
                    st.error(msg)
                    st.session_state.messages.append({"role": "assistant", "content": msg})
                except requests.exceptions.Timeout:
                    st.error("Request timed out (70s).")
                    st.session_state.messages.append({"role": "assistant", "content": "Request timed out."})
                except Exception as e:
                    st.error(f"Unexpected error: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})

        st.rerun()


# ── TAB 2: Upload Documents ───────────────────────────────────────────────────
with tab_upload:
    st.header("Upload Documents to Knowledge Base")
    st.markdown(
        "Upload **PDF**, **TXT**, or **Markdown** files. "
        "Documents are chunked (500 chars / 100 overlap), embedded, and stored in Qdrant immediately."
    )

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "txt", "md"],
        help="PDF pages are extracted via pypdf. TXT/MD files are read as-is.",
    )

    col_l, col_r = st.columns([3, 1])
    custom_source = col_l.text_input(
        "Source label (optional)",
        placeholder="e.g. my-paper.pdf",
        help="Used as metadata in Qdrant. Defaults to filename.",
    )
    chunk_preview = col_r.number_input("Preview chunks", min_value=0, max_value=20, value=3)

    if uploaded_file is not None:
        source_name = custom_source.strip() or uploaded_file.name
        file_type = uploaded_file.type or ""

        st.info(f"File: **{uploaded_file.name}** ({uploaded_file.size:,} bytes) → source: `{source_name}`")

        text_content = None

        # Extract text
        if "pdf" in file_type or uploaded_file.name.lower().endswith(".pdf"):
            try:
                import pypdf
                reader = pypdf.PdfReader(uploaded_file)
                pages_text = []
                for pg in reader.pages:
                    extracted = pg.extract_text()
                    if extracted:
                        pages_text.append(extracted)
                text_content = "\n\n".join(pages_text)
                st.success(f"Extracted text from {len(reader.pages)} PDF pages ({len(text_content):,} chars)")
            except ImportError:
                st.error("pypdf not installed in the container. Rebuild with `docker-compose up -d --build`.")
            except Exception as e:
                st.error(f"PDF extraction failed: {e}")
        else:
            try:
                text_content = uploaded_file.read().decode("utf-8", errors="replace")
                st.success(f"Read {len(text_content):,} characters from {uploaded_file.name}")
            except Exception as e:
                st.error(f"File read error: {e}")

        if text_content and chunk_preview > 0:
            st.subheader("Text preview (first 500 chars)")
            st.text(text_content[:500])

        if text_content:
            if st.button("📥 Ingest into Knowledge Base", type="primary"):
                with st.spinner("Chunking → embedding → storing in Qdrant…"):
                    try:
                        t0 = time.time()
                        resp = requests.post(
                            f"{backend_url}/ingest",
                            json={"text": text_content, "source": source_name},
                            timeout=180,
                        )
                        elapsed = (time.time() - t0) * 1000
                        if resp.status_code == 200:
                            data = resp.json()
                            st.success(
                                f"✅ Ingested **{data['chunks_added']} chunks** from `{data['source']}` "
                                f"in {data['processing_time_ms']:.0f} ms"
                            )
                            st.balloons()
                        else:
                            detail = resp.json().get("detail", resp.text)
                            st.error(f"Ingestion failed ({resp.status_code}): {detail}")
                    except requests.exceptions.ConnectionError:
                        st.error(f"Cannot reach {backend_url}.")
                    except Exception as e:
                        st.error(f"Ingestion error: {e}")

    st.divider()
    st.subheader("Ingestion Tips")
    st.markdown(
        """
        - **Re-ingesting** the same file (same source name) safely overwrites previous chunks using deterministic IDs.
        - After ingesting, ask a question in the **Chat** tab — the pipeline will retrieve your uploaded content.
        - The seed corpus (`data/corpus.txt`) is ingested automatically on first startup via `utils/ingestion.py`.
        - For large PDFs (>50 pages) the ingestion may take 30–90 seconds.
        """
    )


# ── TAB 3: Metrics ────────────────────────────────────────────────────────────
with tab_metrics:
    st.header("Pipeline Metrics")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Prometheus Queries")
        st.markdown(
            """
            Open **http://localhost:9090** and run:

            | Metric | PromQL |
            |--------|--------|
            | Request rate | `rate(api_gateway_queries_total[30s])` |
            | P95 latency | `histogram_quantile(0.95, rate(api_gateway_query_latency_seconds_bucket[30s]))` |
            | Cache hit rate | `cache_hits_total / (cache_hits_total + cache_misses_total)` |
            | Active jobs | `api_gateway_active_jobs` |
            | Worker load | `api_gateway_replica_load` |
            | Semantic hits | `semantic_cache_hits_total` |
            """
        )

    with col_b:
        st.subheader("Raw Metrics Endpoint")
        if st.button("Fetch /metrics"):
            try:
                r = requests.get(f"{backend_url}/metrics", timeout=5)
                if r.status_code == 200:
                    lines = [l for l in r.text.splitlines() if not l.startswith("#") and l.strip()]
                    st.code("\n".join(lines[:60]), language="text")
                else:
                    st.error(f"HTTP {r.status_code}")
            except Exception as e:
                st.error(f"Could not fetch metrics: {e}")

        st.subheader("Health Check")
        if st.button("Fetch /health"):
            h = get_health(backend_url, force=True)
            st.json(h)
