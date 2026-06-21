import os
import time
import requests
import streamlit as st

API_GATEWAY_URL = os.getenv("API_GATEWAY_URL", "http://localhost:8000")

st.set_page_config(
    page_title="RL-Optimized RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 RL-Optimized RAG")
    st.caption("Disaggregated microservices pipeline")
    st.divider()

    st.subheader("Pipeline Architecture")
    st.markdown(
        """
        ```
        Client Query
             ↓
        API Gateway  (port 8000)
             ↓ Redis Queue
        Encoder Service  (MiniLM-L6-v2)
             ↓ Redis Queue
        Retriever Service  (Qdrant)
             ↓ Redis Queue
        LLM Generator  (×2 workers)
             ↓ Redis Key
        API Gateway → Response
        ```
        """
    )
    st.divider()

    st.subheader("Session Stats")
    if "total_queries" not in st.session_state:
        st.session_state.total_queries = 0
    if "total_latency_ms" not in st.session_state:
        st.session_state.total_latency_ms = 0.0
    if "replica_counts" not in st.session_state:
        st.session_state.replica_counts = {}

    col_a, col_b = st.columns(2)
    col_a.metric("Queries", st.session_state.total_queries)
    avg_lat = (
        st.session_state.total_latency_ms / st.session_state.total_queries
        if st.session_state.total_queries > 0
        else 0
    )
    col_b.metric("Avg Latency", f"{avg_lat:.0f} ms")

    if st.session_state.replica_counts:
        st.markdown("**Replica distribution**")
        for replica, count in sorted(st.session_state.replica_counts.items()):
            pct = count / max(st.session_state.total_queries, 1) * 100
            st.progress(pct / 100, text=f"{replica}: {count} ({pct:.0f}%)")

    st.divider()

    backend_url = st.text_input("API Gateway URL", value=API_GATEWAY_URL)
    if st.button("Check health"):
        try:
            r = requests.get(f"{backend_url}/health", timeout=3)
            if r.status_code == 200:
                st.success("Backend healthy")
            else:
                st.warning(f"HTTP {r.status_code}")
        except Exception as e:
            st.error(f"Unreachable: {e}")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.session_state.total_latency_ms = 0.0
        st.session_state.replica_counts = {}
        st.rerun()

    st.divider()
    st.markdown(
        "**Try asking:**\n"
        "- What is RAG?\n"
        "- How does load balancing work?\n"
        "- Explain vector embeddings\n"
        "- What is Redis used for?\n"
        "- How does caching reduce latency?\n"
        "- What is the CAP theorem?\n"
        "- Explain Kubernetes"
    )

# ── Main chat area ────────────────────────────────────────────────────────────
st.header("💬 RAG Chat Interface")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "meta" in msg:
            meta = msg["meta"]
            c1, c2, c3 = st.columns(3)
            c1.metric("End-to-end latency", f"{meta['latency_ms']:.0f} ms")
            c2.metric("Replica used", meta["replica"])
            c3.metric("Job ID", meta["job_id"][:8] + "…")

# Chat input
query = st.chat_input("Ask something about distributed systems, RAG, caching…")

if query:
    # Show user message immediately
    with st.chat_message("user", avatar="🧑"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # Call backend and show assistant response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Processing through RAG pipeline…"):
            t0 = time.time()
            try:
                resp = requests.post(
                    f"{backend_url}/query",
                    json={"query": query},
                    timeout=70,
                )
                elapsed_client = (time.time() - t0) * 1000

                if resp.status_code == 200:
                    data = resp.json()
                    answer = data.get("answer", "No answer returned.")
                    latency_ms = data.get("latency_ms", elapsed_client)
                    replica = data.get("selected_replica", "unknown")
                    job_id = data.get("job_id", "")

                    st.markdown(answer)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("End-to-end latency", f"{latency_ms:.0f} ms")
                    c2.metric("Replica used", replica)
                    c3.metric("Job ID", job_id[:8] + "…" if job_id else "—")

                    # Update session stats
                    st.session_state.total_queries += 1
                    st.session_state.total_latency_ms += latency_ms
                    st.session_state.replica_counts[replica] = (
                        st.session_state.replica_counts.get(replica, 0) + 1
                    )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "meta": {
                            "latency_ms": latency_ms,
                            "replica": replica,
                            "job_id": job_id,
                        },
                    })

                elif resp.status_code == 504:
                    st.error("Pipeline timed out (60s). Is Docker Compose fully running?")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "Pipeline timed out. Please check that all services are running.",
                    })
                else:
                    detail = resp.json().get("detail", resp.text)
                    st.error(f"API error {resp.status_code}: {detail}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Error {resp.status_code}: {detail}",
                    })

            except requests.exceptions.ConnectionError:
                st.error(
                    f"Cannot reach {backend_url}. "
                    "Start the stack with `docker-compose up -d` and try again."
                )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Cannot connect to {backend_url}. Is Docker Compose running?",
                })
            except requests.exceptions.Timeout:
                st.error("Request timed out (70s). The pipeline may be overloaded.")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Request timed out.",
                })
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Unexpected error: {e}",
                })

    # Rerun to update sidebar stats
    st.rerun()
