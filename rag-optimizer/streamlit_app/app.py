import streamlit as st
from PIL import Image
import io
import requests
import json
import time

# Configuration
API_GATEWAY_URL = "http://localhost:8000"

st.set_page_config(page_title="RAG Optimizer Chatbot", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.title("📊 RAG Optimizer Dashboard")
    st.markdown("Monitor and interact with your optimized RAG system.")
    st.divider()
    st.markdown("**Chat Mode:** Text inputs")
    st.markdown(f"**Backend:** {API_GATEWAY_URL}")
    
    # Add controls
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# --- Main Chat Interface ---
st.markdown(
    """
    <style>
    .chat-container {
        max-width: 800px;
        margin: auto;
        padding: 1.5rem;
        border-radius: 12px;
        background-color: #f9fafb;
        box-shadow: 0 0 15px rgba(0,0,0,0.05);
    }
    .user-msg {
        background-color: #DCF8C6;
        padding: 10px 15px;
        border-radius: 15px;
        margin-bottom: 10px;
        width: fit-content;
        max-width: 80%;
        margin-left: auto;
    }
    .bot-msg {
        background-color: #E8E8E8;
        padding: 10px 15px;
        border-radius: 15px;
        margin-bottom: 10px;
        width: fit-content;
        max-width: 80%;
    }
    .metadata {
        font-size: 0.75rem;
        color: #666;
        margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("💬 RAG Optimizer Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input form
with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([8, 1])
    with col1:
        user_input = st.text_input(
            "Type your message...",
            placeholder="Ask me anything about RAG optimization...",
            label_visibility="collapsed",
        )
    with col2:
        submitted = st.form_submit_button("➤", use_container_width=True)

if submitted and user_input:
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # Call backend API
    with st.spinner("🤖 Thinking..."):
        try:
            start_time = time.time()
            response = requests.post(
                f"{API_GATEWAY_URL}/query",
                json={"query": user_input},
                timeout=65
            )
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                bot_response = data.get("answer", "No response received")
                job_id = data.get("job_id", "")
                latency_ms = data.get("latency_ms", (end_time - start_time) * 1000)
                selected_replica = data.get("selected_replica", "unknown")
                
                st.session_state.chat_history.append({
                    "role": "bot",
                    "content": bot_response,
                    "job_id": job_id,
                    "latency_ms": latency_ms,
                    "replica": selected_replica
                })
            else:
                error_msg = f"API Error: {response.status_code} - {response.text}"
                st.session_state.chat_history.append({
                    "role": "bot",
                    "content": f"❌ {error_msg}"
                })
        except requests.exceptions.Timeout:
            st.session_state.chat_history.append({
                "role": "bot",
                "content": "❌ Request timed out. The backend may be overloaded."
            })
        except requests.exceptions.ConnectionError:
            st.session_state.chat_history.append({
                "role": "bot",
                "content": f"❌ Cannot reach backend at {API_GATEWAY_URL}. Is Docker Compose running?"
            })
        except Exception as e:
            st.session_state.chat_history.append({
                "role": "bot",
                "content": f"❌ Error: {str(e)}"
            })

# Display chat messages
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        content = msg["content"]
        st.markdown(f'<div class="bot-msg">{content}', unsafe_allow_html=True)
        if "latency_ms" in msg:
            st.markdown(
                f'<div class="metadata">⏱️ {msg["latency_ms"]:.0f}ms | 🎯 {msg["replica"]}</div>',
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

