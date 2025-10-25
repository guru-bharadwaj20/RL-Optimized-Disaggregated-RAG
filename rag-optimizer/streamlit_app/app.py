import streamlit as st
from PIL import Image
import io

st.set_page_config(page_title="RAG Optimizer Chatbot", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.title("📊 RAG Optimizer Dashboard")
    st.markdown("Monitor and interact with your optimized RAG system.")
    st.divider()
    st.markdown("**Chat Mode:** Text, Files, or Multimedia inputs")

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
        align-self: flex-end;
    }
    .bot-msg {
        background-color: #E8E8E8;
        padding: 10px 15px;
        border-radius: 15px;
        margin-bottom: 10px;
        width: fit-content;
        max-width: 80%;
        align-self: flex-start;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("💬 RAG Optimizer Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

col1, col2 = st.columns([8, 1])

with col1:
    user_input = st.text_input(
        "Type your message...",
        placeholder="Ask me anything about RAG optimization...",
        key="user_input",
        label_visibility="collapsed",
    )
with col2:
    submitted = st.button("➤", use_container_width=True)

uploaded_file = st.file_uploader("➕", type=["txt", "pdf", "png", "jpg", "jpeg", "mp4", "avi"], label_visibility="collapsed")

if submitted or uploaded_file:
    if user_input or uploaded_file:
        message = user_input if user_input else "📎 Uploaded a file"
        st.session_state.chat_history.append({"role": "user", "content": message})

        # Placeholder for backend response
        bot_response = "🤖 This is where your backend RAG logic will generate the optimized response."
        st.session_state.chat_history.append({"role": "bot", "content": bot_response})

# Display chat messages
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg">{msg["content"]}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
