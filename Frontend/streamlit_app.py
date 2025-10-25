# streamlit_app.py
import streamlit as st
import requests

# Backend endpoints
BACKEND_URL_JSON = "http://localhost:8000/api/query"
BACKEND_URL_MULTIPART = "http://localhost:8000/api/query-multipart"

# Page setup
st.set_page_config(page_title="RAG Optimization Chatbot", layout="wide")

# Header
st.markdown(
    """
    <style>
        .main-title {
            text-align: center;
            font-size: 2.2rem;
            font-weight: 700;
            color: #2C3E50;
            margin-bottom: 0.5rem;
        }
        .subtext {
            text-align: center;
            font-size: 1.1rem;
            color: #6C757D;
            margin-bottom: 2rem;
        }
        .stButton>button {
            background-color: #4F8BF9;
            color: white;
            font-weight: 600;
            border-radius: 8px;
            padding: 0.6em 1.2em;
        }
        .stButton>button:hover {
            background-color: #3B6ED6;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='main-title'>🤖 RAG-Optimization Chatbot — Demo UI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtext'>Ask questions, upload documents, or share media for processing by the backend.</div>", unsafe_allow_html=True)

# Input Form
with st.form("input_form"):
    st.markdown("### ✍️ Input Section")
    text_input = st.text_area("Enter your question or prompt:", height=120, placeholder="Type your query here...")

    uploaded_files = st.file_uploader(
        "Upload file(s): document (.pdf, .txt), image (.png, .jpg, .jpeg), or video (.mp4, .webm)",
        type=["pdf", "txt", "png", "jpg", "jpeg", "mp4", "webm"],
        accept_multiple_files=True
    )

    submit = st.form_submit_button("🚀 Send to Backend")

# --- Processing ---
if submit:
    st.info("⏳ Sending your request to the backend...")

    if uploaded_files:
        files = {}
        data = {"text": text_input} if text_input else {}

        for i, f in enumerate(uploaded_files):
            files[f"file_{i}"] = (f.name, f.getvalue())

        try:
            resp = requests.post(BACKEND_URL_MULTIPART, data=data, files=files, timeout=60)
            st.success("✅ Backend responded successfully!")
            st.json(resp.json())
        except Exception as e:
            st.error(f"⚠️ Error contacting backend: {e}")
    else:
        payload = {"text": text_input}
        try:
            resp = requests.post(BACKEND_URL_JSON, json=payload, timeout=60)
            st.success("✅ Backend responded successfully!")
            st.json(resp.json())
        except Exception as e:
            st.error(f"⚠️ Error contacting backend: {e}")

# --- Preview Section ---
if uploaded_files:
    st.markdown("### 👀 Uploaded File Previews")
    for f in uploaded_files:
        if f.type.startswith("image/"):
            st.image(f, caption=f.name)
        elif f.type.startswith("video/"):
            st.video(f)
        else:
            st.write(f"📄 {f.name}")

# Footer
st.markdown("---")
st.caption("🧠 Running locally on port 8501 | Backend API on port 8000 | Metrics enabled via Prometheus")
