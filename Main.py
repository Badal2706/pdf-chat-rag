import streamlit as st
import os
from groq import Groq
import PyPDF2
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import uuid
from dotenv import load_dotenv
import re

# Load .env file
load_dotenv()

# Page configuration
st.set_page_config(page_title="Chat with PDFs", layout="wide", initial_sidebar_state="expanded")


# Cache the client so it doesn't recreate on every message
@st.cache_resource
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("Please create a .env file with GROQ_API_KEY=your_key")
        st.stop()
    return Groq(api_key=api_key)


def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file with error handling"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        # Clean invalid Unicode characters
        text = clean_text(text)
        return text

    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""


def clean_text(text):
    """Remove invalid Unicode surrogates and normalize text"""
    if not text:
        return ""

    # Remove surrogate pairs (invalid Unicode)
    # Surrogates are in range U+D800 to U+DFFF
    text = text.encode('utf-8', 'surrogatepass').decode('utf-8', 'ignore')

    # Alternative: Replace common problematic characters
    replacements = {
        '\x00': '',  # Null bytes
        '\x0b': ' ',  # Vertical tab
        '\x0c': ' ',  # Form feed
        '\ufffd': '',  # Replacement character (�)
        '•': '-',  # Bullet to dash
        '“': '"',  # Smart quotes to regular
        '”': '"',
        '‘': "'",
        '’': "'",
        '–': '-',  # En-dash to regular dash
        '—': '-',  # Em-dash to regular dash
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove any remaining non-printable characters except newlines
    text = ''.join(char for char in text if char.isprintable() or char in '\n\r\t')

    return text.strip()

def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks"""
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk)
        start += chunk_size - overlap
        if end == text_len:
            break
    return chunks


# Initialize session states
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'pdf_chunks' not in st.session_state:
    st.session_state.pdf_chunks = []

if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
    st.session_state.chunk_vectors = None

if 'pdf_list' not in st.session_state:
    st.session_state.pdf_list = []

# Sidebar
with st.sidebar:
    st.title("📄 PDF Upload")

    # Check for API key
    if not os.getenv("GROQ_API_KEY"):
        st.error("⚠️ No .env file found or GROQ_API_KEY not set")
        st.info("Create a .env file with: GROQ_API_KEY=your_key_here")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Process PDFs", type="primary"):
            with st.spinner("Processing..."):
                all_chunks = []
                pdf_names = []

                for pdf_file in uploaded_files:
                    text = extract_text_from_pdf(pdf_file)
                    if text.strip():
                        chunks = chunk_text(text)
                        for chunk in chunks:
                            all_chunks.append({
                                "text": chunk,
                                "source": pdf_file.name,
                                "id": str(uuid.uuid4())
                            })
                        pdf_names.append(pdf_file.name)
                    else:
                        st.warning(f"Could not extract text from {pdf_file.name}")

                if all_chunks:
                    st.session_state.pdf_chunks = all_chunks
                    st.session_state.pdf_list = pdf_names

                    texts = [chunk["text"] for chunk in all_chunks]
                    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
                    chunk_vectors = vectorizer.fit_transform(texts)

                    st.session_state.vectorizer = vectorizer
                    st.session_state.chunk_vectors = chunk_vectors
                    st.success(f"✅ Loaded {len(pdf_names)} PDFs ({len(all_chunks)} chunks)")
                else:
                    st.error("No text extracted")

    if st.session_state.pdf_list:
        st.markdown("### 📚 Loaded Documents")
        for pdf in st.session_state.pdf_list:
            st.markdown(f"- {pdf}")

        if st.button("Clear All PDFs"):
            st.session_state.pdf_chunks = []
            st.session_state.pdf_list = []
            st.session_state.vectorizer = None
            st.session_state.chunk_vectors = None
            st.rerun()

    st.markdown("---")
    model_choice = st.selectbox(
        "Model",
        ["openai/gpt-oss-20b"]
    )

    retrieval_k = st.slider("Context chunks", 1, 10, 5)

# Main Interface
st.title("💬 Chat with your PDFs")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📎 Sources"):
                for source in message["sources"]:
                    st.markdown(f"**{source}**")

# Chat input
if prompt := st.chat_input("Ask something about your PDFs..."):
    if not st.session_state.pdf_chunks:
        st.error("Please upload PDFs first!")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve relevant chunks
    query_vec = st.session_state.vectorizer.transform([prompt])
    similarities = cosine_similarity(query_vec, st.session_state.chunk_vectors).flatten()
    top_indices = similarities.argsort()[-retrieval_k:][::-1]
    relevant_chunks = [st.session_state.pdf_chunks[i] for i in top_indices]

    context_text = "\n\n".join([
        f"[From {chunk['source']}]: {chunk['text']}"
        for chunk in relevant_chunks
    ])
    sources = list(set([chunk["source"] for chunk in relevant_chunks]))

    # Prepare messages
    system_prompt = f"""Analyze the following PDF documents and answer the question.

    GUIDELINES:
    - Primary answer must come from: {context_text}
    - If the PDF mentions "creatinine" but doesn't explain what it is, you may add: "(Creatinine is a waste product filtered by kidneys)"
    - Clearly distinguish: "According to the report..." vs "Generally speaking..."
    - If completely absent from PDF, say: "This information is not in the uploaded documents"""

    api_messages = [{"role": "system", "content": system_prompt}]
    for msg in st.session_state.messages[-10:]:
        api_messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    # Get response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            client = get_groq_client()
            stream = client.chat.completions.create(
                model=model_choice,
                messages=api_messages,
                temperature=0.7,
                max_tokens=2048,
                stream=True
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": sources
            })

        except Exception as e:
            st.error(f"Error: {str(e)}")

if st.session_state.messages:
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

if not st.session_state.pdf_list:
    st.info("👈 Upload PDFs using the sidebar to begin")