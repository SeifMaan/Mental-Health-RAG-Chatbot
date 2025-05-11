import streamlit as st
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import json
from pathlib import Path
from sklearn.preprocessing import normalize

# Set page config
st.set_page_config(
    page_title="Mental Health RAG Chatbot", page_icon="🧠", layout="centered"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Load FAISS index and chunk texts
@st.cache_resource
def load_faiss_index():
    index = faiss.read_index("Data/index.faiss")
    with open("Data/chunks_only.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


# Load embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# Configure Gemini API
@st.cache_resource
def configure_gemini():
    genai.configure(api_key="GEMINI_API_KEY")
    return genai.GenerativeModel("gemini-2.0-flash")


# Search relevant chunks using FAISS
def get_relevant_chunks(query, embedder, index, chunks, top_k=3):
    query_vec = embedder.encode([query])
    query_vec = normalize(query_vec, norm="l2").astype(
        "float32"
    )  # ⬅️ Normalize for cosine
    D, I = index.search(query_vec, top_k)
    retrieved_chunks = [chunks[i] for i in I[0]]
    return retrieved_chunks


# Generate answer with Gemini
def generate_answer_with_gemini(query, context_chunks, model):
    context_text = "\n\n".join(context_chunks)
    prompt = f"""You are a helpful mental health assistant. Use the following context to answer the user's question:

Context:
{context_text}

Question: {query}

Answer in a compassionate, professional tone. If the question is outside the provided context, say "I'm sorry, I don't have enough information to answer that. Please consult a mental health professional for more guidance." Otherwise, provide a detailed answer:"""

    response = model.generate_content(prompt)
    return response.text.strip()


# Load all resources
try:
    index, chunks = load_faiss_index()
    embedder = load_embedding_model()
    model = configure_gemini()

    # Display chat interface
    st.title("🧠 Mental Health Information Assistant")
    st.caption(
        "A RAG-powered chatbot providing information about mental health conditions"
    )

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask about mental health..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get relevant chunks
        relevant_chunks = get_relevant_chunks(prompt, embedder, index, chunks)

        # Generate answer
        with st.spinner("Thinking..."):
            answer = generate_answer_with_gemini(prompt, relevant_chunks, model)

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(answer)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})

except Exception as e:
    st.error(f"An error occurred while loading the chatbot: {str(e)}")
    st.info(
        "Please ensure all data files (index.faiss, chunks_only.pkl) are in the Data directory."
    )

# Add sidebar with information
with st.sidebar:
    st.header("About This Chatbot")
    st.markdown(
        """
    This is a Retrieval-Augmented Generation (RAG) chatbot that provides information about mental health conditions.
    
    **How it works:**
    1. You ask a question about mental health
    2. The system finds relevant information from trusted sources
    3. A language model generates a helpful response
    
    **Limitations:**
    - Not a substitute for professional medical advice
    - Only answers questions within its knowledge base
    """
    )

    st.divider()
    st.markdown("**Example questions to try:**")
    st.markdown("- What is depression?")
    st.markdown("- What are the symptoms of anxiety?")
    st.markdown("- How can I help someone with mental health issues?")
