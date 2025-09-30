import streamlit as st
import faiss
import pickle
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# 🔹 Load all models and data
@st.cache_resource
def load_models():
    # Load embedding model
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # ✅ Build correct absolute paths (works in Streamlit Cloud)
    base_path = os.path.dirname(__file__)
    index_path = os.path.join(base_path, "../vector_store/vector_index.faiss")
    meta_path = os.path.join(base_path, "../vector_store/metadata.pkl")

    # Load FAISS index and metadata
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)
    
    # Load text generation model
    generator = pipeline("text2text-generation", model="google/flan-t5-base")
    
    return embedder, index, metadata, generator


# ✅ Load everything once
embedder, index, metadata, generator = load_models()

# 🧠 Streamlit UI
st.title("🗳️ Election RAG Chatbot")
st.write("Ask questions about **campaigns**, **voter behavior**, or **media influence** 📊")

query = st.text_input("🔍 Enter your question:")

if query:
    # Step 1: Convert query into embedding
    query_embedding = embedder.encode([query]).astype("float32")

    # Step 2: Retrieve top 3 similar chunks
    k = 3
    distances, indices = index.search(query_embedding, k)

    # Step 3: Combine retrieved context
    context = ""
    sources = []
    for i, idx in enumerate(indices[0]):
        text = metadata[idx]["text"]
        src = metadata[idx]["source"]
        context += f"\n{text}\n"
        sources.append((src, text[:200] + "..."))

    # Step 4: Generate answer
    st.write("🤖 Generating answer... please wait ⏳")
    prompt = f"Answer the question using the context below:\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    result = generator(prompt, max_length=300, do_sample=False)
    answer = result[0]['generated_text']

    # Step 5: Display
    st.subheader("💬 Answer:")
    st.write(answer)

    st.subheader("📚 Retrieved Sources:")
    for s in sources:
        st.markdown(f"- **{s[0]}** → `{s[1]}`")
