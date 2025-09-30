import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# 🔹 Load all models and data
@st.cache_resource
def load_models():
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    index = faiss.read_index("../vector_store/vector_index.faiss")
    with open("../vector_store/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    generator = pipeline("text2text-generation", model="google/flan-t5-base")
    return embedder, index, metadata, generator

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

    # Step 3: Combine context
    context = ""
    sources = []
    for i, idx in enumerate(indices[0]):
        text = metadata[idx]["text"]
        src = metadata[idx]["source"]
        context += text + "\n\n"
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
