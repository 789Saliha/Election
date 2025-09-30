import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Load your OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Load embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Load FAISS index and metadata
index = faiss.read_index("../vector_store/vector_index.faiss")
with open("../vector_store/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# Step 1: Get user question
query = input("🔍 Enter your question: ")

# Step 2: Convert query into embedding
query_embedding = model.encode([query]).astype("float32")

# Step 3: Search top 3 similar chunks
k = 3
distances, indices = index.search(query_embedding, k)

# Step 4: Combine top chunks
context = ""
for i, idx in enumerate(indices[0]):
    context += metadata[idx]["text"] + "\n\n"

# Step 5: Create GPT prompt
prompt = f"""
You are an assistant helping answer questions based only on the given context.

Context:
{context}

Question: {query}

Answer clearly and concisely using only the context above.
"""

print("🤖 Generating answer from GPT...\n")

# Step 6: Generate answer using GPT
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}],
)

answer = response.choices[0].message.content
print("💬 GPT Answer:\n")
print(answer)
