import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Load FAISS index and metadata
index = faiss.read_index("../vector_store/vector_index.faiss")
with open("../vector_store/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# Step 1: Take user question
query = input("🔍 Enter your question: ")

# Step 2: Convert query into embedding
query_embedding = model.encode([query]).astype("float32")

# Step 3: Search top 3 similar chunks
k = 3
distances, indices = index.search(query_embedding, k)

# Step 4: Combine top chunks into one context
context = ""
for i, idx in enumerate(indices[0]):
    context += metadata[idx]["text"] + "\n\n"

# Step 5: Create final prompt
prompt = f"Answer the question using the context below:\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"

# Step 6: Load summarization model
print("🤖 Loading free Hugging Face model (this may take 1-2 minutes)...")
generator = pipeline("text2text-generation", model="facebook/bart-large-cnn")

# Step 7: Generate answer
print("✨ Generating answer locally...\n")
result = generator(prompt, max_length=300, do_sample=False)

print("💬 Answer:\n")
print(result[0]['generated_text'])
