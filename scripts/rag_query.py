import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model and FAISS index
print("🔹 Loading model and FAISS index...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
index = faiss.read_index("../vector_store/vector_index.faiss")

# Load metadata (contains chunk text and source file)
with open("../vector_store/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# Take user input
query = input("🔍 Enter your question: ")

# Convert query to embedding
query_embedding = model.encode([query]).astype("float32")

# Search top 3 similar chunks
distances, indices = index.search(query_embedding, 3)

print("\n🎯 Top 3 relevant chunks:\n")
for i, idx in enumerate(indices[0]):
    print(f"🔹 Rank {i+1} (Score: {distances[0][i]:.2f})")
    print(f"Source File: {metadata[idx]['source']}")
    print("--------------------------------------------------")
    print(metadata[idx]['text'][:500], "...")
    print("--------------------------------------------------\n")
