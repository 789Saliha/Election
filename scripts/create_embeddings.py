import os
import pickle
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Paths
input_folder = "../processed_text/"
vector_folder = "../vector_store/"
os.makedirs(vector_folder, exist_ok=True)

# Load embedding model
print("🔹 Loading embedding model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Combine all chunks into a list
documents = []
metadata = []

for file_name in os.listdir(input_folder):
    if file_name.endswith("_chunks.txt"):
        with open(os.path.join(input_folder, file_name), "r", encoding="utf-8") as f:
            content = f.read()
            chunks = content.split("--- Chunk ")
            for chunk in chunks[1:]:  # skip first empty split
                lines = chunk.split("\n", 1)
                if len(lines) > 1:
                    chunk_text = lines[1].strip()
                    if chunk_text:
                        documents.append(chunk_text)
                        metadata.append({
                            "source": file_name,
                            "text": chunk_text
                        })

print(f"✅ Total chunks collected: {len(documents)}")

# Create embeddings
print("🔹 Creating embeddings...")
embeddings = model.encode(documents, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print(f"✅ FAISS index created with {index.ntotal} vectors")

# Save index and metadata
faiss.write_index(index, os.path.join(vector_folder, "vector_index.faiss"))
with open(os.path.join(vector_folder, "metadata.pkl"), "wb") as f:
    pickle.dump(metadata, f)

print("🎉 Embeddings and index saved successfully!")
