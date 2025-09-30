import os

# Folder paths
input_folder = "../processed_text/"
output_folder = "../processed_text/"

# Define chunk size (in characters)
CHUNK_SIZE = 1000
OVERLAP = 200

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks

# Process each text file
for file_name in os.listdir(input_folder):
    if file_name.endswith(".txt"):
        file_path = os.path.join(input_folder, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text)
        print(f"✅ {file_name} split into {len(chunks)} chunks")

        # Save chunks as one file for reference (optional)
        chunk_output_path = os.path.join(output_folder, file_name.replace(".txt", "_chunks.txt"))
        with open(chunk_output_path, "w", encoding="utf-8") as out:
            for i, chunk in enumerate(chunks):
                out.write(f"--- Chunk {i+1} ---\n{chunk}\n\n")

print("🎉 All text files have been chunked successfully!")
