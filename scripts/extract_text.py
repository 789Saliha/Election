import fitz
import os

# Define folders
data_folder = "../data/"
output_folder = "../processed_text/"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through each PDF and extract text
for file_name in os.listdir(data_folder):
    if file_name.endswith(".pdf"):
        pdf_path = os.path.join(data_folder, file_name)
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()

        # Save as .txt
        output_file = os.path.join(output_folder, file_name.replace(".pdf", ".txt"))
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"✅ Extracted text from: {file_name}")

print("🎉 All PDFs processed successfully!")
