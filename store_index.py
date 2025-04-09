from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import FAISS
import os

# Step 1: Load data
# extracted_data = load_pdf("data/Medical_book.pdf")  
# text_chunks = text_split(extracted_data)

# Step 2: Load embeddings
embeddings = download_hugging_face_embeddings()

# Step 3: Create FAISS index (skipped as no data to index)
# docsearch = FAISS.from_texts([t.page_content for t in text_chunks], embeddings)

# Step 4: Save index locally
# faiss_path = "faiss_index/"
# docsearch.save_local(faiss_path)

print("⚠️ Skipped FAISS index creation since PDF file is unavailable.")
