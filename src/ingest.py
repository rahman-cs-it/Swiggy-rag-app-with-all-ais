"""
ingest.py
---------
Loads the Swiggy Annual Report PDF, cleans and chunks the text,
generates embeddings with a local SentenceTransformer model, and
stores everything in a FAISS vector index on disk.

Run this once before using the CLI or Streamlit app:
    python src/ingest.py
"""

import os
import re
import json
import pickle

import numpy as np
from pypdf import PdfReader
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss

# ---- Config -----------------------------------------------------------
PDF_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Swiggy_Annual_Report_FY2023-24.pdf")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "vectorstore")
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
META_PATH = os.path.join(INDEX_DIR, "chunks.pkl")

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # small, fast, runs locally, no API key needed
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
# ------------------------------------------------------------------------


def load_pdf_text(pdf_path: str):
    """Extract raw text from every page, keeping track of page numbers.

    Uses strict=False so that slightly malformed/truncated PDFs (e.g. an
    interrupted browser download) can still be parsed as far as possible,
    instead of raising and aborting the whole ingestion run.
    """
    reader = PdfReader(pdf_path, strict=False)
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            print(f"  Warning: could not extract text from page {i + 1}: {e}")
            text = ""
        pages.append({"page_number": i + 1, "text": text})
    return pages


def clean_text(text: str) -> str:
    """Basic cleanup: collapse whitespace, drop stray control chars."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text


def chunk_pages(pages):
    """Split each page's text into overlapping chunks, attaching metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for page in pages:
        cleaned = clean_text(page["text"])
        if not cleaned:
            continue
        for piece in splitter.split_text(cleaned):
            if len(piece.strip()) < 30:
                continue  # skip near-empty fragments
            chunks.append({
                "text": piece.strip(),
                "page_number": page["page_number"],
            })
    return chunks


def build_index(chunks, model):
    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks with '{EMBED_MODEL_NAME}' ...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # so we can use inner product == cosine similarity
    ).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def main():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"Could not find PDF at {PDF_PATH}")

    os.makedirs(INDEX_DIR, exist_ok=True)

    print("Loading PDF ...")
    pages = load_pdf_text(PDF_PATH)
    print(f"Loaded {len(pages)} pages.")

    print("Chunking text ...")
    chunks = chunk_pages(pages)
    print(f"Created {len(chunks)} chunks.")

    model = SentenceTransformer(EMBED_MODEL_NAME)
    index = build_index(chunks, model)

    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Saved FAISS index to {INDEX_PATH}")
    print(f"Saved chunk metadata to {META_PATH}")
    print("Ingestion complete.")


if __name__ == "__main__":
    main()
