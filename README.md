# Swiggy Annual Report — RAG Q&A System

A Retrieval-Augmented Generation (RAG) application that answers natural
language questions about Swiggy's FY 2023-24 Annual Report, grounded
strictly in the document content (no hallucination).

## Source Document

- **Document:** Swiggy Annual Report FY 2023-24
- **Source:** Publicly available on Swiggy's investor relations site
  (BSE/NSE filings) — place the PDF at `data/Swiggy_Annual_Report_FY2023-24.pdf`
  before running ingestion. *(Add the exact URL you downloaded it from here.)*

## Architecture

```
PDF (170 pages)
   │  pypdf: extract text per page
   ▼
Cleaning + Chunking (LangChain RecursiveCharacterTextSplitter,
   1000 chars / 150 overlap, page-number metadata retained)
   ▼
Embeddings (sentence-transformers: all-MiniLM-L6-v2 — local, free, no API key)
   ▼
FAISS vector index (cosine similarity via normalized inner product)
   ▼
User query → embed → top-k retrieval → context-only prompt → local LLM (Ollama)
   ▼
Answer + supporting context (with page numbers)
```

## Why these choices

- **Embeddings:** `all-MiniLM-L6-v2` is small, fast, fully local, and
  requires no API key — good for an assignment that should run anywhere.
- **Vector store:** FAISS is lightweight, in-process, and needs no
  external service.
- **LLM:** Ollama runs models like Llama 3.1 locally, keeping the whole
  pipeline free and offline. Swap `OLLAMA_MODEL` env var for any model
  you've pulled (e.g. `mistral`, `phi3`).
- **Strict grounding:** the system prompt instructs the model to answer
  only from retrieved context and to explicitly say when information
  isn't present, reducing hallucination.

## Setup

### 1. Install Python dependencies

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Install and start Ollama

Download from https://ollama.com, then pull a model:

```bash
ollama pull llama3.1
```

Ollama runs as a local server in the background automatically after install.

### 3. Add the document

Place the annual report PDF at:

```
data/Swiggy_Annual_Report_FY2023-24.pdf
```

### 4. Build the vector index (run once, or whenever the PDF changes)

```bash
python src/ingest.py
```

This creates `vectorstore/faiss.index` and `vectorstore/chunks.pkl`.

### 5. Ask questions

**CLI:**

```bash
python src/cli.py
```

**Web UI (Streamlit):**

```bash
streamlit run src/app.py
```

## Project structure

```
swiggy-rag/
├── data/
│   └── Swiggy_Annual_Report_FY2023-24.pdf
├── vectorstore/
│   ├── faiss.index        # created by ingest.py
│   └── chunks.pkl         # created by ingest.py
├── src/
│   ├── ingest.py          # PDF loading, cleaning, chunking, embedding
│   ├── rag.py             # retrieval + generation pipeline
│   ├── cli.py             # command-line interface
│   └── app.py             # Streamlit interface
├── requirements.txt
└── README.md
```

## Example questions to try

- "What was Swiggy's total revenue from operations in FY24?"
- "What are the key risk factors mentioned in the report?"
- "Who are the members of Swiggy's board of directors?"
- "What is Swiggy's strategy for profitability?"

## Notes / limitations

- Answer quality depends on the local LLM chosen via Ollama — larger
  models (e.g. `llama3.1:70b`) give better reasoning at the cost of speed.
- Table-heavy financial pages may extract imperfectly with `pypdf`;
  for production use, consider a layout-aware extractor (e.g. `pdfplumber`
  or `unstructured`) for tables.
- Re-run `ingest.py` any time the source PDF changes.
