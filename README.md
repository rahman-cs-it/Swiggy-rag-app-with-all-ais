# 🛵 Swiggy Annual Report — RAG Q&A System

A Retrieval-Augmented Generation (RAG) application that answers natural language questions about Swiggy's FY 2023-24 Annual Report, grounded strictly in the document content — no hallucination.

🔗 **Live Demo:** [rahman-cs-it-swiggy-rag-app.streamlit.app](https://rahman-cs-it-swiggy-rag-app.streamlit.app)

---

## 📄 Source Document

| Field | Detail |
|---|---|
| Document | Swiggy Annual Report FY 2023-24 |
| Pages | 170 |
| Format | PDF |
| Source | [Swiggy Investor Relations — BSE/NSE Filing](https://www.bseindia.com/stock-share-price/swiggy-ltd/swiggy/544356/) |

---

## 🏗️ Architecture

```
PDF (170 pages)
   │  pypdf — extract text per page
   ▼
Text Cleaning + Chunking
   │  LangChain RecursiveCharacterTextSplitter
   │  chunk size: 1000 chars | overlap: 150 | page-number metadata retained
   ▼
Embeddings
   │  sentence-transformers: all-MiniLM-L6-v2 (local, free, no API key)
   ▼
FAISS Vector Index
   │  cosine similarity via normalized inner product
   ▼
User Query → embed → top-k retrieval → context-only prompt
   ▼
LLM (choose one):
   ├── 🔵 Google Gemini     (free tier available)
   ├── 🖥️  Ollama / Local LLM (fully offline)
   ├── 🟢 OpenAI GPT        (requires billing)
   └── 🟠 Anthropic Claude  (requires billing)
   ▼
Answer + Supporting Context (with page numbers)
```

---

## 🤖 Supported LLM Providers

| Provider | Model (default) | API Key Required | Free Tier |
|---|---|---|---|
| 🔵 Google Gemini | `gemini-1.5-flash` | Yes | ✅ Yes |
| 🖥️ Ollama (Local) | `qwen2.5-3bgpu` | No | ✅ Fully free |
| 🟢 OpenAI | `gpt-4o-mini` | Yes | ❌ Billing required |
| 🟠 Anthropic Claude | `claude-3-5-haiku-20241022` | Yes | ❌ Billing required |

> **Note:** Ollama works only when running the app locally on your own machine. It is not available on the hosted Streamlit deployment.

---

## ⚙️ Design Decisions

- **Embeddings — `all-MiniLM-L6-v2`:** Small (80MB), fast, fully local, no API key needed. Works identically across local and cloud deployments.
- **Vector store — FAISS:** Lightweight, in-process, no external service or database needed. Index stored as files alongside the code.
- **Multi-provider LLM:** Users can switch between free (Gemini, Ollama) and paid (OpenAI, Claude) providers from the sidebar without changing any code.
- **Strict grounding:** The system prompt instructs every LLM to answer only from retrieved context and explicitly state when information is not found — minimising hallucination.
- **Page-number metadata:** Every chunk retains its source page number, shown in the "Supporting context" expander so answers are traceable.

---

## 🚀 Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/rahman-cs-it/swiggy-rag-app-with-all-ais.git
cd swiggy-rag-app-with-all-ais
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv

# Windows (PowerShell)
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure API keys

Copy the example env file and fill in your keys:

```bash
cp .env.example .env
```

Edit `.env`:

```env
OLLAMA_MODEL=qwen2.5-3bgpu        # your pulled Ollama model name
GEMINI_API_KEY=AIza...            # free at aistudio.google.com
OPENAI_API_KEY=sk-...             # requires billing
ANTHROPIC_API_KEY=sk-ant-...      # requires billing
```

Only fill the keys for providers you intend to use. Gemini's free tier is enough to get started.

### 4. (Optional) Set up Ollama for local LLM

Download Ollama from [ollama.com](https://ollama.com), then pull any model:

```bash
ollama pull qwen2.5        # fast, 1.9 GB
ollama pull gemma3         # good quality, 5 GB
ollama pull llama3.1       # best quality, 4.7 GB
```

### 5. Build the vector index (run once)

```bash
python src/ingest.py
```

This reads the PDF, chunks it, generates embeddings, and writes:
- `vectorstore/faiss.index`
- `vectorstore/chunks.pkl`

### 6. Run the app

**Streamlit UI:**
```bash
streamlit run src/app.py
```

**CLI:**
```bash
# Ollama (local, no key needed)
python src/cli.py --provider ollama --model qwen2.5-3bgpu

# Gemini
python src/cli.py --provider gemini --api-key AIza...

# OpenAI
python src/cli.py --provider openai --api-key sk-...

# Anthropic
python src/cli.py --provider anthropic --api-key sk-ant-...
```

---

## ☁️ Streamlit Cloud Deployment

1. Push the repo to GitHub (including `vectorstore/` and `data/` — these must be committed)
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → select your repo, branch `main`, main file `src/app.py`
3. Under **Settings → Secrets**, add your keys in TOML format (all values must be quoted):

```toml
GEMINI_API_KEY = "AIza..."
OPENAI_API_KEY = "sk-..."
ANTHROPIC_API_KEY = "sk-ant-..."
```

4. Deploy — first load takes ~60 seconds while dependencies install; subsequent loads are faster.

---

## 📁 Project Structure

```
swiggy-rag/
├── data/
│   └── Swiggy_Annual_Report_FY2023-24.pdf   # source document
├── vectorstore/
│   ├── faiss.index                           # generated by ingest.py
│   └── chunks.pkl                           # generated by ingest.py
├── src/
│   ├── ingest.py      # PDF loading, cleaning, chunking, embedding, indexing
│   ├── rag.py         # retrieval + multi-provider generation pipeline
│   ├── cli.py         # command-line interface with provider flags
│   └── app.py         # Streamlit web interface
├── .env.example       # API key template (copy to .env)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 💬 Example Questions

- "What was Swiggy's total revenue from operations in FY24?"
- "What are the key risk factors mentioned in the report?"
- "Who are the members of Swiggy's board of directors?"
- "What is Swiggy's strategy for profitability?"
- "How many orders did Swiggy fulfill in FY24?"
- "What is Instamart and what is its contribution to the business?"

---

## ⚠️ Limitations

- **Table extraction:** Financial tables in PDFs may not extract perfectly with `pypdf`. For production use, consider `pdfplumber` or `unstructured` for better table parsing.
- **Chunk count:** The report produced 70 chunks — some niche questions may not retrieve the right context if the relevant section is sparse.
- **Local LLM on cloud:** Ollama is only usable when running the app locally. It is disabled/not recommended for the hosted deployment.
- **First load time:** Streamlit Cloud cold-starts take ~60 seconds due to model and dependency loading.
