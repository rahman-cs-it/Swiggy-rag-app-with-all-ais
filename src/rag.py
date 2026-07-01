"""
rag.py
------
Core RAG pipeline with multi-provider LLM support.
API keys are loaded from .env automatically (via python-dotenv),
and can also be overridden at runtime from the UI or CLI.

Providers: ollama | openai | anthropic | gemini
"""

import os
import pickle
from pathlib import Path

import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load .env for local development
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")

# On Streamlit Cloud, also pull keys from st.secrets into os.environ
try:
    import streamlit as _st
    for _k in ["GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OLLAMA_MODEL"]:
        if _k in _st.secrets and not os.environ.get(_k):
            os.environ[_k] = _st.secrets[_k]
except Exception:
    pass  # CLI mode or secrets not configured — safe to ignore

INDEX_DIR = _ROOT / "vectorstore"
INDEX_PATH = INDEX_DIR / "faiss.index"
META_PATH  = INDEX_DIR / "chunks.pkl"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5

SYSTEM_PROMPT = """You are a precise financial document assistant specializing in Swiggy's Annual Report FY 2023-24.

Answer the user's question using ONLY the information provided in the context below.

Response format rules (follow these strictly):
- Give a thorough, well-structured answer — do NOT give a one-line response.
- Start with a direct answer to the question in 1-2 sentences.
- Then elaborate with relevant details, numbers, and facts from the context.
- Use bullet points or numbered lists when presenting multiple data points or facts.
- Always cite the page number(s) your answer is drawn from, e.g. (Page 12).
- If the answer is not contained in the context, respond exactly:
  "I could not find this information in the Swiggy Annual Report."
- Never use outside knowledge or make assumptions beyond what the context states.
"""

PROVIDER_DEFAULTS = {
    "ollama":    os.environ.get("OLLAMA_MODEL", "qwen2.5-3bgpu"),
    "openai":    "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-20241022",
    "gemini":    "gemini-2.5-flash",
}


class SwiggyRAG:
    def __init__(self):
        if not (INDEX_PATH.exists() and META_PATH.exists()):
            raise FileNotFoundError(
                "Vector store not found. Run `python src/ingest.py` first."
            )
        self.embed_model = SentenceTransformer(
            EMBED_MODEL_NAME,
            cache_folder=str(_ROOT / "models"),
        )
        self.index = faiss.read_index(str(INDEX_PATH))
        with open(META_PATH, "rb") as f:
            self.chunks = pickle.load(f)

    # ── Retrieval ────────────────────────────────────────────────────────────────
    def retrieve(self, query: str, top_k: int = TOP_K):
        query_vec = self.embed_model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        scores, indices = self.index.search(query_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append({
                "text":        chunk["text"],
                "page_number": chunk["page_number"],
                "score":       float(score),
            })
        return results

    def build_prompt(self, query: str, contexts: list) -> str:
        context_block = "\n\n".join(
            f"[Page {c['page_number']}]\n{c['text']}" for c in contexts
        )
        return (
            f"Context from the Swiggy Annual Report:\n\n{context_block}\n\n"
            f"Question: {query}\n\nAnswer:"
        )

    # ── Provider calls ───────────────────────────────────────────────────────────
    def _call_ollama(self, prompt: str, model: str) -> str:
        import ollama as _ollama
        response = _ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
        )
        return response["message"]["content"]

    def _call_openai(self, prompt: str, model: str, api_key: str) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
        )
        return response.choices[0].message.content

    def _call_anthropic(self, prompt: str, model: str, api_key: str) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def _call_gemini(self, prompt: str, model: str, api_key: str) -> str:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        gen_model = genai.GenerativeModel(
            model_name=model,
            system_instruction=SYSTEM_PROMPT,
        )
        return gen_model.generate_content(prompt).text

    # ── Resolve API key: runtime arg > .env ─────────────────────────────────────
    @staticmethod
    def _resolve_key(provider: str, runtime_key: str | None) -> str | None:
        if runtime_key:
            return runtime_key
        env_map = {
            "openai":    "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini":    "GEMINI_API_KEY",
        }
        return os.environ.get(env_map.get(provider, ""))

    # ── Public entry point ───────────────────────────────────────────────────────
    def answer(
        self,
        query:    str,
        top_k:    int  = TOP_K,
        provider: str  = "ollama",
        model:    str  = None,
        api_key:  str  = None,
    ) -> dict:
        provider = provider.lower()
        model    = model or PROVIDER_DEFAULTS.get(provider, "")
        contexts = self.retrieve(query, top_k=top_k)
        prompt   = self.build_prompt(query, contexts)

        if provider == "ollama":
            answer_text = self._call_ollama(prompt, model)
        else:
            key = self._resolve_key(provider, api_key)
            if not key:
                raise ValueError(
                    f"No API key found for '{provider}'. "
                    f"Add it to your .env file or paste it in the sidebar."
                )
            if provider == "openai":
                answer_text = self._call_openai(prompt, model, key)
            elif provider == "anthropic":
                answer_text = self._call_anthropic(prompt, model, key)
            elif provider == "gemini":
                answer_text = self._call_gemini(prompt, model, key)
            else:
                raise ValueError(f"Unknown provider '{provider}'.")

        return {"answer": answer_text, "contexts": contexts}
