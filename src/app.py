"""
app.py  —  Streamlit UI for Swiggy Annual Report RAG
Default provider order: Gemini → Ollama (Local) → OpenAI → Anthropic
"""

import os
import streamlit as st
from rag import SwiggyRAG, PROVIDER_DEFAULTS

# Wrap in a try/except block to safely catch secondary execution attempts
try:
    st.set_page_config(page_title="Swiggy Annual Report Q&A", page_icon="🛵", layout="wide")
except st.errors.StreamlitAPIException:
    pass # Already set, safely ignore the exception

st.title("🛵 Swiggy Annual Report — RAG Q&A")
st.caption("Ask questions about Swiggy's FY 2023-24 Annual Report. "
           "Answers are generated strictly from the document content.")
# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ LLM Provider")

    PROVIDER_ORDER  = ["gemini", "ollama", "openai", "anthropic"]
    PROVIDER_LABELS = {
        "gemini":    "🔵 Google Gemini (Recommended)",
        "ollama":    "🖥️  Ollama — Local LLM",
        "openai":    "🟢 OpenAI (GPT-4o-mini)",
        "anthropic": "🟠 Anthropic Claude",
    }

    provider = st.selectbox(
        "Choose provider",
        options=PROVIDER_ORDER,
        format_func=lambda x: PROVIDER_LABELS[x],
        index=0,
    )

    model = st.text_input("Model name", value=PROVIDER_DEFAULTS[provider])

    st.divider()

    # ── GEMINI ────────────────────────────────────────────────────────────────
    api_key_override = None

    if provider == "gemini":
        key_loaded = bool(os.environ.get("GEMINI_API_KEY"))

        if key_loaded:
            st.success("✅ Gemini API key loaded from `.env`")
        else:
            st.info(
                "**Google Gemini** offers a generous **free tier** — "
                "no billing required to get started.\n\n"
                "Get your free API key at 👉 "
                "[aistudio.google.com](https://aistudio.google.com/app/apikey)"
            )

        # Always show model change notice
        st.warning(
            "⚠️ **Model notice:**\n\n"
            "The developer has configured this app with the **free Gemini API** "
            f"(`{PROVIDER_DEFAULTS['gemini']}`). This model works without billing.\n\n"
            "If you change the model name above to a **paid Gemini model** "
            "(e.g. `gemini-1.5-pro`, `gemini-2.0-flash`), "
            "you will need to enter **your own API key** with billing enabled below, "
            "as the developer's free key will not support it."
        )

        api_key_override = st.text_input(
            "Override Gemini API key (optional)",
            type="password",
            placeholder="AIza... — required if using a paid model",
        )
        st.caption(
            "Leave blank to use the developer's free key (default model only). "
            "Get your own key 👉 [aistudio.google.com](https://aistudio.google.com/app/apikey)"
        )

    # ── OLLAMA (LOCAL) ────────────────────────────────────────────────────────
    elif provider == "ollama":
        st.error(
            "🖥️ **Local LLM — Important Notice**\n\n"
            "Ollama runs **only on the machine it is installed on**. "
            "Visitors accessing this app over the internet **cannot** use "
            "their own local models — only the machine hosting this app can serve them.\n\n"
            "**This option is not suitable for public/production deployments.** "
            "Use Gemini, OpenAI, or Claude for a hosted app.\n\n"
            "**To run locally on your own machine:**\n"
            "1. Install Ollama → [ollama.com](https://ollama.com)\n"
            "2. Pull a model: `ollama pull qwen2.5-3bgpu`\n"
            "3. Run this app on your own machine (not on a remote server)\n\n"
            f"Currently configured model: `{model}`"
        )

    # ── OPENAI ────────────────────────────────────────────────────────────────
    elif provider == "openai":
        key_loaded = bool(os.environ.get("OPENAI_API_KEY"))

        if key_loaded:
            st.success("✅ OpenAI API key loaded from `.env`")

        # Always show developer billing notice
        st.warning(
            "⚠️ **Developer notice — OpenAI requires billing**\n\n"
            "The developer of this app **has not set up an OpenAI billing account**, "
            "so no default OpenAI key is provided.\n\n"
            "If **you** have an OpenAI account with billing enabled, "
            "you can enter your own API key below to use GPT models at your own cost.\n\n"
            "Get your key at 👉 "
            "[platform.openai.com/api-keys](https://platform.openai.com/api-keys)"
        )

        api_key_override = st.text_input(
            "Your OpenAI API key" if not key_loaded else "Override OpenAI API key (optional)",
            type="password",
            placeholder="sk-...",
        )

    # ── ANTHROPIC ─────────────────────────────────────────────────────────────
    elif provider == "anthropic":
        key_loaded = bool(os.environ.get("ANTHROPIC_API_KEY"))

        if key_loaded:
            st.success("✅ Anthropic API key loaded from `.env`")

        # Always show developer billing notice
        st.warning(
            "⚠️ **Developer notice — Anthropic requires billing**\n\n"
            "The developer of this app **has not set up an Anthropic billing account**, "
            "so no default Claude key is provided.\n\n"
            "If **you** have an Anthropic account with billing enabled, "
            "you can enter your own API key below to use Claude models at your own cost.\n\n"
            "Get your key at 👉 "
            "[console.anthropic.com](https://console.anthropic.com/settings/keys)"
        )

        api_key_override = st.text_input(
            "Your Anthropic API key" if not key_loaded else "Override Anthropic API key (optional)",
            type="password",
            placeholder="sk-ant-...",
        )

    st.divider()
    top_k = st.slider("Context chunks to retrieve", min_value=2, max_value=10, value=5)

    st.divider()
    st.markdown("""
**How it works**
1. Question embedded locally (MiniLM)
2. Top-k chunks retrieved from FAISS
3. Chunks + question sent to the LLM
4. Answer grounded strictly in context
    """)

# ── Load RAG ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔄 Loading AI model and document index (first load only)...")
def load_rag():
    return SwiggyRAG()

try:
    rag = load_rag()
except FileNotFoundError as e:
    st.error(f"⚠️ {e}")
    st.stop()

st.info("⚡ First load takes 30–60 seconds while the AI model initialises. Subsequent questions are fast.", icon="ℹ️")

# ── Main Q&A ──────────────────────────────────────────────────────────────────
query = st.text_input(
    "Your question",
    placeholder="e.g. What was Swiggy's total revenue in FY24?",
)

if st.button("Ask", type="primary") and query.strip():
    with st.spinner(f"Retrieving context and generating via **{provider}** (`{model}`) ..."):
        try:
            result = rag.answer(
                query,
                top_k=top_k,
                provider=provider,
                model=model,
                api_key=api_key_override or None,
            )
        except ValueError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.stop()

    st.subheader("💬 Answer")
    st.markdown(result["answer"])

    with st.expander("📄 Supporting context (retrieved chunks)"):
        for i, c in enumerate(result["contexts"], 1):
            st.markdown(
                f"**Chunk {i} — Page {c['page_number']}** &nbsp; "
                f"_(similarity: {c['score']:.3f})_"
            )
            st.write(c["text"])
            st.divider()
