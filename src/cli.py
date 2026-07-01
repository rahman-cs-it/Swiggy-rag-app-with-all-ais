"""
cli.py  —  Command-line interface for the Swiggy Annual Report RAG system
Supports: Ollama (local) | OpenAI | Anthropic Claude | Google Gemini

Usage examples:
  python src/cli.py                                          # Ollama (default)
  python src/cli.py --provider openai --api-key sk-...
  python src/cli.py --provider anthropic --api-key sk-ant-...
  python src/cli.py --provider gemini --api-key AIza...
  python src/cli.py --provider ollama --model qwen2.5-3bgpu
"""

import argparse
from rag import SwiggyRAG, PROVIDER_DEFAULTS


def main():
    parser = argparse.ArgumentParser(description="Swiggy Annual Report RAG CLI")
    parser.add_argument(
        "--provider",
        choices=["ollama", "openai", "anthropic", "gemini"],
        default="ollama",
        help="LLM provider to use (default: ollama)",
    )
    parser.add_argument("--model",   default=None, help="Override model name")
    parser.add_argument("--api-key", default=None, help="API key for cloud providers")
    parser.add_argument("--top-k",   type=int, default=5, help="Chunks to retrieve (default: 5)")
    args = parser.parse_args()

    model   = args.model or PROVIDER_DEFAULTS.get(args.provider, "")
    api_key = args.api_key

    print(f"\n🛵  Swiggy Annual Report RAG")
    print(f"   Provider : {args.provider}")
    print(f"   Model    : {model}")
    print(f"   Top-K    : {args.top_k}")
    print("   Loading vector store ...")

    rag = SwiggyRAG()
    print("   Ready. Type your question (or 'exit' to quit).\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        try:
            result = rag.answer(
                query,
                top_k=args.top_k,
                provider=args.provider,
                model=model,
                api_key=api_key,
            )
        except Exception as e:
            print(f"Error: {e}\n")
            continue

        print(f"\nAnswer:\n{result['answer']}\n")
        print("Supporting context:")
        for c in result["contexts"]:
            preview = c["text"][:180].replace("\n", " ")
            print(f"  • Page {c['page_number']} (score={c['score']:.3f})  {preview}...")
        print()


if __name__ == "__main__":
    main()
