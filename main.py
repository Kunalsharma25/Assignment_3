# main.py
import os
from dotenv import load_dotenv
from ingestion import load_vector_store, ingest_documents
from router import route_query
from generator import generate_answer

load_dotenv()


def print_separator():
    print("\n" + "─" * 70 + "\n")


def run_interactive_demo():
    """Interactive Q&A demo for the video walkthrough."""
    print("="*70)
    print("  AGENTIC RAG SYSTEM — AI Regulation Q&A")
    print("="*70)

    # Check for API key based on provider
    from config import LLM_PROVIDER
    if LLM_PROVIDER == "groq":
        key_name = "GROQ_API_KEY"
        placeholder = "your_groq_api_key_here"
    else:
        key_name = "OPENAI_API_KEY"
        placeholder = "your_openai_api_key_here"

    if not os.getenv(key_name) or os.getenv(key_name) == placeholder:
        print(f"\n[WARNING] {key_name} not found or still set to placeholder.")
        print(f"Please update the .env file with a valid API key for your provider ({LLM_PROVIDER}).")
        print("Factual and Synthesis routes will return errors without a valid key.")

    # Load or build vector store
    try:
        vector_store = load_vector_store()
        print("Vector store loaded successfully.")
    except FileNotFoundError:
        print("Vector store not found. Running ingestion first...")
        vector_store = ingest_documents()

    print("\nType your question and press Enter.")
    print("Type 'quit' to exit.\n")

    while True:
        query = input("Your Question: ").strip()

        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not query:
            continue

        print_separator()

        # Route the query
        routing_result = route_query(query, vector_store)

        print(f"[ROUTING DECISION]")
        print(f"  Route:      {routing_result.route.upper()}")
        print(f"  Max Score:  {routing_result.max_score:.4f}")
        print(f"  Sources:    {routing_result.sources_found}")
        print(f"  Reason:     {routing_result.reason}")

        print(f"\n[ANSWER]")
        answer = generate_answer(query, routing_result)
        print(answer)

        if routing_result.route != "out_of_scope":
            print(f"\n[SOURCES USED]")
            for i, chunk in enumerate(routing_result.top_chunks[:3]):
                src = chunk.metadata.get("source", "unknown")
                print(f"  Chunk {i+1} ({src}): {chunk.page_content[:100]}...")

        print_separator()


if __name__ == "__main__":
    run_interactive_demo()
