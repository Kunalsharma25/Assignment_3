import os
from dotenv import load_dotenv
from openai import OpenAI
from router import RoutingResult
from config import LLM_MODEL, LLM_TEMPERATURE, MAX_TOKENS, LLM_PROVIDER, GROQ_BASE_URL

# Load environment variables early
load_dotenv()

def initialize_client():
    """Points to the correct provider and handles missing keys gracefully."""
    provider = LLM_PROVIDER
    if provider == "groq":
        key = os.getenv("GROQ_API_KEY")
        base_url = GROQ_BASE_URL
        placeholder = "your_groq_api_key_here"
    else:
        key = os.getenv("OPENAI_API_KEY")
        base_url = None
        placeholder = "your_openai_api_key_here"

    # Use a dummy key if missing to prevent initialization crash
    # The actual error will be caught during the first API call
    safe_key = key if (key and key != placeholder) else "missing_key"
    
    return OpenAI(api_key=safe_key, base_url=base_url)

client = initialize_client()


def _format_chunks(chunks: list, max_chunks: int) -> str:
    """Format retrieved chunks into a numbered context block for the LLM prompt."""
    context_parts = []
    for i, chunk in enumerate(chunks[:max_chunks]):
        source = chunk.metadata.get("source", "Unknown")
        context_parts.append(f"[Chunk {i+1} | Source: {source}]\n{chunk.page_content}")
    return "\n\n---\n\n".join(context_parts)


def generate_factual_answer(query: str, routing_result: RoutingResult) -> str:
    """Generate a grounded factual answer from top-3 chunks."""
    context = _format_chunks(routing_result.top_chunks, max_chunks=3)

    system_prompt = """You are a precise Q&A assistant for AI regulation documents.
Answer the user's question using ONLY the provided context chunks.
If the context does not contain enough information to answer, say:
"The provided context does not contain enough detail to answer this question."
Do not add information from outside the context. Be concise and direct."""

    user_prompt = f"""Context:
{context}

Question: {query}

Answer (based only on the context above):"""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating factual answer: {str(e)}"


def generate_synthesis_answer(query: str, routing_result: RoutingResult) -> str:
    """Generate a synthesis answer from top-7 chunks across multiple documents."""
    context = _format_chunks(routing_result.top_chunks, max_chunks=7)
    sources = ", ".join(routing_result.sources_found)

    system_prompt = """You are an analytical assistant specialising in AI regulation.
You will receive multiple chunks from different source documents.
Your job is to:
1. Synthesize information across all chunks to answer the question comprehensively.
2. If different chunks or documents present conflicting information, explicitly
   acknowledge the contradiction and present both perspectives with their source.
3. Use only the provided context. Do not add external knowledge.
4. Structure your answer clearly, citing which source supports each claim."""

    user_prompt = f"""Sources consulted: {sources}

Context chunks:
{context}

Question: {query}

Synthesized Answer (note any contradictions between sources):"""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating synthesis answer: {str(e)}"


def generate_out_of_scope_response(query: str) -> str:
    """
    Hardcoded decline response. NO LLM call is made.
    This guarantees zero hallucination for out-of-scope queries.
    """
    return (
        "I'm sorry, but the provided documents do not contain enough information "
        "to answer this question. This query appears to be outside the scope of "
        "the AI regulation knowledge base. Please consult additional sources."
    )


def generate_answer(query: str, routing_result: RoutingResult) -> str:
    """
    Master dispatcher: routes to the correct generation strategy
    based on the RoutingResult.
    """
    if routing_result.route == "factual":
        return generate_factual_answer(query, routing_result)
    elif routing_result.route == "synthesis":
        return generate_synthesis_answer(query, routing_result)
    elif routing_result.route == "out_of_scope":
        return generate_out_of_scope_response(query)
    else:
        raise ValueError(f"Unknown route: {routing_result.route}")
