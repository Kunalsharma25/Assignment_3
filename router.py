# router.py
import os
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from config import (OUT_OF_SCOPE_THRESHOLD, SYNTHESIS_SOURCE_MIN,
                    SYNTHESIS_KEYWORDS, TOP_K_SYNTHESIS, CONFIDENCE_GAP_THRESHOLD,
                    SYNTHESIS_MIN_CONFIDENCE, SCORE_SPREAD_THRESHOLD, GROQ_BASE_URL)

# Ensure API keys are available
load_dotenv()

@dataclass
class RoutingResult:
    """Holds the routing decision and the evidence behind it."""
    route: str                  # "factual" | "synthesis" | "out_of_scope"
    top_chunks: list            # Retrieved LangChain Document objects
    scores: list[float]         # Cosine similarity scores for each chunk
    max_score: float            # Highest similarity score
    sources_found: list[str]    # Unique source documents in top results
    reason: str                 # Human-readable explanation of the decision


def route_query(query: str, vector_store: FAISS) -> RoutingResult:
    """
    Explicit, inspectable routing logic with Independent Agentic Sanitization.
    """
    # Step 1: Retrieve with scores
    results_with_scores = vector_store.similarity_search_with_score(
        query, k=TOP_K_SYNTHESIS
    )

    if not results_with_scores:
        return RoutingResult(
            route="out_of_scope",
            top_chunks=[],
            scores=[],
            max_score=0.0,
            sources_found=[],
            reason="No chunks retrieved."
        )

    chunks = [r[0] for r in results_with_scores]
    raw_scores = [r[1] for r in results_with_scores]
    scores = [max(0.0, 1 - (d / 2)) for d in raw_scores]
    max_score = max(scores)

    # Step 2: Basic rejection
    if max_score < OUT_OF_SCOPE_THRESHOLD:
        return RoutingResult(
            route="out_of_scope",
            top_chunks=chunks,
            scores=scores,
            max_score=max_score,
            sources_found=[],
            reason=f"Score {max_score:.3f} below floor {OUT_OF_SCOPE_THRESHOLD}."
        )

    # Step 3: Noise rejection
    score_spread = max(scores) - min(scores)
    if score_spread < SCORE_SPREAD_THRESHOLD and max_score < 0.65:
        return RoutingResult(
            route="out_of_scope",
            top_chunks=chunks,
            scores=scores,
            max_score=max_score,
            sources_found=[],
            reason=f"Noise Filter: Scores too uniform ({score_spread:.4f})."
        )

    # Step 4: Independent Agentic Sanitizer (Fixed local initialization to prevent hang)
    if max_score < 0.65:
        # Create a local client to avoid circular imports during sanitization
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
            sanitizer_prompt = (
                f"Is the following question specifically related to AI regulation in the EU, US, China, or UK?\n"
                f"If the question is about Brazil, India, general coding, or environment, answer NO.\n"
                f"Question: {query}\n"
                f"Answer only YES or NO."
            )
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": sanitizer_prompt}],
                max_tokens=5,
                temperature=0.0
            )
            if "NO" in resp.choices[0].message.content.strip().upper():
                return RoutingResult(
                    route="out_of_scope",
                    top_chunks=chunks,
                    scores=scores,
                    max_score=max_score,
                    sources_found=[],
                    reason="Agentic Rejection: LLM confirmed this topic is unrelated to core regions."
                )

    # Step 5: Multi-source analysis
    source_scores = {}
    for chunk, score in zip(chunks, scores):
        src = chunk.metadata.get("source", "unknown")
        if src not in source_scores or score > source_scores[src]:
            source_scores[src] = score

    sources_found = list(source_scores.keys())
    query_lower = query.lower()
    matched_keywords = [kw for kw in SYNTHESIS_KEYWORDS if kw in query_lower]
    
    is_factual_dominant = False
    if len(sources_found) > 1:
        sorted_scores = sorted(source_scores.values(), reverse=True)
        if (sorted_scores[0] - sorted_scores[1]) >= CONFIDENCE_GAP_THRESHOLD:
            is_factual_dominant = True

    # Step 6: Synthesis
    if matched_keywords or (len(sources_found) >= SYNTHESIS_SOURCE_MIN and not is_factual_dominant and max_score >= SYNTHESIS_MIN_CONFIDENCE):
        return RoutingResult(
            route="synthesis",
            top_chunks=chunks,
            scores=scores,
            max_score=max_score,
            sources_found=sources_found,
            reason="Synthesis signals detected."
        )

    # Step 7: Factual
    return RoutingResult(
        route="factual",
        top_chunks=chunks[:3],
        scores=scores[:3],
        max_score=max_score,
        sources_found=sources_found,
        reason="Factual match secured."
    )
