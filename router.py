# router.py
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from models import RoutingResult
from llm_utils import initialize_client
from config import (OUT_OF_SCOPE_THRESHOLD, SYNTHESIS_SOURCE_MIN,
                    SYNTHESIS_KEYWORDS, TOP_K_SYNTHESIS, CONFIDENCE_GAP_THRESHOLD,
                    SYNTHESIS_MIN_CONFIDENCE, SCORE_SPREAD_THRESHOLD)

# Ensure environment is loaded
load_dotenv()

def route_query(query: str, vector_store: FAISS) -> RoutingResult:
    """
    Explicit, inspectable routing logic with Agentic Sanitization.
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
            reason="No chunks retrieved from the vector store."
        )

    chunks = [r[0] for r in results_with_scores]
    raw_scores = [r[1] for r in results_with_scores]
    # FAISS returns L2 distance; convert to similarity (0 to 1)
    scores = [max(0.0, 1 - (d / 2)) for d in raw_scores]
    max_score = max(scores)

    # Step 2: Out-of-scope gate
    if max_score < OUT_OF_SCOPE_THRESHOLD:
        return RoutingResult(
            route="out_of_scope",
            top_chunks=chunks,
            scores=scores,
            max_score=max_score,
            sources_found=[],
            reason=f"Max similarity score {max_score:.3f} is below threshold {OUT_OF_SCOPE_THRESHOLD}."
        )

    # Step 3: Noise Filter (Score Dispersion)
    score_spread = max(scores) - min(scores)
    if score_spread < SCORE_SPREAD_THRESHOLD and max_score < 0.65:
        return RoutingResult(
            route="out_of_scope",
            top_chunks=chunks,
            scores=scores,
            max_score=max_score,
            sources_found=[],
            reason=f"Noise detected: Score spread {score_spread:.4f} is too flat."
        )

    # Step 4: Agentic Sanitizer (Verify marginal queries)
    if max_score < 0.65:
        client = initialize_client()
        sanitizer_prompt = (
            f"Is the following question related to AI regulation in the EU, US, China, or UK?\n"
            f"If it asks about other regions (like Brazil, India) or unrelated topics, answer NO.\n"
            f"Question: {query}\n"
            f"Answer only YES or NO."
        )
        # Use a faster, high-limit model for the gate
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": sanitizer_prompt}],
            max_tokens=5,
            temperature=0.0
        )
        is_relevant = response.choices[0].message.content.strip().upper()
        if "NO" in is_relevant:
            return RoutingResult(
                route="out_of_scope",
                top_chunks=chunks,
                scores=scores,
                max_score=max_score,
                sources_found=[],
                reason="Agentic Rejection: LLM verified topic is unrelated to core regions."
            )

    # Step 5: Identify unique sources and their max scores
    source_scores = {}
    for chunk, score in zip(chunks, scores):
        src = chunk.metadata.get("source", "unknown")
        if src not in source_scores or score > source_scores[src]:
            source_scores[src] = score

    sources_found = list(source_scores.keys())
    query_lower = query.lower()

    # Step 6: Identify signals
    matched_keywords = [kw for kw in SYNTHESIS_KEYWORDS if kw in query_lower]
    multi_source = len(sources_found) >= SYNTHESIS_SOURCE_MIN
    
    is_factual_dominant = False
    if len(sources_found) > 1:
        sorted_scores = sorted(source_scores.values(), reverse=True)
        # Confidence Gap check
        if sorted_scores[0] - sorted_scores[1] >= CONFIDENCE_GAP_THRESHOLD:
            is_factual_dominant = True

    # Step 7: Routing Decisions
    # Synthesis
    is_synthesis = False
    if matched_keywords:
        is_synthesis = True
    elif multi_source and not is_factual_dominant:
        # Require higher confidence without keywords
        if max_score >= (SYNTHESIS_MIN_CONFIDENCE + 0.10):
            is_synthesis = True

    if is_synthesis:
        return RoutingResult(
            route="synthesis",
            top_chunks=chunks,
            scores=scores,
            max_score=max_score,
            sources_found=sources_found,
            reason="Synthesis signals detected."
        )

    # Step 8: Ambiguous Noise Rejection
    if not is_factual_dominant and len(sources_found) > 1 and max_score < 0.48 and not matched_keywords:
         return RoutingResult(
            route="out_of_scope",
            top_chunks=chunks,
            scores=scores,
            max_score=max_score,
            sources_found=sources_found,
            reason="Ambiguous noise: Multi-source retrieval with low confidence."
        )

    # Default to Factual
    reason = f"Max score {max_score:.3f} above threshold. "
    if is_factual_dominant:
        reason += "Dominant source found (Confidence Gap hit)."
    else:
        reason += "Specific factual match."

    return RoutingResult(
        route="factual",
        top_chunks=chunks[:3],
        scores=scores[:3],
        max_score=max_score,
        sources_found=sources_found,
        reason=reason
    )
