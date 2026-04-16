# models.py
from dataclasses import dataclass
from typing import List

@dataclass
class RoutingResult:
    """Holds the routing decision and the evidence behind it."""
    route: str                  # "factual" | "synthesis" | "out_of_scope"
    top_chunks: list            # Retrieved LangChain Document objects
    scores: list[float]         # Cosine similarity scores for each chunk
    max_score: float            # Highest similarity score
    sources_found: List[str]    # Unique source documents in top results
    reason: str                 # Human-readable explanation of the decision
