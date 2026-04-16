# evaluator.py
import os
from ingestion import load_vector_store
from router import route_query
from generator import generate_answer

# Load embeddings/DLLs FIRST before other libraries can conflict
vector_store = load_vector_store()

import time
import pandas as pd
from rouge_score import rouge_scorer
from dotenv import load_dotenv
from test_questions import TEST_QUESTIONS
from config import RESULTS_DIR

load_dotenv()


def compute_rouge_l(hypothesis: str, reference_keywords: list[str]) -> float:
    """
    Compute ROUGE-L between generated answer and a pseudo-reference
    built from expected keywords.
    Returns a score between 0.0 and 1.0.
    """
    if not reference_keywords:
        return 1.0  # Out-of-scope: full score if system correctly declines

    reference = " ".join(reference_keywords)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return round(scores["rougeL"].fmeasure, 4)


def compute_keyword_overlap(answer: str, keywords: list[str]) -> float:
    """
    Proportion of expected keywords that appear in the generated answer.
    Returns a score between 0.0 and 1.0.
    """
    if not keywords:
        return 1.0  # Out-of-scope: no keywords expected

    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return round(hits / len(keywords), 4)


def check_retrieval_hit(routing_result, expected_sources: list) -> int:
    """
    Check whether at least one expected source document appears
    in the retrieved chunks. Returns 1 (hit) or 0 (miss).
    For out-of-scope, always return 1 (correct behaviour = nothing retrieved).
    """
    if routing_result.route == "out_of_scope":
        return 1  # Correct behavior for out-of-scope

    if not expected_sources:
        return 1  # Synthesis questions: no single expected source

    retrieved_sources = {
        chunk.metadata.get("source", "") for chunk in routing_result.top_chunks
    }
    return int(bool(retrieved_sources.intersection(set(expected_sources))))


def run_evaluation() -> pd.DataFrame:
    """Run the full evaluation pipeline over all 15 test questions."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # Using the global vector_store loaded at the top to avoid DLL conflicts
    global vector_store

    records = []

    print("\n" + "="*80)
    print("RUNNING EVALUATION — 15 TEST QUESTIONS")
    print("="*80 + "\n")

    for q in TEST_QUESTIONS:
        qid = q["id"]
        query = q["query"]
        expected_route = q["expected_route"]
        expected_keywords = q["expected_answer_keywords"]
        expected_sources = q["expected_source_docs"] or []

        print(f"[{qid}] {query[:70]}...")

        # Run the full pipeline
        start = time.time()
        routing_result = route_query(query, vector_store)
        answer = generate_answer(query, routing_result)
        elapsed = round(time.time() - start, 2)

        # Compute metrics
        routing_correct = int(routing_result.route == expected_route)
        retrieval_hit = check_retrieval_hit(routing_result, expected_sources)
        rouge_l = compute_rouge_l(answer, expected_keywords)
        keyword_overlap = compute_keyword_overlap(answer, expected_keywords)

        print(f"  Route: {routing_result.route} (expected: {expected_route}) "
              f"| Correct: {'OK' if routing_correct else 'X'}")
        print(f"  ROUGE-L: {rouge_l} | Keyword Overlap: {keyword_overlap} "
              f"| Retrieval Hit: {retrieval_hit} | Time: {elapsed}s\n")

        records.append({
            "id": qid,
            "query": query,
            "query_type": expected_route,
            "expected_route": expected_route,
            "predicted_route": routing_result.route,
            "routing_correct": routing_correct,
            "retrieval_hit": retrieval_hit,
            "rouge_l": rouge_l,
            "keyword_overlap": keyword_overlap,
            "max_similarity_score": round(routing_result.max_score, 4),
            "sources_retrieved": "|".join(routing_result.sources_found),
            "routing_reason": routing_result.reason,
            "answer_snippet": answer[:150].replace("\n", " "),
            "time_seconds": elapsed
        })

    df = pd.DataFrame(records)

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "evaluation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Overall Routing Accuracy:  {df['routing_correct'].mean():.1%}  "
          f"({df['routing_correct'].sum()}/15)")
    print(f"Overall Retrieval Hit Rate: {df['retrieval_hit'].mean():.1%}  "
          f"({df['retrieval_hit'].sum()}/15)")
    print(f"Mean ROUGE-L Score:         {df['rouge_l'].mean():.4f}")
    print(f"Mean Keyword Overlap:       {df['keyword_overlap'].mean():.4f}")

    print("\nPer-type breakdown:")
    for route_type in ["factual", "synthesis", "out_of_scope"]:
        sub = df[df["query_type"] == route_type]
        print(f"  {route_type.upper():15s} | "
              f"Routing: {sub['routing_correct'].mean():.1%} | "
              f"ROUGE-L: {sub['rouge_l'].mean():.4f} | "
              f"KW Overlap: {sub['keyword_overlap'].mean():.4f}")

    print("\nFull results table:")
    display_cols = ["id", "query_type", "routing_correct",
                    "retrieval_hit", "rouge_l", "keyword_overlap"]
    print(df[display_cols].to_string(index=False))

    return df


if __name__ == "__main__":
    run_evaluation()
