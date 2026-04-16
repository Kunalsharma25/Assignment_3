# evaluator.py
import os
import time
import pandas as pd
from dotenv import load_dotenv
from rouge_score import rouge_scorer
from ingestion import load_vector_store
from router import route_query
from generator import generate_answer
from test_questions import TEST_QUESTIONS
from config import RESULTS_DIR

# Ensure environment variables are loaded
load_dotenv()

# Load vector store globally before other libraries can conflict
vector_store = load_vector_store()

def compute_rouge_l(hypothesis: str, reference_keywords: list[str]) -> float:
    """Compute ROUGE-L between generated answer and pseudo-reference from keywords."""
    if not reference_keywords:
        return 1.0  # Out-of-scope success
    reference = " ".join(reference_keywords)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return round(scores["rougeL"].fmeasure, 4)

def compute_keyword_overlap(answer: str, keywords: list[str]) -> float:
    """Proportion of expected keywords present in the generated answer."""
    if not keywords:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return round(hits / len(keywords), 4)

def check_retrieval_hit(routing_result, expected_sources: list) -> int:
    """Check if at least one expected source document was retrieved."""
    if routing_result.route == "out_of_scope":
        return 1
    if not expected_sources:
        return 1
    retrieved_sources = {
        chunk.metadata.get("source", "") for chunk in routing_result.top_chunks
    }
    return int(bool(retrieved_sources.intersection(set(expected_sources))))

def run_evaluation() -> pd.DataFrame:
    """Run full evaluation pipeline with real-time progress flushing."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    records = []

    print("\n" + "="*80, flush=True)
    print("RUNNING EVALUATION — 15 TEST QUESTIONS", flush=True)
    print("="*80 + "\n", flush=True)

    for q in TEST_QUESTIONS:
        qid = q["id"]
        query = q["query"]
        expected_route = q["expected_route"]
        expected_keywords = q["expected_answer_keywords"]
        expected_sources = q["expected_source_docs"] or []

        print(f"[{qid}] {query[:70]}...", end=" ", flush=True)

        # Execution
        start = time.time()
        routing_result = route_query(query, vector_store)
        answer = generate_answer(query, routing_result)
        elapsed = round(time.time() - start, 2)

        # Metric calculation
        routing_correct = int(routing_result.route == expected_route)
        retrieval_hit = check_retrieval_hit(routing_result, expected_sources)
        rouge_l = compute_rouge_l(answer, expected_keywords)
        keyword_overlap = compute_keyword_overlap(answer, expected_keywords)

        icon = "OK" if routing_correct else "X"
        print(f"-> {routing_result.route} [{icon}] ({elapsed}s)", flush=True)

        records.append({
            "id": qid,
            "query": query,
            "query_type": expected_route,
            "routing_correct": routing_correct,
            "retrieval_hit": retrieval_hit,
            "rouge_l": rouge_l,
            "keyword_overlap": keyword_overlap,
            "max_score": round(routing_result.max_score, 4),
            "predicted_route": routing_result.route,
            "reason": routing_result.reason
        })

    df = pd.DataFrame(records)
    csv_path = os.path.join(RESULTS_DIR, "evaluation_results.csv")
    df.to_csv(csv_path, index=False)
    
    # Final Summary
    print("\n" + "="*80, flush=True)
    print("SUMMARY STATISTICS", flush=True)
    print("="*80, flush=True)
    print(f"Overall Routing Accuracy:  {df['routing_correct'].mean():.1%}", flush=True)
    print(f"Overall Retrieval Hit Rate: {df['retrieval_hit'].mean():.1%}", flush=True)
    print(df[["id", "query_type", "routing_correct", "retrieval_hit"]].to_string(index=False), flush=True)

    return df

if __name__ == "__main__":
    run_evaluation()
