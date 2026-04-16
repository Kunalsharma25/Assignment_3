# test_router.py
from ingestion import load_vector_store
from router import route_query

def test_router():
    print("Loading vector store...")
    try:
        vector_store = load_vector_store()
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return

    test_queries = [
        # --- FACTUAL ---
        {
            "type": "FACTUAL",
            "query": "What are the four risk categories defined by the EU AI Act?"
        },
        {
            "type": "FACTUAL (Source Specific)",
            "query": "Which body was created by the EU AI Act to oversee implementation?"
        },
        {
            "type": "FACTUAL (Source Specific)",
            "query": "What is the threshold for frontier models mentioned in the US Executive Order?"
        },
        
        # --- SYNTHESIS ---
        {
            "type": "SYNTHESIS (Keywords)",
            "query": "What are the differences between EU and US AI regulations?"
        },
        {
            "type": "SYNTHESIS (Comparison)",
            "query": "Compare the UK's approach to AI regulation with the EU's prescriptive model."
        },
        {
            "type": "SYNTHESIS (Comparison)",
            "query": "How do the US and China differ in their regulation of generative AI based on these texts?"
        },
        {
            "type": "SYNTHESIS (Thematic)",
            "query": "What are the common data governance themes mentioned across all the documents?"
        },
        {
            "type": "SYNTHESIS (Thematic)",
            "query": "How do the documents collectively address the issue of human oversight?"
        },

        # --- OUT OF SCOPE ---
        {
            "type": "OUT_OF_SCOPE (Off-topic)",
            "query": "Who won the FIFA World Cup in 2022?"
        },
        {
            "type": "OUT_OF_SCOPE (Scientific)",
            "query": "What is the distance from the Earth to the Moon?"
        },
        {
            "type": "OUT_OF_SCOPE (Literary)",
            "query": "Who wrote the play Hamlet?"
        },
        {
            "type": "OUT_OF_SCOPE (DIY)",
            "query": "How can I fix a leaking faucet at home?"
        }
    ]

    print("\n" + "="*80)
    print("EXPANDED ROUTER TEST RESULTS")
    print("="*80)

    for item in test_queries:
        query = item["query"]
        q_type = item["type"]
        
        print(f"\n[Test Type: {q_type}]")
        print(f"Query: {query}")
        
        result = route_query(query, vector_store)
        
        print(f"Decision: {result.route.upper()}")
        print(f"Max Similarity Score: {result.max_score:.4f}")
        print(f"Sources Found: {result.sources_found}")
        print(f"Reason: {result.reason}")
        print("-" * 40)

if __name__ == "__main__":
    test_router()
