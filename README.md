# Agentic RAG System for AI Regulation Q&A

This project implements a three-layer agentic Q&A system over AI regulation documents from the EU, US, China, and UK. Unlike standard RAG, this system reasons about each query before responding, routing it to the most appropriate strategy.

## System Architecture

1.  **Ingestion Pipeline:** Parses text/PDF documents, performs paragraph-aware chunking, generates embeddings using `BAAI/bge-small-en-v1.5`, and persists them in a FAISS vector store.
2.  **Agentic Query Router:** An explicit, rule-based router that classifies queries into three routes:
    *   **Factual:** Answer exists in a single document.
    *   **Synthesis:** Answer requires combining information from multiple documents or handling contradictions.
    *   **Out of Scope:** Information is not present in the knowledge base.
3.  **Answer Generator:** Generates grounded answers using `gpt-4o-mini` for factual and synthesis routes, while providing a hardcoded refusal for out-of-scope queries to prevent hallucinations.

## Chunking Strategy

*   **Chunk size:** 400 tokens (~1600 characters).
*   **Overlap:** 80 tokens (~320 characters).
*   **Method:** Paragraph-aware splitting (splits on double newlines and then accumulates into chunks).
*   **Justification:** Regulation documents are highly structured. Paragraph-aware splitting ensures semantic units (clauses/articles) are preserved, while the 400-token size captures sufficient context for legal definitions.

## Embedding Model

*   **Model:** `BAAI/bge-small-en-v1.5`
*   **Justification:** This model currently sits at the top of the MTEB leaderboard for retrieval tasks and is free to use locally via `langchain-huggingface`.

## Routing Logic

The system uses a two-layer explicit routing logic:
*   **Layer 1 (Similarity Gate):** If the maximum retrieval similarity score is less than `0.35`, the query is routed as `out_of_scope`.
*   **Layer 2 (Context Signal):** If the query contains synthesis keywords (e.g., "compare", "difference") OR if the top retrieved chunks span 2 or more distinct documents, it is routed as `synthesis`. Otherwise, it defaults to `factual`.

## How to Run

1.  **Environment Setup:**
    ```bash
    # Activation of the existing environment
    # On Windows (cmd): qna_agent1\Scripts\activate.bat
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    ```
2.  **Configure API Key:**
    Update the `.env` file with your `OPENAI_API_KEY`.
3.  **Ingestion:**
    ```bash
    python ingestion.py
    ```
4.  **Interactive Demo:**
    ```bash
    python main.py
    ```
5.  **Evaluation:**
    ```bash
    python evaluator.py
    ```

## Evaluation Methodology

The system is evaluated using 15 human-labeled questions (5 per route). Metrics include:
*   **Routing Accuracy:** Correctness of the agentic classification.
*   **Retrieval Hit Rate:** Whether the expected source documents were retrieved.
*   **ROUGE-L & Keyword Overlap:** Semantic and lexical similarity between the generated answer and expected keywords.

## Failure Analysis

See [FAILURES.md](FAILURES.md) for a detailed analysis of system limitations.
