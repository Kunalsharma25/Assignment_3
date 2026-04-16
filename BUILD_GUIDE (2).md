# Agentic RAG System — Complete Build Guide

> **Assignment:** AI Engineer Intern — Assignment 3: Agentic RAG & Evaluation  
> **Deadline:** 3 days from receiving the assignment  
> **Submission:** https://docs.google.com/forms/d/e/1FAIpQLScDUFoy3njfDve5ZCxTdbBOd9pwLOXi_yOFGbvSdk33JQR8uw/viewform?usp=header

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Final Project Structure](#2-final-project-structure)
3. [Environment Setup](#3-environment-setup)
4. [Phase 1 — Ingestion Pipeline](#4-phase-1--ingestion-pipeline)
5. [Phase 2 — Agentic Query Router](#5-phase-2--agentic-query-router)
6. [Phase 3 — Answer Generator](#6-phase-3--answer-generator)
7. [Phase 4 — Evaluation Framework](#7-phase-4--evaluation-framework)
8. [Phase 5 — Main Entry Point](#8-phase-5--main-entry-point)
9. [Phase 6 — README.md](#9-phase-6--readmemd)
10. [Phase 7 — FAILURES.md](#10-phase-7--failuresmd)
11. [Verification Checklist](#11-verification-checklist)
12. [Video Demo Guide](#12-video-demo-guide)

---

## 1. Project Overview

You are building a **three-layer agentic Q&A system** over 4 AI regulation documents.

### What makes it "agentic"

Unlike standard RAG (which always retrieves and always answers), your system **reasons before responding**. For every query, it first decides *how* to handle it, then acts accordingly:

| Route | Meaning | System Action |
|---|---|---|
| **Factual** | Answer exists directly in one chunk/document | Retrieve top-3 chunks, generate grounded answer |
| **Synthesis** | Answer requires combining info from multiple documents | Retrieve top-7 chunks across sources, synthesize with contradiction handling |
| **Out of Scope** | Information is not in the documents | Return a hardcoded decline message — **no LLM call, no hallucination** |

### Hard constraints from the assignment

- Routing logic must be **explicit and inspectable** — no "ask the LLM to classify"
- **No LangChain agents** — LangChain allowed only for embeddings/vector store
- Out-of-scope hallucination is an **automatic disqualifier**
- Evaluation must be **runnable code** producing a results table
- Evaluation script must run with **a single command**

---

## 2. Final Project Structure

```
agentic-rag/
│
├── data/                        # Downloaded PDFs go here (not committed to git)
│   ├── document1.pdf
│   ├── document2.pdf
│   ├── document3.pdf
│   └── document4.pdf
│
├── vector_store/                # Auto-created by ingestion.py (FAISS index files)
│   ├── index.faiss
│   └── index.pkl
│
├── ingestion.py                 # Phase 1: Parse, chunk, embed, store
├── router.py                    # Phase 2: Classify query type
├── generator.py                 # Phase 3: Generate answers per route
├── evaluator.py                 # Phase 4: Run 15 test questions, produce CSV
├── main.py                      # Phase 5: Interactive demo entry point
│
├── test_questions.py            # 15 labelled test questions (used by evaluator)
├── config.py                    # Central config: paths, thresholds, model names
│
├── results/                     # Auto-created by evaluator.py
│   └── evaluation_results.csv
│
├── requirements.txt
├── .gitignore
├── README.md
└── FAILURES.md
```

---

## 3. Environment Setup

### Step 3.1 — Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### Step 3.2 — Install dependencies

Create `requirements.txt` with the following content:

```txt
# PDF parsing
PyMuPDF==1.24.3

# Embeddings and vector store
langchain-community==0.2.5
langchain-huggingface==0.0.3
faiss-cpu==1.8.0
sentence-transformers==3.0.1

# LLM API
openai==1.35.3

# NLP for routing
spacy==3.7.5

# Evaluation metrics
rouge-score==0.1.2
scikit-learn==1.5.0
numpy==1.26.4

# Utilities
python-dotenv==1.0.1
pandas==2.2.2
tqdm==4.66.4
```

Install everything:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Step 3.3 — Create a `.env` file

```bash
# .env
OPENAI_API_KEY=your_openai_api_key_here
```

### Step 3.4 — Create `.gitignore`

```gitignore
venv/
.env
data/
vector_store/
results/
__pycache__/
*.pyc
.DS_Store
```

### Step 3.5 — Create `config.py`

This file centralises all tunable parameters so nothing is hardcoded across modules.

```python
# config.py
import os

# Paths
DATA_DIR = "data/"
VECTOR_STORE_DIR = "vector_store/"
RESULTS_DIR = "results/"

# Chunking
CHUNK_SIZE = 400          # tokens (characters / 4 approx)
CHUNK_OVERLAP = 80        # token overlap between consecutive chunks

# Embedding model (free, no API key needed)
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# LLM
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.0     # deterministic answers
MAX_TOKENS = 512

# Retrieval
TOP_K_FACTUAL = 3
TOP_K_SYNTHESIS = 7

# Routing thresholds
OUT_OF_SCOPE_THRESHOLD = 0.35   # cosine similarity below this → out of scope
SYNTHESIS_SOURCE_MIN = 2        # top chunks from >= N different docs → synthesis

# Synthesis keywords (linguistic signal layer)
SYNTHESIS_KEYWORDS = [
    "compare", "comparison", "difference", "differences",
    "contrast", "across", "between", "multiple", "various",
    "all documents", "both", "each", "overall", "collectively",
    "how do", "how does", "in general", "broadly"
]
```

---

## 4. Phase 1 — Ingestion Pipeline

**File:** `ingestion.py`

### What this file does

1. Reads all PDFs from `data/` using PyMuPDF
2. Splits text into overlapping chunks (paragraph-aware, max 400 tokens)
3. Tags each chunk with its source document name
4. Embeds chunks using `BAAI/bge-small-en-v1.5`
5. Stores everything in a FAISS vector store

### Chunking strategy — justification

- **Chunk size 400 tokens:** AI regulation documents use structured clauses. 400 tokens captures a full clause plus surrounding context without being too noisy.
- **Overlap 80 tokens:** Prevents answers that span a clause boundary from being missed. ~20% overlap is the standard sweet spot.
- **Paragraph-aware splitting:** Text is first split on double newlines (paragraph boundaries) before token-length splitting. This preserves semantic units.
- **Source tagging:** Every chunk stores `{"source": "document_name.pdf"}` as metadata. This is critical for the router to detect multi-source retrieval.

### Full code

```python
# ingestion.py
import os
import pickle
import fitz  # PyMuPDF
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from config import (DATA_DIR, VECTOR_STORE_DIR, CHUNK_SIZE,
                    CHUNK_OVERLAP, EMBEDDING_MODEL)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract raw text from a PDF file using PyMuPDF."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Paragraph-aware chunking with max token cap and overlap.
    Steps:
      1. Split on paragraph boundaries (double newlines)
      2. Accumulate paragraphs until chunk_size is reached
      3. Apply overlap by carrying forward the last `overlap` characters
    """
    # Approximate token count as len(text) / 4
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        # If adding this paragraph exceeds chunk_size, save current and start new
        if len(current_chunk) + len(para) > chunk_size * 4:  # *4: chars to tokens
            if current_chunk:
                chunks.append(current_chunk.strip())
            # Start new chunk with overlap from previous
            overlap_text = current_chunk[-overlap * 4:] if current_chunk else ""
            current_chunk = overlap_text + " " + para
        else:
            current_chunk += " " + para
    
    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def ingest_documents() -> FAISS:
    """
    Full ingestion pipeline:
    Parse → Chunk → Embed → Index → Save
    Returns the FAISS vector store.
    """
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in '{DATA_DIR}'. "
                                f"Download the dataset and place it there.")

    print(f"Found {len(pdf_files)} PDF(s): {pdf_files}")

    all_documents = []

    for pdf_file in tqdm(pdf_files, desc="Parsing and chunking PDFs"):
        pdf_path = os.path.join(DATA_DIR, pdf_file)
        raw_text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(raw_text, CHUNK_SIZE, CHUNK_OVERLAP)

        print(f"  {pdf_file}: {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": pdf_file,
                    "chunk_id": i,
                    "total_chunks": len(chunks)
                }
            )
            all_documents.append(doc)

    print(f"\nTotal chunks across all documents: {len(all_documents)}")
    print(f"Loading embedding model: {EMBEDDING_MODEL}")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}  # Required for cosine similarity
    )

    print("Building FAISS index...")
    vector_store = FAISS.from_documents(all_documents, embeddings)

    # Persist to disk
    vector_store.save_local(VECTOR_STORE_DIR)
    print(f"Vector store saved to '{VECTOR_STORE_DIR}'")

    return vector_store


def load_vector_store() -> FAISS:
    """Load an existing FAISS vector store from disk."""
    if not os.path.exists(os.path.join(VECTOR_STORE_DIR, "index.faiss")):
        raise FileNotFoundError(
            "Vector store not found. Run ingestion first: python ingestion.py"
        )
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return FAISS.load_local(
        VECTOR_STORE_DIR, embeddings, allow_dangerous_deserialization=True
    )


if __name__ == "__main__":
    ingest_documents()
    print("\nIngestion complete. Run main.py to query the system.")
```

### Run ingestion

```bash
python ingestion.py
```

Expected output:
```
Found 4 PDF(s): [...]
  document1.pdf: 42 chunks
  document2.pdf: 38 chunks
  ...
Total chunks across all documents: 163
Loading embedding model: BAAI/bge-small-en-v1.5
Building FAISS index...
Vector store saved to 'vector_store/'
```

---

## 5. Phase 2 — Agentic Query Router

**File:** `router.py`

### What this file does

Classifies every incoming query into one of three types **before** any LLM is called. The routing logic uses **two explicit, inspectable layers**:

**Layer 1 — Retrieval Score Gate (Out-of-Scope detection)**
- Run a similarity search for the query
- If the highest cosine similarity score < `OUT_OF_SCOPE_THRESHOLD` (0.35), the topic is not in the knowledge base → route as `out_of_scope`

**Layer 2 — Linguistic + Source Distribution Signal (Factual vs. Synthesis)**
- Check for synthesis keywords ("compare", "difference", "across", etc.)
- Count how many distinct source documents appear in the top-K results
- If ≥ 2 sources AND/OR synthesis keywords present → `synthesis`
- Otherwise → `factual`

> **Why this is "explicit and inspectable":** Every routing decision can be explained: "Query scored 0.28 similarity (below threshold 0.35) → out_of_scope" or "Top 5 chunks came from 3 different documents and query contains 'compare' → synthesis."

### Full code

```python
# router.py
from dataclasses import dataclass
from langchain_community.vectorstores import FAISS
from config import (OUT_OF_SCOPE_THRESHOLD, SYNTHESIS_SOURCE_MIN,
                    SYNTHESIS_KEYWORDS, TOP_K_SYNTHESIS)


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
    Explicit, inspectable routing logic.
    
    Decision tree:
      1. Retrieve top-K chunks with similarity scores
      2. If max_score < OUT_OF_SCOPE_THRESHOLD  → out_of_scope
      3. If synthesis keywords in query OR sources >= SYNTHESIS_SOURCE_MIN → synthesis
      4. Otherwise → factual
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
    # FAISS returns L2 distance; convert to similarity (lower distance = higher similarity)
    # For normalized vectors: similarity = 1 - (distance / 2)
    raw_scores = [r[1] for r in results_with_scores]
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
            reason=(
                f"Max similarity score {max_score:.3f} is below threshold "
                f"{OUT_OF_SCOPE_THRESHOLD}. Topic likely not in the knowledge base."
            )
        )

    # Step 3: Identify unique sources in top results
    sources_found = list({c.metadata.get("source", "unknown") for c in chunks})
    query_lower = query.lower()

    # Step 4: Check synthesis signals
    matched_keywords = [kw for kw in SYNTHESIS_KEYWORDS if kw in query_lower]
    multi_source = len(sources_found) >= SYNTHESIS_SOURCE_MIN

    if matched_keywords or multi_source:
        reason_parts = []
        if matched_keywords:
            reason_parts.append(f"Synthesis keywords detected: {matched_keywords}")
        if multi_source:
            reason_parts.append(
                f"Top chunks span {len(sources_found)} source documents: {sources_found}"
            )
        return RoutingResult(
            route="synthesis",
            top_chunks=chunks,
            scores=scores,
            max_score=max_score,
            sources_found=sources_found,
            reason="; ".join(reason_parts)
        )

    # Step 5: Default to factual
    return RoutingResult(
        route="factual",
        top_chunks=chunks[:3],  # Top 3 most relevant chunks
        scores=scores[:3],
        max_score=max_score,
        sources_found=sources_found,
        reason=(
            f"Max score {max_score:.3f} above threshold. "
            f"No synthesis signals found. Single-document factual answer."
        )
    )
```

---

## 6. Phase 3 — Answer Generator

**File:** `generator.py`

### What this file does

Takes a `RoutingResult` and generates the appropriate response:

- **Factual:** Sends top-3 chunks to GPT, instructs it to answer only from provided context
- **Synthesis:** Sends top-7 chunks from multiple sources, instructs it to synthesize and flag contradictions
- **Out of scope:** Returns a hardcoded message — **no LLM call is made**

### Full code

```python
# generator.py
import os
from openai import OpenAI
from router import RoutingResult
from config import LLM_MODEL, LLM_TEMPERATURE, MAX_TOKENS

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


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
```

---

## 7. Phase 4 — Evaluation Framework

### Step 7.1 — Define 15 test questions

**File:** `test_questions.py`

Write exactly 5 questions per category. For each question, define:
- `expected_route`: the correct classification
- `expected_answer_keywords`: keywords that should appear in a correct answer (used for keyword overlap metric)
- `expected_source_docs`: which PDFs should be retrieved (used for retrieval accuracy)

```python
# test_questions.py

TEST_QUESTIONS = [

    # ──────────────────────────────────────────────
    # FACTUAL QUESTIONS (5) — Answer in one document
    # ──────────────────────────────────────────────
    {
        "id": "F1",
        "query": "What is the EU AI Act?",
        "expected_route": "factual",
        "expected_answer_keywords": ["European Union", "regulation", "artificial intelligence", "risk"],
        "expected_source_docs": ["eu_ai_act.pdf"],   # Update with actual filenames
        "notes": "Definition query, answer exists in a single document"
    },
    {
        "id": "F2",
        "query": "What are high-risk AI systems according to the EU AI Act?",
        "expected_route": "factual",
        "expected_answer_keywords": ["high-risk", "biometric", "critical infrastructure", "safety"],
        "expected_source_docs": ["eu_ai_act.pdf"],
        "notes": "Specific classification query from one document"
    },
    {
        "id": "F3",
        "query": "What penalties does the EU AI Act impose for non-compliance?",
        "expected_route": "factual",
        "expected_answer_keywords": ["fine", "penalty", "million", "turnover", "sanction"],
        "expected_source_docs": ["eu_ai_act.pdf"],
        "notes": "Specific factual detail, single source"
    },
    {
        "id": "F4",
        "query": "What is the role of the AI Office under EU regulation?",
        "expected_route": "factual",
        "expected_answer_keywords": ["AI Office", "oversight", "enforcement", "Commission"],
        "expected_source_docs": ["eu_ai_act.pdf"],
        "notes": "Institutional role question"
    },
    {
        "id": "F5",
        "query": "What transparency obligations apply to AI systems that interact with humans?",
        "expected_route": "factual",
        "expected_answer_keywords": ["transparency", "disclosure", "human", "chatbot", "inform"],
        "expected_source_docs": ["eu_ai_act.pdf"],
        "notes": "Obligation query from specific document section"
    },

    # ──────────────────────────────────────────────────────────────────
    # SYNTHESIS QUESTIONS (5) — Answer requires combining multiple docs
    # ──────────────────────────────────────────────────────────────────
    {
        "id": "S1",
        "query": "How do different regulatory frameworks compare in their approach to AI risk classification?",
        "expected_route": "synthesis",
        "expected_answer_keywords": ["risk", "classification", "framework", "approach", "differ"],
        "expected_source_docs": None,  # Should pull from multiple docs
        "notes": "Comparison query across all documents"
    },
    {
        "id": "S2",
        "query": "What are the differences between the EU and US approaches to AI regulation?",
        "expected_route": "synthesis",
        "expected_answer_keywords": ["EU", "US", "United States", "difference", "approach"],
        "expected_source_docs": None,
        "notes": "Direct comparison — synthesis keyword 'differences' present"
    },
    {
        "id": "S3",
        "query": "How do various AI governance documents address algorithmic transparency?",
        "expected_route": "synthesis",
        "expected_answer_keywords": ["transparency", "algorithmic", "explainability", "accountability"],
        "expected_source_docs": None,
        "notes": "Cross-document theme query"
    },
    {
        "id": "S4",
        "query": "Across all documents, what common principles for responsible AI development emerge?",
        "expected_route": "synthesis",
        "expected_answer_keywords": ["responsible", "principles", "fairness", "safety", "accountability"],
        "expected_source_docs": None,
        "notes": "'Across all documents' is an explicit synthesis signal"
    },
    {
        "id": "S5",
        "query": "How do the documents collectively address the regulation of generative AI?",
        "expected_route": "synthesis",
        "expected_answer_keywords": ["generative", "large language model", "foundation model", "GPAI"],
        "expected_source_docs": None,
        "notes": "Multi-document synthesis on a specific technology"
    },

    # ─────────────────────────────────────────────────────────────────────
    # OUT-OF-SCOPE QUESTIONS (5) — Topic not present in any document
    # ─────────────────────────────────────────────────────────────────────
    {
        "id": "O1",
        "query": "What is the GDP of Germany in 2024?",
        "expected_route": "out_of_scope",
        "expected_answer_keywords": [],
        "expected_source_docs": [],
        "notes": "Completely unrelated topic — economic statistics"
    },
    {
        "id": "O2",
        "query": "How do I train a neural network from scratch using PyTorch?",
        "expected_route": "out_of_scope",
        "expected_answer_keywords": [],
        "expected_source_docs": [],
        "notes": "Technical ML tutorial — not in regulation documents"
    },
    {
        "id": "O3",
        "query": "Who won the FIFA World Cup in 2022?",
        "expected_route": "out_of_scope",
        "expected_answer_keywords": [],
        "expected_source_docs": [],
        "notes": "Completely off-topic"
    },
    {
        "id": "O4",
        "query": "What are the best Python libraries for data visualisation?",
        "expected_route": "out_of_scope",
        "expected_answer_keywords": [],
        "expected_source_docs": [],
        "notes": "Software recommendation — unrelated to AI regulation"
    },
    {
        "id": "O5",
        "query": "What is the current stock price of NVIDIA?",
        "expected_route": "out_of_scope",
        "expected_answer_keywords": [],
        "expected_source_docs": [],
        "notes": "Real-time financial data — not in static documents"
    },
]
```

> **Important:** After downloading the 4 PDFs, update `expected_source_docs` in the Factual questions to match the actual filenames in your `data/` folder.

---

### Step 7.2 — Write the evaluation script

**File:** `evaluator.py`

This script:
1. Loads the vector store
2. Runs all 15 test questions through the full pipeline
3. Computes three metrics per question:
   - **Routing Accuracy:** Did it classify correctly? (1 or 0)
   - **Retrieval Hit:** Is at least one expected source document in the top-K results?
   - **Answer Quality:** ROUGE-L score between generated answer and keyword-joined reference
4. Saves results to `results/evaluation_results.csv`
5. Prints a formatted summary table

```python
# evaluator.py
import os
import time
import pandas as pd
from rouge_score import rouge_scorer
from dotenv import load_dotenv

from ingestion import load_vector_store
from router import route_query
from generator import generate_answer
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
    vector_store = load_vector_store()

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
              f"| Correct: {'✓' if routing_correct else '✗'}")
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
```

### Run evaluation (single command as required)

```bash
python evaluator.py
```

---

## 8. Phase 5 — Main Entry Point

**File:** `main.py`

This is the interactive demo used in your video walkthrough.

```python
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
```

### Run the interactive demo

```bash
python main.py
```

---

## 9. Phase 6 — README.md

Your README must cover: system architecture, chunking choices, routing logic, evaluation methodology, and results summary. Use this template:

```markdown
# Agentic RAG System for AI Regulation Q&A

## System Architecture

[Brief description of the three-layer architecture: Ingestion → Router → Generator]

## Chunking Strategy

- **Chunk size:** 400 tokens
- **Overlap:** 80 tokens (~20%)
- **Method:** Paragraph-aware splitting
- **Justification:** [Explain why in 2-3 sentences]

## Embedding Model

- **Model:** BAAI/bge-small-en-v1.5
- **Justification:** [Free, no API key, state-of-the-art for retrieval tasks]

## Routing Logic

[Describe the two-layer explicit routing logic in plain English. Include the threshold value and the synthesis keyword list.]

## Evaluation Methodology

- 15 test questions (5 per type)
- Metrics: Routing Accuracy, Retrieval Hit Rate, ROUGE-L, Keyword Overlap

## Results Summary

[Paste your summary table here after running evaluator.py]

## How to Run

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
# Place PDFs in data/ folder
python ingestion.py
python main.py           # Interactive demo
python evaluator.py      # Evaluation (single command)
```

## Failure Analysis

See [FAILURES.md](FAILURES.md)
```

---

## 10. Phase 7 — FAILURES.md

The assignment requires **at least 3 honest, specific failure cases**. Write this after running your evaluation and examining the results.

```markdown
# Failure Analysis

## Failure 1 — Synthesis/Factual Boundary Misclassification

**Description:** Queries like "What transparency requirements does the EU AI Act impose?"
are classified as `synthesis` even though the answer exists in a single document,
because the word "requirements" appears in multiple chunks across different source files.

**Root Cause:** The multi-source signal in the router fires based on top-K chunk
distribution. When a concept (e.g., transparency) appears in multiple documents,
the top-7 chunks naturally span several sources, triggering `synthesis` even for
factual questions.

**What I would do differently:** Add a confidence gap signal: if one source has a
score ≥0.1 higher than chunks from other sources, treat it as factual regardless
of source count.

---

## Failure 2 — Low ROUGE-L for Out-of-Scope Questions

**Description:** ROUGE-L scores for out-of-scope queries appear as 1.0 artificially
because we return 1.0 when no keywords are expected. This inflates aggregate metrics
and doesn't reflect whether the decline message is well-worded.

**Root Cause:** The evaluation metric design doesn't assess out-of-scope *response quality*,
only whether the route was correct.

**What I would do differently:** Add a separate "is_hallucination" binary metric for
out-of-scope answers — checking whether the response contains any factual claims beyond
the decline message.

---

## Failure 3 — Threshold Sensitivity for Near-Boundary Queries

**Description:** Queries that touch on AI regulation tangentially (e.g., "What is machine
learning?") may score between 0.30–0.40, making them borderline between `out_of_scope`
and `factual`. The fixed threshold of 0.35 can misclassify these.

**Root Cause:** A static threshold cannot account for query length and specificity
variation. Short, vague queries produce lower similarity scores than specific ones
even when they are in-scope.

**What I would do differently:** Use a dynamic threshold based on query length or
implement a secondary pass: for borderline scores (0.30–0.40), retrieve the top
chunk's full text and use a lightweight classifier to confirm relevance.
```

---

## 11. Verification Checklist

Before submitting, verify every item below:

### Code Completeness
- [ ] `ingestion.py` runs without errors and creates `vector_store/`
- [ ] `router.py` routing logic is rule-based (not an LLM call)
- [ ] `generator.py` never calls the LLM for `out_of_scope` queries
- [ ] `evaluator.py` runs with `python evaluator.py` (single command)
- [ ] `main.py` runs interactively and shows routing decisions clearly

### Assignment Requirements
- [ ] 4 documents ingested from the provided Google Drive link
- [ ] 15 test questions (exactly 5 per type) in `test_questions.py`
- [ ] Evaluation produces a CSV file in `results/`
- [ ] README covers all required sections
- [ ] FAILURES.md has 3+ specific, honest failure cases
- [ ] No LangChain agents used (only LangChain embeddings/vectorstore)
- [ ] `requirements.txt` is complete and up to date

### Quality Checks
- [ ] Out-of-scope queries NEVER produce hallucinated factual content
- [ ] Routing reason is printed and is human-readable
- [ ] Code is split into separate files (not one giant script)
- [ ] GitHub repo is clean (no `.env`, no `data/`, no `venv/`)

---

## 12. Video Demo Guide

Your video must be under 7 minutes and show three things live:

### Part 1 — All Three Query Types (~3 min)
Run `python main.py` and demonstrate:

1. A **Factual** query → show routing decision printed, answer generated, source cited
2. A **Synthesis** query → show multi-source retrieval, synthesized answer, contradiction handling if present
3. An **Out-of-scope** query → show routing decision, show the decline message, emphasize no LLM was called

### Part 2 — Evaluation Script (~2 min)
Run `python evaluator.py` and show:
- The 15 questions being processed
- The summary table printed to terminal
- The CSV file created in `results/`

### Part 3 — Failure Case Walkthrough (~2 min)
Pick Failure 1 (boundary misclassification). Show:
- A query that gets misclassified
- Print the routing reason to explain *why* it was misclassified
- Briefly explain what the fix would be

---

*Build this in the order: ingestion → router → generator → evaluator → main. Each module is independent; test each one before moving to the next.*
