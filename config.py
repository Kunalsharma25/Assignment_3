# config.py
import os

# Paths
DATA_DIR = "data/"
VECTOR_STORE_DIR = "vector_store/"
RESULTS_DIR = "results/"

# Chunking
CHUNK_SIZE = 400          # tokens (characters / 4 approx)
CHUNK_OVERLAP = 80        # token overlap between consecutive chunks

# Embeddings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LLM
LLM_MODEL = "llama-3.1-8b-instant"  # High rate-limit model
LLM_TEMPERATURE = 0.0     # deterministic answers
MAX_TOKENS = 1024

# LLM Provider (Switch to Groq)
LLM_PROVIDER = "groq"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Retrieval
TOP_K_FACTUAL = 3
TOP_K_SYNTHESIS = 7

# Routing thresholds
OUT_OF_SCOPE_THRESHOLD = 0.40   # cosine similarity below this → out of scope
SYNTHESIS_SOURCE_MIN = 2        # top chunks from >= N different docs → synthesis
CONFIDENCE_GAP_THRESHOLD = 0.02 # gap between top two source scores below this → synthesis
SYNTHESIS_MIN_CONFIDENCE = 0.60 # multi-source synthesis only triggers if max_score > this
SCORE_SPREAD_THRESHOLD = 0.015  # noise filter: reject if scores are too "flat"/similar

# Synthesis keywords (linguistic signal layer)
SYNTHESIS_KEYWORDS = [
    "compare", "comparison", "difference", "differences",
    "contrast", "across", "between", "multiple", "various",
    "all documents", "both", "each", "overall", "collectively",
    "how do", "how does", "in general", "broadly"
]
