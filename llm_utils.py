# llm_utils.py
import os
from dotenv import load_dotenv
from openai import OpenAI
from config import LLM_PROVIDER, GROQ_BASE_URL

# Ensure environment is loaded only once
load_dotenv()

def initialize_client():
    """Centralized LLM client initialization."""
    provider = LLM_PROVIDER
    if provider == "groq":
        key = os.getenv("GROQ_API_KEY")
        base_url = GROQ_BASE_URL
        placeholder = "your_groq_api_key_here"
    else:
        key = os.getenv("OPENAI_API_KEY")
        base_url = None
        placeholder = "your_openai_api_key_here"

    # Handle missing/placeholder keys gracefully
    safe_key = key if (key and key != placeholder) else "missing_key"
    
    return OpenAI(api_key=safe_key, base_url=base_url)
