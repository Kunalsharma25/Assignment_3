# ingestion.py
import os
import fitz  # PyMuPDF
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"  # Fix for Windows permission errors
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
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


def extract_text_from_txt(txt_path: str) -> str:
    """Extract raw text from a plain text file."""
    with open(txt_path, 'r', encoding='utf-8') as f:
        return f.read()


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

    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf") or f.endswith(".txt")]
    if not files:
        raise FileNotFoundError(f"No PDF or TXT files found in '{DATA_DIR}'.")

    print(f"Found {len(files)} file(s): {files}")

    all_documents = []

    for file_name in tqdm(files, desc="Parsing and chunking documents"):
        file_path = os.path.join(DATA_DIR, file_name)
        
        if file_name.endswith(".pdf"):
            raw_text = extract_text_from_pdf(file_path)
        else:
            raw_text = extract_text_from_txt(file_path)
            
        chunks = chunk_text(raw_text, CHUNK_SIZE, CHUNK_OVERLAP)

        print(f"  {file_name}: {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": file_name,
                    "chunk_id": i,
                    "total_chunks": len(chunks)
                }
            )
            all_documents.append(doc)

    print(f"\nTotal chunks across all documents: {len(all_documents)}")
    print(f"Loading fastembed model: {EMBEDDING_MODEL}")

    # Initialize FastEmbed (ONNX based, no torch required)
    embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)

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
    # Initialize FastEmbed (same model name as during ingestion)
    embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)
    
    return FAISS.load_local(
        VECTOR_STORE_DIR, embeddings, allow_dangerous_deserialization=True
    )


if __name__ == "__main__":
    ingest_documents()
    print("\nIngestion complete. Run main.py to query the system.")
