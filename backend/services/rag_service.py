import numpy as np
import google.generativeai as genai
import json
import os
import logging
from dotenv import load_dotenv
import chromadb

load_dotenv(override=True)
logger = logging.getLogger(__name__)

# Configure API key only if available
api_key = os.getenv("GEMINI_API_KEY")
if not api_key or not api_key.strip():
    logger.warning(
        "GEMINI_API_KEY environment variable is not set or empty. "
        "RAG features will be unavailable until configured."
    )
else:
    genai.configure(api_key=api_key)

# Load Legal Corpus
CORPUS_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'legal_corpus.json')
legal_corpus = []

try:
    with open(CORPUS_PATH, 'r') as f:
        legal_corpus = json.load(f)

    if not legal_corpus:
        logger.warning(
            "RAG system degraded: legal_corpus.json is empty. "
            "Operating in fallback mode (no legal retrieval)."
        )

except FileNotFoundError:
    logger.warning(
        f"RAG system degraded: legal_corpus.json not found at {CORPUS_PATH}. "
        "Operating in fallback mode."
    )
except Exception as e:
    logger.error(
        f"RAG system degraded: failed to load corpus: {e}"
    )

# Initialize ChromaDB
chroma_host = os.getenv("CHROMA_HOST")
if chroma_host and chroma_host.strip():
    chroma_port = int(os.getenv("CHROMA_PORT", 8000))
    logger.info(f"Connecting to ChromaDB Server at {chroma_host}:{chroma_port}...")
    chroma_client = chromadb.HttpClient(host=chroma_host.strip(), port=chroma_port)
else:
    logger.info("CHROMA_HOST not set. Falling back to local PersistentClient...")
    chroma_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'chroma_db')
    chroma_client = chromadb.PersistentClient(path=chroma_path)

collection = chroma_client.get_or_create_collection(name="legal_corpus")

def get_embeddings(texts: list) -> list:
    try:
        if not texts or not os.getenv("GEMINI_API_KEY"):
            return []
        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=texts,
            task_type="retrieval_document",
        )
        return result['embedding']
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return []

def build_index():
    if not legal_corpus:
        logger.warning(
            "RAG system degraded: skipping Chroma build (empty corpus)."
        )
        return

    logger.info("Building ChromaDB collection...")
    
    # Check if already indexed
    if collection.count() > 0:
        logger.info("ChromaDB collection already populated. Skipping build.")
        return

    embeddings = get_embeddings(legal_corpus)
    
    if not embeddings or len(embeddings) == 0:
        logger.warning(
            "RAG system degraded: embedding generation failed."
        )
        return

    # Upsert into Chroma
    ids = [str(i) for i in range(len(legal_corpus))]
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=legal_corpus
    )
    logger.info("Successfully populated ChromaDB collection.")

def init_index():
    if collection.count() > 0:
        logger.info("ChromaDB index found. Loaded successfully.")
    else:
        logger.info("No cache found. Building ChromaDB index from Gemini API...")
        build_index()

init_index()

def retrieve_relevant_laws(query_text: str, k=2) -> list:
    """Search ChromaDB for the most relevant laws given the document's extracted text or sections"""
    if collection.count() == 0:
        logger.warning(
            "RAG system degraded: ChromaDB collection unavailable or empty. "
            "Returning no legal context."
        )
        return []

    try:
        query_embed = genai.embed_content(
            model="models/gemini-embedding-001",
            content=query_text,
            task_type="retrieval_query",
        )

        results = collection.query(
            query_embeddings=[query_embed["embedding"]],
            n_results=k
        )

        if not results['documents'] or len(results['documents'][0]) == 0:
            logger.warning(
                "RAG system degraded: no retrieval results for query."
            )
            return []

        return results['documents'][0]

    except Exception as e:
        logger.error(
            f"RAG system degraded: retrieval failed: {e}"
        )
        return []
