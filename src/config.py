import os

from dotenv import load_dotenv

load_dotenv()

ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "arxiv_papers")

ARXIV_CATEGORIES = [c.strip() for c in os.getenv("ARXIV_CATEGORIES", "cs.AI,cs.LG").split(",")]
ARXIV_MONTHS_BACK = int(os.getenv("ARXIV_MONTHS_BACK", "18"))
ARXIV_MAX_RESULTS = int(os.getenv("ARXIV_MAX_RESULTS", "20000"))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
