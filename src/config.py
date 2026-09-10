import os

from dotenv import load_dotenv

load_dotenv()

# The embedding model is loaded from a local cache after the first run, so
# skip Hugging Face Hub's version-check request (which otherwise warns about
# unauthenticated rate limits on every run). Set HF_HUB_OFFLINE=0 in .env to
# re-enable it, e.g. if you need to fetch a different model.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "arxiv_papers")
ES_TIMEOUT_SECONDS = int(os.getenv("ES_TIMEOUT_SECONDS", "10"))

ARXIV_CATEGORIES = [c.strip() for c in os.getenv("ARXIV_CATEGORIES", "cs.AI,cs.LG").split(",")]
ARXIV_MONTHS_BACK = int(os.getenv("ARXIV_MONTHS_BACK", "18"))
ARXIV_MAX_RESULTS = int(os.getenv("ARXIV_MAX_RESULTS", "5000"))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))
