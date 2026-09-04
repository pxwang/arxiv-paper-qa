# ArXiv Scientific Paper Summarizer and Q&A

Download recent ArXiv papers in a chosen field, index them in Elasticsearch
with dense-vector embeddings, and chat with the collection — find related
work, compare methodologies, or summarize findings across papers.

## Scope (v1)

- **Category:** `cs.AI` and `cs.LG` (AI / Machine Learning)
- **Time window:** last 1-2 years
- **Data source:** ArXiv's public API (`export.arxiv.org`) — always current,
  no Kaggle account needed. (The Kaggle/S3 snapshot dataset is a fine
  alternative later if you want the full historical corpus.)

## Architecture

```text
ArXiv API (export.arxiv.org)
    │  metadata + abstracts, filtered by category + date
    ▼
Ingestion (src/ingest.py)
    │  embed abstracts with BAAI/bge-small-en-v1.5 (local, CPU)
    ▼
Elasticsearch (dense_vector field + text fields)
    │  hybrid search: BM25 + kNN
    ▼
RAG chain (src/rag.py, LangChain)
    │  retrieve top-k chunks → prompt → local LLM
    ▼
Ollama (local LLM, Metal-accelerated on Apple Silicon)
    │
    ▼
Answer, with source paper citations
```

## Prerequisites

1. **Docker** (for Elasticsearch) — already installed.
2. **Ollama** (for the local LLM):
   ```bash
   brew install ollama
   ollama pull llama3.1:8b
   ```
3. **Python 3.11+** in a virtualenv:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   > If you're on a very new Python (3.13+) and `torch`/`sentence-transformers`
   > fail to install, use `pyenv` to install Python 3.11 or 3.12 and recreate
   > the venv with that instead — ML package wheels lag behind new Python
   > releases.

## Setup

```bash
# 1. Start Elasticsearch
docker compose up -d

# 2. Copy env template and adjust if needed
cp .env.example .env

# 3. Ingest papers (cs.AI + cs.LG, last ~18 months)
python -m src.ingest

# 4. Ask questions
python -m src.rag "What are recent approaches to reducing LLM hallucination?"
```

### Caching

Fetched papers are cached to `data/papers.jsonl`. Re-running `python -m
src.ingest` reuses that cache instead of re-hitting the ArXiv API — handy
when re-indexing after changing the embedding model or index mapping, since
a full ~18-month fetch can take over an hour due to ArXiv's API rate limit.
Pass `--refresh` to force a fresh fetch:

```bash
python -m src.ingest --refresh
```

## Project layout

```text
src/
  config.py   - settings: ES connection, index name, category, date window, model names
  ingest.py   - fetch from ArXiv API, embed abstracts, index into Elasticsearch
  rag.py      - retrieve + generate: hybrid search in ES, then answer via Ollama
docker-compose.yml - single-node Elasticsearch for local dev
requirements.txt
.env.example
```

## Notes on scale

Scoped to `cs.AI`/`cs.LG` rather than the full ~2.7M-paper ArXiv corpus.
Observed throughput is ~270 papers/day across both categories combined, so
18 months is roughly 150,000 papers. At 384-dim embeddings
(`bge-small-en-v1.5`), that's small in practice:

| Component | Estimate |
|---|---|
| Raw vectors (384 dims × 4 bytes × 150k docs) | ~230 MB |
| HNSW graph overhead (default `m=16`) | ~20 MB |
| Text (title/abstract, indexed + stored) | ~600 MB - 1 GB |
| **Total Elasticsearch data size** | **~1-1.5 GB** |

Comfortably within the 2GB JVM heap in `docker-compose.yml`, with room to
spare on a 16GB machine even with Ollama's model loaded. The actual
bottleneck for a full 18-month ingest is ArXiv's API rate limit (~75
minutes), not memory — see the Caching section above for avoiding repeat
fetches.

## Status

Early scaffold — ingestion and RAG pipeline are functional for the MVP
scope above. Next steps: chunking for full-text (currently abstract-only),
re-ranking, and a simple chat UI.
