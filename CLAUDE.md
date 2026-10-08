# CLAUDE.md

Guidance for Claude Code when working in this repo. See README.md for the
full user-facing docs; this file covers what matters when changing code.

## What this is

RAG over recent ArXiv `cs.AI`/`cs.LG` abstracts: ArXiv API → embed with
`BAAI/bge-small-en-v1.5` → Elasticsearch (BM25 + kNN) → RRF fusion →
cross-encoder re-rank (`BAAI/bge-reranker-base`) → local LLM via Ollama
(`llama3.1:8b`). CLI in `src/rag.py`, Streamlit UI in `app.py`.

## Commands

```bash
source .venv/bin/activate          # Python 3.11 (CI pins 3.11)
pip install -r requirements-dev.txt
ruff check .                       # lint (CI runs this first)
pytest -v                          # tests; no live services needed
docker compose up -d               # Elasticsearch on :9200
python -m src.ingest [--refresh]   # fetch + index papers
python -m src.rag "question"       # ask from the CLI
streamlit run app.py               # web UI on :8501
python -m src.evaluate_rerank --n 50 --k 10   # MRR: RRF-only vs RRF+rerank
```

## Layout

- `src/config.py` - all settings, env-overridable via `.env` (see `.env.example`)
- `src/ingest.py` - chunked ArXiv fetch (14-day windows, retry/backoff),
  cache in `data/papers.jsonl`, resume state in `data/ingest_state.json`,
  index mapping (`abstract_vector` is 384-dim cosine)
- `src/rag.py` - `search()` (BM25 + kNN → RRF → rerank), ArXiv-ID direct
  lookup, `ask()` entry point
- `src/evaluate_rerank.py` - self-retrieval MRR eval
- `app.py` - Streamlit front end over `ask()`
- `tests/` - pytest with mocks; `tests/fakes.py` has `FakeLLM`

## Conventions and gotchas

- **Import order in `src/ingest.py` and `src/rag.py` is deliberate**:
  `from src import config` comes before third-party imports so
  `HF_HUB_OFFLINE` is set before `huggingface_hub` loads. Ruff's `I001` is
  ignored for these files in `pyproject.toml` - don't reorder them.
- `ask()` and `search()` take injectable `es`/`embed_model`/`reranker`/`llm`
  so tests can pass fakes. Keep new dependencies injectable the same way.
- `search(rerank=False)` exists so the eval compares both stages on the
  same fusion code - don't duplicate fusion logic elsewhere.
- Changing `EMBEDDING_MODEL` means changing the `dims` in `INDEX_MAPPING`
  and re-indexing (the cached `data/papers.jsonl` avoids re-fetching).
- Direct deps in `requirements.txt` are pinned exactly; transitive deps are
  intentionally unpinned (torch pulls Linux-only CUDA packages).
- Ruff: line length 100, rules `E,F,I,UP,B`, target py311.
- `data/` and `.env` are gitignored.

## Status / tracking

Current state:
- [x] Ingestion with caching, chunked fetch, retry, resume
- [x] Hybrid BM25 + kNN retrieval fused with RRF (#1)
- [x] Cross-encoder re-ranking (#2) with MRR evaluation script
- [x] Titles searched alongside abstracts in BM25, kNN, and re-ranking (#3)
- [x] Streamlit web UI
- [x] CI: ruff + pytest on Python 3.11

Next up:
- [ ] Chunking for full-text papers (currently abstract-only)
- [ ] Pipeline ingest (#4): index each date chunk right after it's fetched, instead of
  fetching everything then re-indexing the whole cache (makes papers searchable
  sooner and stops re-embedding papers that are already indexed)
- [ ] Set `num_ctx` in `build_llm()`: Ollama's default context (2048/4096 by
  version) can silently truncate the start of the prompt (the first excerpts)

Keep this section updated as work lands.
