"""Fetch recent ArXiv papers for the configured categories/date window,
embed their abstracts, and index them into Elasticsearch.

The ArXiv API becomes unreliable at deep pagination offsets (observed HTTP
500/503 errors starting around offset 10,000 in a single query, even after
the client's built-in per-page retries). To stay well clear of that, the
date window is split into small chunks (default 14 days, ~1500-4000 papers
each for cs.AI/cs.LG) and fetched separately.

Sustained fetching over many chunks can also trigger HTTP 429 (rate
limiting) from the API itself, independent of the pagination-depth issue.
Chunk-level retries back off with an increasing pause (30s, 60s, 90s) to
give an active rate limit time to clear before retrying.

Fetched papers are appended to data/papers.jsonl as they come in, and
completed date chunks are recorded in data/ingest_state.json. Re-running
this script skips chunks already completed, so an interrupted or partially
failed run can just be re-run to fill in the gaps - no need to re-fetch
everything. Pass --refresh to discard the cache/state and start over.
"""

import argparse
import datetime
import json
import pathlib
import time

from src import config  # sets HF_HUB_OFFLINE before sentence_transformers is imported

import arxiv
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
CACHE_FILE = DATA_DIR / "papers.jsonl"
STATE_FILE = DATA_DIR / "ingest_state.json"

CHUNK_DAYS = 14
CHUNK_RETRIES = 3
CHUNK_RETRY_BACKOFF_SECONDS = 30

INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "arxiv_id": {"type": "keyword"},
            "title": {"type": "text"},
            "abstract": {"type": "text"},
            "authors": {"type": "keyword"},
            "categories": {"type": "keyword"},
            "published": {"type": "date"},
            "pdf_url": {"type": "keyword"},
            "abstract_vector": {
                "type": "dense_vector",
                "dims": 384,
                "index": True,
                "similarity": "cosine",
            },
        }
    }
}


def date_chunks(months_back: int, chunk_days: int = CHUNK_DAYS) -> list[tuple[datetime.date, datetime.date]]:
    end = datetime.date.today()
    start = end - datetime.timedelta(days=months_back * 30)
    chunks = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + datetime.timedelta(days=chunk_days - 1), end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + datetime.timedelta(days=1)
    return chunks


def build_query(start: datetime.date, end: datetime.date) -> str:
    cat_clause = " OR ".join(f"cat:{c}" for c in config.ARXIV_CATEGORIES)
    date_clause = f"submittedDate:[{start:%Y%m%d}0000 TO {end:%Y%m%d}2359]"
    return f"({cat_clause}) AND {date_clause}"


def fetch_chunk(start: datetime.date, end: datetime.date):
    search = arxiv.Search(
        query=build_query(start, end),
        max_results=config.ARXIV_MAX_RESULTS,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=3)
    for result in client.results(search):
        yield {
            "arxiv_id": result.get_short_id(),
            "title": result.title,
            "abstract": result.summary,
            "authors": [a.name for a in result.authors],
            "categories": result.categories,
            "published": result.published.isoformat(),
            "pdf_url": result.pdf_url,
        }


def load_state() -> set[str]:
    if STATE_FILE.exists():
        return set(json.loads(STATE_FILE.read_text()).get("completed_chunks", []))
    return set()


def save_state(completed: set[str]):
    STATE_FILE.write_text(json.dumps({"completed_chunks": sorted(completed)}, indent=2))


def load_or_fetch_papers(refresh: bool = False) -> list[dict]:
    DATA_DIR.mkdir(exist_ok=True)
    if refresh:
        CACHE_FILE.unlink(missing_ok=True)
        STATE_FILE.unlink(missing_ok=True)

    completed = load_state()
    chunks = date_chunks(config.ARXIV_MONTHS_BACK)
    failed = []

    with open(CACHE_FILE, "a") as f:
        for start, end in chunks:
            key = f"{start:%Y-%m-%d}_{end:%Y-%m-%d}"
            if key in completed:
                continue

            print(f"Fetching {key}...")
            for attempt in range(1, CHUNK_RETRIES + 1):
                try:
                    count = 0
                    for paper in fetch_chunk(start, end):
                        f.write(json.dumps(paper) + "\n")
                        count += 1
                    f.flush()
                    print(f"  {key}: {count} papers")
                    completed.add(key)
                    save_state(completed)
                    time.sleep(5)  # brief courtesy pause between chunks
                    break
                except Exception as e:
                    print(f"  {key}: attempt {attempt}/{CHUNK_RETRIES} failed ({e})")
                    if attempt == CHUNK_RETRIES:
                        failed.append(key)
                    else:
                        pause = CHUNK_RETRY_BACKOFF_SECONDS * attempt
                        print(f"  {key}: backing off {pause}s before retry")
                        time.sleep(pause)

    if failed:
        print(
            f"\nWARNING: {len(failed)} date chunk(s) failed after {CHUNK_RETRIES} "
            f"attempts: {failed}\nRe-run `python -m src.ingest` to retry just these."
        )

    # Note: a chunk that fails mid-fetch (after writing some papers but before
    # completing) may leave duplicate lines in the cache on retry. Harmless -
    # indexing upserts by arxiv_id, so duplicates just overwrite themselves.
    with open(CACHE_FILE) as f:
        return [json.loads(line) for line in f]


def index_papers(papers: list[dict], batch_size: int = 64):
    es = Elasticsearch(config.ES_URL)
    if not es.indices.exists(index=config.ES_INDEX):
        es.indices.create(index=config.ES_INDEX, body=INDEX_MAPPING)

    model = SentenceTransformer(config.EMBEDDING_MODEL)

    total = 0
    for i in range(0, len(papers), batch_size):
        batch = papers[i : i + batch_size]
        _flush_batch(es, model, batch)
        total += len(batch)
        print(f"Indexed {total}/{len(papers)} papers...")

    print(f"Done. Indexed {total} papers into '{config.ES_INDEX}'.")


def _flush_batch(es: Elasticsearch, model: SentenceTransformer, batch: list[dict]):
    vectors = model.encode(
        [p["abstract"] for p in batch],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    actions = [
        {
            "_index": config.ES_INDEX,
            "_id": paper["arxiv_id"],
            "_source": {**paper, "abstract_vector": vector.tolist()},
        }
        for paper, vector in zip(batch, vectors)
    ]
    helpers.bulk(es, actions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Discard the local cache/state and re-fetch everything from the ArXiv API",
    )
    args = parser.parse_args()

    papers = load_or_fetch_papers(refresh=args.refresh)
    index_papers(papers)
