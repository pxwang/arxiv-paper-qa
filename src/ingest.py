"""Fetch recent ArXiv papers for the configured categories/date window,
embed their abstracts, and index them into Elasticsearch.

Fetched papers are cached to data/papers.jsonl. Re-running this script
reuses that cache instead of re-hitting the ArXiv API (which is
rate-limited and can take well over an hour for a large date window) -
handy when re-indexing after a mapping or embedding-model change. Pass
--refresh to force a fresh fetch.
"""

import argparse
import datetime
import json
import pathlib

import arxiv
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer

from src import config

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
CACHE_FILE = DATA_DIR / "papers.jsonl"

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


def build_query() -> str:
    end = datetime.date.today()
    start = end - datetime.timedelta(days=config.ARXIV_MONTHS_BACK * 30)
    cat_clause = " OR ".join(f"cat:{c}" for c in config.ARXIV_CATEGORIES)
    date_clause = f"submittedDate:[{start:%Y%m%d}0000 TO {end:%Y%m%d}2359]"
    return f"({cat_clause}) AND {date_clause}"


def fetch_papers():
    search = arxiv.Search(
        query=build_query(),
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


def load_or_fetch_papers(refresh: bool = False) -> list[dict]:
    if CACHE_FILE.exists() and not refresh:
        print(f"Loading cached papers from {CACHE_FILE}")
        with open(CACHE_FILE) as f:
            return [json.loads(line) for line in f]

    print("Fetching papers from the ArXiv API (rate-limited, this may take a while)...")
    DATA_DIR.mkdir(exist_ok=True)

    # Write to a temp file and only rename to the final cache path once the
    # fetch completes fully, so an interrupted fetch never gets mistaken for
    # a complete cache on the next run.
    tmp_file = CACHE_FILE.with_suffix(".jsonl.tmp")
    papers = []
    with open(tmp_file, "w") as f:
        for paper in fetch_papers():
            f.write(json.dumps(paper) + "\n")
            papers.append(paper)
            if len(papers) % 500 == 0:
                print(f"Fetched {len(papers)} papers so far...")
    tmp_file.rename(CACHE_FILE)

    print(f"Fetched {len(papers)} papers, cached to {CACHE_FILE}")
    return papers


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
        help="Re-fetch from the ArXiv API even if a local cache exists",
    )
    args = parser.parse_args()

    papers = load_or_fetch_papers(refresh=args.refresh)
    index_papers(papers)
