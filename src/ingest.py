"""Fetch recent ArXiv papers for the configured categories/date window,
embed their abstracts, and index them into Elasticsearch."""

import datetime

import arxiv
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer

from src import config

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


def index_papers(batch_size: int = 64):
    es = Elasticsearch(config.ES_URL)
    if not es.indices.exists(index=config.ES_INDEX):
        es.indices.create(index=config.ES_INDEX, body=INDEX_MAPPING)

    model = SentenceTransformer(config.EMBEDDING_MODEL)

    batch = []
    total = 0
    for paper in fetch_papers():
        batch.append(paper)
        if len(batch) >= batch_size:
            _flush_batch(es, model, batch)
            total += len(batch)
            print(f"Indexed {total} papers...")
            batch = []
    if batch:
        _flush_batch(es, model, batch)
        total += len(batch)

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
    index_papers()
