"""Compare RRF-fusion-only retrieval against RRF + cross-encoder re-ranking
using Mean Reciprocal Rank (MRR), via self-retrieval: each sampled paper's
own title is used as the query, and we check what rank the paper's own
arxiv_id comes back at.

This is a proxy, not a real relevance eval - it measures "can retrieval find
the exact paper whose title matches the query," not "are the results
relevant to an open-ended question." Read a large MRR gap as a real signal,
but confirm on real questions (e.g. app.py's EXAMPLES) before trusting it.

Usage:
    python -m src.evaluate_rerank [--n 30] [--k 10]
"""

import argparse
import sys

from src import config
from src.rag import build_embed_model, build_es_client, build_reranker, search


def sample_papers(es, n: int) -> list[dict]:
    resp = es.search(
        index=config.ES_INDEX,
        size=n,
        query={"function_score": {"query": {"match_all": {}}, "random_score": {}}},
    )
    return [hit["_source"] for hit in resp["hits"]["hits"]]


def reciprocal_ranks(es, model, reranker, papers: list[dict], k: int, rerank: bool) -> list[float]:
    ranks = []
    for paper in papers:
        results = search(es, model, reranker, paper["title"], k=k, rerank=rerank)
        rank = next(
            (i for i, p in enumerate(results, start=1) if p["arxiv_id"] == paper["arxiv_id"]),
            None,
        )
        ranks.append(1 / rank if rank else 0.0)
    return ranks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n", type=int, default=30, help="number of papers to sample as self-retrieval queries"
    )
    parser.add_argument("--k", type=int, default=10, help="how many results to consider per query")
    args = parser.parse_args()

    es = build_es_client()
    model = build_embed_model()
    reranker = build_reranker()

    papers = sample_papers(es, args.n)
    if not papers:
        print("No papers in the index. Run `python -m src.ingest` first.")
        sys.exit(1)

    print(f"Evaluating on {len(papers)} papers (self-retrieval by title), k={args.k}\n")

    before = reciprocal_ranks(es, model, reranker, papers, args.k, rerank=False)
    after = reciprocal_ranks(es, model, reranker, papers, args.k, rerank=True)

    print(f"{'Query (paper title)':<70} {'RRF only':>10} {'+ Rerank':>10}")
    for paper, rr_before, rr_after in zip(papers, before, after, strict=True):
        title = paper["title"]
        title = title if len(title) <= 67 else title[:67] + "..."
        print(f"{title:<70} {rr_before:>10.3f} {rr_after:>10.3f}")

    mrr_before = sum(before) / len(before)
    mrr_after = sum(after) / len(after)
    print(f"\nMRR@{args.k} (RRF only):      {mrr_before:.3f}")
    print(f"MRR@{args.k} (RRF + rerank): {mrr_after:.3f}")


if __name__ == "__main__":
    main()
