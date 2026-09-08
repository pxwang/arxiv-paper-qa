"""Retrieve relevant papers from Elasticsearch (hybrid BM25 + kNN) and
answer a question about them using a local LLM via LangChain/Ollama."""

import re
import sys

from src import config  # sets HF_HUB_OFFLINE before sentence_transformers is imported

from elasticsearch import Elasticsearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from sentence_transformers import SentenceTransformer

PROMPT = ChatPromptTemplate.from_template(
    """You are a research assistant. Answer the question using only the
paper excerpts below. Cite papers by their arxiv_id in brackets, e.g. [2401.12345].
If the excerpts don't contain enough information, say so.

Excerpts:
{context}

Question: {question}

Answer:"""
)

# Matches an ArXiv ID like "2401.12345" or "2401.12345v2", whether bare or
# embedded in an arxiv.org URL.
ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")


def extract_arxiv_id(text: str) -> str | None:
    match = ARXIV_ID_RE.search(text)
    return match.group(0) if match else None


def find_by_id(es: Elasticsearch, arxiv_id: str) -> dict | None:
    """Look up a specific paper by ArXiv ID, tolerant of a missing/different
    version suffix (the stored _id includes the version, e.g. '...v1')."""
    bare_id = ARXIV_ID_RE.match(arxiv_id).group(1)
    resp = es.search(
        index=config.ES_INDEX,
        size=1,
        query={"wildcard": {"arxiv_id": f"{bare_id}*"}},
    )
    hits = resp["hits"]["hits"]
    return hits[0]["_source"] if hits else None


def corpus_date_range(es: Elasticsearch) -> tuple[str, str, int]:
    resp = es.search(
        index=config.ES_INDEX,
        size=0,
        track_total_hits=True,
        aggs={
            "oldest": {"min": {"field": "published"}},
            "newest": {"max": {"field": "published"}},
        },
    )
    count = resp["hits"]["total"]["value"]
    oldest = resp["aggregations"]["oldest"]["value_as_string"]
    newest = resp["aggregations"]["newest"]["value_as_string"]
    return oldest, newest, count


def search(es: Elasticsearch, model: SentenceTransformer, question: str, k: int = 5):
    """Retrieve top-k via BM25 and top-k via kNN independently, then merge
    the two lists (BM25 first, then any kNN hits not already included).
    BM25 and kNN scores aren't on comparable scales, so rather than summing
    them into one ranking, each method's own top-k stands on its own and
    duplicates (by arxiv_id) are dropped."""
    vector = model.encode(question, normalize_embeddings=True).tolist()

    bm25_resp = es.search(
        index=config.ES_INDEX,
        size=k,
        query={"match": {"abstract": question}},
    )
    knn_resp = es.search(
        index=config.ES_INDEX,
        size=k,
        knn={
            "field": "abstract_vector",
            "query_vector": vector,
            "k": k,
            "num_candidates": 50,
        },
    )

    seen_ids = set()
    papers = []
    for hit in bm25_resp["hits"]["hits"] + knn_resp["hits"]["hits"]:
        paper = hit["_source"]
        if paper["arxiv_id"] in seen_ids:
            continue
        seen_ids.add(paper["arxiv_id"])
        papers.append(paper)
    return papers


def format_context(papers: list[dict]) -> str:
    return "\n\n".join(
        f"[{p['arxiv_id']}] {p['title']}\n"
        f"Authors: {', '.join(p['authors'])}\n"
        f"{p['abstract']}"
        for p in papers
    )


def ask(question: str) -> str:
    es = Elasticsearch(config.ES_URL)

    arxiv_id = extract_arxiv_id(question)
    if arxiv_id:
        paper = find_by_id(es, arxiv_id)
        if paper is None:
            oldest, newest, count = corpus_date_range(es)
            return (
                f"Paper {arxiv_id} is not in the indexed corpus.\n\n"
                f"The index currently holds {count} papers spanning "
                f"{oldest} to {newest}. You can view the paper directly at "
                f"https://arxiv.org/abs/{arxiv_id}."
            )
        papers = [paper]
    else:
        embed_model = SentenceTransformer(config.EMBEDDING_MODEL)
        papers = search(es, embed_model, question)

    if not papers:
        return "No indexed papers matched this question. Have you run `python -m src.ingest` yet?"

    llm = ChatOllama(model=config.OLLAMA_MODEL, base_url=config.OLLAMA_BASE_URL)
    chain = PROMPT | llm

    result = chain.invoke({"context": format_context(papers), "question": question})
    sources = "\n".join(f"- [{p['arxiv_id']}] {p['title']} ({p['pdf_url']})" for p in papers)
    return f"{result.content}\n\nSources:\n{sources}"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python -m src.rag "your question here"')
        sys.exit(1)
    print(ask(" ".join(sys.argv[1:])))
