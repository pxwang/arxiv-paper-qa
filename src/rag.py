"""Retrieve relevant papers from Elasticsearch (hybrid BM25 + kNN) and
answer a question about them using a local LLM via LangChain/Ollama."""

import sys

from elasticsearch import Elasticsearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from sentence_transformers import SentenceTransformer

from src import config

PROMPT = ChatPromptTemplate.from_template(
    """You are a research assistant. Answer the question using only the
paper excerpts below. Cite papers by their arxiv_id in brackets, e.g. [2401.12345].
If the excerpts don't contain enough information, say so.

Excerpts:
{context}

Question: {question}

Answer:"""
)


def search(es: Elasticsearch, model: SentenceTransformer, question: str, k: int = 5):
    vector = model.encode(question, normalize_embeddings=True).tolist()
    resp = es.search(
        index=config.ES_INDEX,
        size=k,
        query={"match": {"abstract": question}},
        knn={
            "field": "abstract_vector",
            "query_vector": vector,
            "k": k,
            "num_candidates": 50,
        },
    )
    return [hit["_source"] for hit in resp["hits"]["hits"]]


def format_context(papers: list[dict]) -> str:
    return "\n\n".join(
        f"[{p['arxiv_id']}] {p['title']}\n{p['abstract']}" for p in papers
    )


def ask(question: str) -> str:
    es = Elasticsearch(config.ES_URL)
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
