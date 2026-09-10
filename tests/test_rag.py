from unittest.mock import MagicMock

from src.rag import ask, corpus_date_range, extract_arxiv_id, find_by_id, format_context, search
from tests.fakes import FakeLLM


class TestExtractArxivId:
    def test_bare_id(self):
        assert extract_arxiv_id("what about 2401.12345?") == "2401.12345"

    def test_id_with_version(self):
        assert extract_arxiv_id("see 2401.12345v2 for details") == "2401.12345v2"

    def test_id_inside_url(self):
        assert extract_arxiv_id("https://arxiv.org/abs/2401.12345") == "2401.12345"

    def test_no_id_present(self):
        assert extract_arxiv_id("what are transformers?") is None

    def test_too_few_digits_after_dot_does_not_match(self):
        assert extract_arxiv_id("see 2401.123 for details") is None


class TestFindById:
    def test_found(self):
        es = MagicMock()
        es.search.return_value = {
            "hits": {"hits": [{"_source": {"arxiv_id": "2401.12345v2", "title": "T"}}]}
        }

        result = find_by_id(es, "2401.12345")

        assert result == {"arxiv_id": "2401.12345v2", "title": "T"}
        assert es.search.call_args.kwargs["query"] == {"wildcard": {"arxiv_id": "2401.12345*"}}

    def test_not_found(self):
        es = MagicMock()
        es.search.return_value = {"hits": {"hits": []}}

        assert find_by_id(es, "9999.99999") is None

    def test_strips_version_suffix_before_querying(self):
        es = MagicMock()
        es.search.return_value = {"hits": {"hits": []}}

        find_by_id(es, "2401.12345v3")

        assert es.search.call_args.kwargs["query"] == {"wildcard": {"arxiv_id": "2401.12345*"}}


class TestCorpusDateRange:
    def test_normal_index(self):
        es = MagicMock()
        es.search.return_value = {
            "hits": {"total": {"value": 125112}},
            "aggregations": {
                "oldest": {"value_as_string": "2025-03-13T00:00:00.000Z"},
                "newest": {"value_as_string": "2026-09-03T00:00:00.000Z"},
            },
        }

        oldest, newest, count = corpus_date_range(es)

        assert count == 125112
        assert oldest == "2025-03-13T00:00:00.000Z"
        assert newest == "2026-09-03T00:00:00.000Z"
        assert es.search.call_args.kwargs["track_total_hits"] is True

    def test_empty_index_does_not_crash(self):
        """Regression test: min/max aggs return no value on an empty index,
        which used to raise a TypeError downstream when sliced with [:10]."""
        es = MagicMock()
        es.search.return_value = {
            "hits": {"total": {"value": 0}},
            "aggregations": {"oldest": {"value": None}, "newest": {"value": None}},
        }

        oldest, newest, count = corpus_date_range(es)

        assert count == 0
        assert oldest == "n/a"
        assert newest == "n/a"


class TestSearch:
    def test_fuses_bm25_and_knn_by_reciprocal_rank(self):
        """Paper B ranks 2nd in BM25 and 1st in kNN, so its RRF score
        (1/62 + 1/61) beats paper A's BM25-only score (1/61) even though A
        was ranked above B in BM25 alone. Paper C (kNN rank 2 only, 1/62)
        scores lowest. The reranker here just preserves the RRF order (equal
        descending scores), isolating this test to the fusion step."""
        paper_a = {"arxiv_id": "1", "title": "A", "abstract": "a"}
        paper_b = {"arxiv_id": "2", "title": "B", "abstract": "b"}
        paper_c = {"arxiv_id": "3", "title": "C", "abstract": "c"}

        es = MagicMock()
        es.search.side_effect = [
            {"hits": {"hits": [{"_source": paper_a}, {"_source": paper_b}]}},  # BM25: A, B
            {"hits": {"hits": [{"_source": paper_b}, {"_source": paper_c}]}},  # kNN: B, C
        ]

        model = MagicMock()
        model.encode.return_value.tolist.return_value = [0.1, 0.2]
        reranker = MagicMock()
        reranker.predict.return_value = [0.9, 0.6, 0.3]  # preserves fused order: B, A, C

        papers = search(es, model, reranker, "some question", k=2)

        assert [p["arxiv_id"] for p in papers] == ["2", "1"]
        assert es.search.call_count == 2
        model.encode.assert_called_once_with("some question", normalize_embeddings=True)

    def test_reranker_score_determines_final_order(self):
        """RRF fuses to the order B, A, C (see the test above), but the
        cross-encoder scores A highest and B lowest here - proving the final
        result follows the reranker's judgment, not the RRF fusion order."""
        paper_a = {"arxiv_id": "1", "title": "A", "abstract": "a"}
        paper_b = {"arxiv_id": "2", "title": "B", "abstract": "b"}
        paper_c = {"arxiv_id": "3", "title": "C", "abstract": "c"}

        es = MagicMock()
        es.search.side_effect = [
            {"hits": {"hits": [{"_source": paper_a}, {"_source": paper_b}]}},  # BM25: A, B
            {"hits": {"hits": [{"_source": paper_b}, {"_source": paper_c}]}},  # kNN: B, C
        ]

        model = MagicMock()
        model.encode.return_value.tolist.return_value = [0.1, 0.2]
        reranker = MagicMock()
        reranker.predict.return_value = [0.2, 0.9, 0.5]  # scores for fused order B, A, C

        papers = search(es, model, reranker, "some question", k=3)

        assert [p["arxiv_id"] for p in papers] == ["1", "3", "2"]
        pairs = reranker.predict.call_args.args[0]
        assert pairs == [("some question", "b"), ("some question", "a"), ("some question", "c")]

    def test_no_candidates_skips_reranker(self):
        es = MagicMock()
        es.search.return_value = {"hits": {"hits": []}}
        model = MagicMock()
        model.encode.return_value.tolist.return_value = [0.0]
        reranker = MagicMock()

        papers = search(es, model, reranker, "some question")

        assert papers == []
        reranker.predict.assert_not_called()

    def test_rerank_false_returns_rrf_order_without_calling_reranker(self):
        """Used by src/evaluate_rerank.py to compare RRF-only against
        RRF+rerank through the same fusion code path."""
        paper_a = {"arxiv_id": "1", "title": "A", "abstract": "a"}
        paper_b = {"arxiv_id": "2", "title": "B", "abstract": "b"}
        paper_c = {"arxiv_id": "3", "title": "C", "abstract": "c"}

        es = MagicMock()
        es.search.side_effect = [
            {"hits": {"hits": [{"_source": paper_a}, {"_source": paper_b}]}},  # BM25: A, B
            {"hits": {"hits": [{"_source": paper_b}, {"_source": paper_c}]}},  # kNN: B, C
        ]
        model = MagicMock()
        model.encode.return_value.tolist.return_value = [0.1, 0.2]
        reranker = MagicMock()

        papers = search(es, model, reranker, "some question", k=2, rerank=False)

        assert [p["arxiv_id"] for p in papers] == ["2", "1"]  # RRF fusion order
        reranker.predict.assert_not_called()


class TestFormatContext:
    def test_includes_id_title_authors_abstract(self):
        papers = [
            {
                "arxiv_id": "2401.12345",
                "title": "Test Paper",
                "authors": ["Alice", "Bob"],
                "abstract": "This is the abstract.",
            }
        ]

        result = format_context(papers)

        assert "[2401.12345] Test Paper" in result
        assert "Authors: Alice, Bob" in result
        assert "This is the abstract." in result

    def test_multiple_papers_both_present_in_order(self):
        papers = [
            {"arxiv_id": "1", "title": "A", "authors": ["X"], "abstract": "a"},
            {"arxiv_id": "2", "title": "B", "authors": ["Y"], "abstract": "b"},
        ]

        result = format_context(papers)

        assert result.index("[1] A") < result.index("[2] B")


class TestAsk:
    def _paper(self, arxiv_id="1", **overrides):
        paper = {
            "arxiv_id": arxiv_id,
            "title": f"Paper {arxiv_id}",
            "authors": ["Alice"],
            "abstract": "Abstract text.",
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
        }
        paper.update(overrides)
        return paper

    def test_direct_id_found(self):
        paper = self._paper("2401.12345v2")
        es = MagicMock()
        es.search.return_value = {"hits": {"hits": [{"_source": paper}]}}
        llm = FakeLLM("This paper is about X. [2401.12345v2]")

        answer = ask("tell me about 2401.12345", es=es, llm=llm)

        assert "This paper is about X." in answer
        assert "[2401.12345v2] Paper 2401.12345v2" in answer
        assert "https://arxiv.org/pdf/2401.12345v2" in answer
        es.search.assert_called_once()  # only find_by_id, no separate search
        assert len(llm.invocations) == 1

    def test_direct_id_not_found_reports_corpus_range(self):
        es = MagicMock()
        es.search.side_effect = [
            {"hits": {"hits": []}},  # find_by_id: nothing
            {
                "hits": {"total": {"value": 100}},
                "aggregations": {
                    "oldest": {"value_as_string": "2025-01-01"},
                    "newest": {"value_as_string": "2026-01-01"},
                },
            },
        ]
        llm = FakeLLM()

        answer = ask("what about 9999.99999", es=es, llm=llm)

        assert "9999.99999 is not in the indexed corpus" in answer
        assert "100 papers" in answer
        assert "arxiv.org/abs/9999.99999" in answer
        assert llm.invocations == []  # never reached the LLM

    def test_topic_search_no_results(self):
        es = MagicMock()
        es.search.return_value = {"hits": {"hits": []}}
        model = MagicMock()
        model.encode.return_value.tolist.return_value = [0.0]
        reranker = MagicMock()
        llm = FakeLLM()

        answer = ask("an obscure topic", es=es, embed_model=model, reranker=reranker, llm=llm)

        assert "No indexed papers matched" in answer
        assert llm.invocations == []
        reranker.predict.assert_not_called()

    def test_topic_search_with_results(self):
        paper1 = self._paper("1")
        paper2 = self._paper("2")
        es = MagicMock()
        es.search.side_effect = [
            {"hits": {"hits": [{"_source": paper1}]}},  # BM25
            {"hits": {"hits": [{"_source": paper2}]}},  # kNN
        ]
        model = MagicMock()
        model.encode.return_value.tolist.return_value = [0.0]
        reranker = MagicMock()
        reranker.predict.return_value = [0.9, 0.8]
        llm = FakeLLM("Summary of both papers.")

        answer = ask("some topic question", es=es, embed_model=model, reranker=reranker, llm=llm)

        assert "Summary of both papers." in answer
        assert "[1] Paper 1" in answer
        assert "[2] Paper 2" in answer
