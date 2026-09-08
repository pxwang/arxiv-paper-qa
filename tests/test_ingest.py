import datetime
from itertools import pairwise
from unittest.mock import MagicMock

import numpy as np

from src import ingest


class TestDateChunks:
    def test_splits_into_expected_chunk_count(self):
        chunks = ingest.date_chunks(months_back=1, chunk_days=14)
        assert len(chunks) == 3  # ceil(30 days / 14-day chunks)

    def test_chunks_are_contiguous_and_end_today(self):
        chunks = ingest.date_chunks(months_back=1, chunk_days=14)
        today = datetime.date.today()

        assert chunks[-1][1] == today
        for (_, end), (next_start, _) in pairwise(chunks):
            assert end + datetime.timedelta(days=1) == next_start

    def test_zero_months_back_gives_single_one_day_chunk(self):
        chunks = ingest.date_chunks(months_back=0, chunk_days=14)
        today = datetime.date.today()

        assert chunks == [(today, today)]


class TestBuildQuery:
    def test_format(self, monkeypatch):
        monkeypatch.setattr(ingest.config, "ARXIV_CATEGORIES", ["cs.AI", "cs.LG"])
        start = datetime.date(2025, 3, 13)
        end = datetime.date(2025, 3, 26)

        query = ingest.build_query(start, end)

        assert query == "(cat:cs.AI OR cat:cs.LG) AND submittedDate:[202503130000 TO 202503262359]"


class TestStateFile:
    def test_load_state_missing_file_returns_empty_set(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ingest, "STATE_FILE", tmp_path / "state.json")
        assert ingest.load_state() == set()

    def test_save_then_load_roundtrips(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ingest, "STATE_FILE", tmp_path / "state.json")

        ingest.save_state({"2025-01-01_2025-01-14", "2025-01-15_2025-01-28"})

        assert ingest.load_state() == {"2025-01-01_2025-01-14", "2025-01-15_2025-01-28"}


class TestLoadOrFetchPapers:
    def _patch_data_paths(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "data"
        monkeypatch.setattr(ingest, "DATA_DIR", data_dir)
        monkeypatch.setattr(ingest, "CACHE_FILE", data_dir / "papers.jsonl")
        monkeypatch.setattr(ingest, "STATE_FILE", data_dir / "ingest_state.json")
        monkeypatch.setattr(ingest.config, "ARXIV_MONTHS_BACK", 0)  # single 1-day chunk
        monkeypatch.setattr(ingest.time, "sleep", lambda s: None)

    def test_fetches_and_caches_then_skips_completed_chunk_on_rerun(self, tmp_path, monkeypatch):
        self._patch_data_paths(tmp_path, monkeypatch)
        calls = {"n": 0}

        def fake_fetch_chunk(start, end):
            calls["n"] += 1
            yield {
                "arxiv_id": "1",
                "title": "T",
                "abstract": "a",
                "authors": [],
                "categories": [],
                "published": "2025-01-01",
                "pdf_url": "u",
            }

        monkeypatch.setattr(ingest, "fetch_chunk", fake_fetch_chunk)

        first = ingest.load_or_fetch_papers()
        assert len(first) == 1
        assert calls["n"] == 1

        second = ingest.load_or_fetch_papers()
        assert len(second) == 1  # still just the cached paper
        assert calls["n"] == 1  # fetch_chunk not called again - chunk already completed

    def test_retries_then_gives_up_without_marking_chunk_completed(
        self, tmp_path, monkeypatch, capsys
    ):
        self._patch_data_paths(tmp_path, monkeypatch)
        monkeypatch.setattr(ingest, "CHUNK_RETRIES", 2)
        attempts = {"n": 0}

        def always_fail(start, end):
            attempts["n"] += 1
            raise RuntimeError("simulated API failure")
            yield  # pragma: no cover - makes this a generator function

        monkeypatch.setattr(ingest, "fetch_chunk", always_fail)

        papers = ingest.load_or_fetch_papers()

        assert papers == []
        assert attempts["n"] == 2
        assert ingest.load_state() == set()
        assert "failed after 2 attempts" in capsys.readouterr().out


class TestFlushBatch:
    def test_embeds_abstracts_only_and_builds_bulk_actions(self, monkeypatch):
        es = MagicMock()
        model = MagicMock()
        model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])

        batch = [
            {"arxiv_id": "1", "abstract": "abs1", "title": "T1"},
            {"arxiv_id": "2", "abstract": "abs2", "title": "T2"},
        ]

        captured = {}

        def fake_bulk(client, actions):
            captured["actions"] = list(actions)

        monkeypatch.setattr(ingest.helpers, "bulk", fake_bulk)

        ingest._flush_batch(es, model, batch)

        model.encode.assert_called_once_with(
            ["abs1", "abs2"], normalize_embeddings=True, show_progress_bar=False
        )
        actions = captured["actions"]
        assert [a["_id"] for a in actions] == ["1", "2"]
        assert actions[0]["_source"]["abstract_vector"] == [0.1, 0.2]
        assert actions[1]["_source"]["abstract_vector"] == [0.3, 0.4]
        assert actions[0]["_source"]["title"] == "T1"
