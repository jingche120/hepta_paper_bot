import json
import sys
import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from heptabase_reading_history import ReadingHistoryError  # noqa: E402
from zotero_arxiv_daily.executor import Executor, normalize_path_patterns  # noqa: E402
from zotero_arxiv_daily.protocol import CorpusPaper  # noqa: E402


def cache(path, papers):
    path.write_text(
        json.dumps(
            {
                "whiteboard_id": "wb",
                "fetched_at": "2026-01-01T00:00:00+00:00",
                "papers": papers,
            }
        ),
        encoding="utf-8",
    )


def executor_with(reading_history_path, include_path=None, ignore_path=None):
    """建立只填好 heptabase 區塊的 Executor，跳過檢索器與 LLM 用戶端的初始化。"""
    executor = Executor.__new__(Executor)
    executor.config = OmegaConf.create(
        {
            "heptabase": {
                "reading_history_path": str(reading_history_path),
                "include_path": include_path,
                "ignore_path": ignore_path,
            }
        }
    )
    executor.include_path_patterns = normalize_path_patterns(include_path, "include_path")
    executor.ignore_path_patterns = normalize_path_patterns(ignore_path, "ignore_path")
    return executor


class FetchReadingHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "reading_history.json"

    def test_fetch_returns_reading_history(self):
        cache(
            self.path,
            [
                {
                    "title": "舊的",
                    "abstract": "A",
                    "order": 0,
                    "paths": [],
                    "created_at": "2026-01-01T00:00:00Z",
                },
                {
                    "title": "新的",
                    "abstract": "B",
                    "order": 1,
                    "paths": [],
                    "created_at": "2026-09-01T00:00:00Z",
                },
            ],
        )
        history = executor_with(self.path).fetch_reading_history()
        self.assertTrue(all(isinstance(c, CorpusPaper) for c in history))
        self.assertEqual([c.title for c in history], ["舊的", "新的"])

    def test_later_created_card_is_newer_so_reranker_weights_it_higher(self):
        cache(
            self.path,
            [
                {
                    "title": "先讀",
                    "abstract": "A",
                    "order": 0,
                    "paths": [],
                    "created_at": "2026-01-01T00:00:00Z",
                },
                {
                    "title": "後讀",
                    "abstract": "B",
                    "order": 1,
                    "paths": [],
                    "created_at": "2026-09-01T00:00:00Z",
                },
            ],
        )
        history = executor_with(self.path).fetch_reading_history()
        newest = max(history, key=lambda c: c.added_date)
        self.assertEqual(newest.title, "後讀")

    def test_missing_cache_tells_user_to_export(self):
        with self.assertRaises(ReadingHistoryError) as error:
            executor_with(self.path).fetch_reading_history()
        self.assertIn("--export", str(error.exception))


class FilterReadingHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "reading_history.json"
        cache(
            self.path,
            [
                {
                    "title": "甲",
                    "abstract": "A",
                    "order": 0,
                    "paths": ["LLM/RAG"],
                    "created_at": "2026-01-01T00:00:00Z",
                },
                {
                    "title": "乙",
                    "abstract": "B",
                    "order": 1,
                    "paths": ["archive/old"],
                    "created_at": "2026-02-01T00:00:00Z",
                },
                {
                    "title": "丙",
                    "abstract": "C",
                    "order": 2,
                    "paths": [],
                    "created_at": "2026-03-01T00:00:00Z",
                },
            ],
        )

    def filtered(self, **kwargs):
        executor = executor_with(self.path, **kwargs)
        return [c.title for c in executor.filter_reading_history(executor.fetch_reading_history())]

    def test_no_patterns_keeps_everything(self):
        self.assertEqual(self.filtered(), ["甲", "乙", "丙"])

    def test_include_path_selects_matching_paths(self):
        self.assertEqual(self.filtered(include_path=["LLM/**"]), ["甲"])

    def test_ignore_path_excludes_matching_paths(self):
        self.assertEqual(self.filtered(ignore_path=["archive/**"]), ["甲", "丙"])

    def test_papers_without_paths_are_dropped_by_include_but_kept_by_ignore(self):
        self.assertNotIn("丙", self.filtered(include_path=["LLM/**"]))
        self.assertIn("丙", self.filtered(ignore_path=["archive/**"]))

    def test_string_pattern_is_rejected(self):
        with self.assertRaises(TypeError) as error:
            normalize_path_patterns("LLM/**", "include_path")
        self.assertIn("config.heptabase.include_path", str(error.exception))


if __name__ == "__main__":
    unittest.main()
