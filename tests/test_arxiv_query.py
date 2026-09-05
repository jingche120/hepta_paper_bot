import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from zotero_arxiv_daily.retriever.arxiv_retriever import (  # noqa: E402
    ArxivRetriever,
    build_date_range_query,
)

NOW = datetime(2026, 9, 6, 12, 0, tzinfo=timezone.utc)


def result(entry_id, primary):
    return SimpleNamespace(entry_id=entry_id, primary_category=primary)


def retriever(**arxiv_overrides):
    config = OmegaConf.create(
        {
            "source": {
                "arxiv": {
                    "category": ["physics.geo-ph", "eess.SP"],
                    "include_cross_list": False,
                    "lookback_days": 7,
                    "max_results": 2000,
                    **arxiv_overrides,
                }
            },
            "executor": {"debug": False},
        }
    )
    return ArxivRetriever(config)


class BuildQueryTests(unittest.TestCase):
    def test_window_spans_the_requested_days(self):
        query = build_date_range_query(["physics.geo-ph"], 7, now=NOW)
        self.assertIn("submittedDate:[202608301200 TO 202609061200]", query)

    def test_multiple_categories_are_combined_with_or(self):
        query = build_date_range_query(["physics.geo-ph", "eess.SP"], 7, now=NOW)
        self.assertIn("(cat:physics.geo-ph OR cat:eess.SP)", query)

    def test_one_day_window(self):
        query = build_date_range_query(["eess.SP"], 1, now=NOW)
        self.assertIn("submittedDate:[202609051200 TO 202609061200]", query)

    def test_empty_categories_rejected(self):
        with self.assertRaises(ValueError):
            build_date_range_query([], 7, now=NOW)

    def test_zero_days_rejected(self):
        with self.assertRaises(ValueError):
            build_date_range_query(["eess.SP"], 0, now=NOW)


class RetrievalModeTests(unittest.TestCase):
    """lookback_days 決定走 API 日期區間還是上游的當日 RSS。"""

    def dispatch(self, **overrides):
        r = retriever(**overrides)
        with (
            patch.object(ArxivRetriever, "_retrieve_by_date_range", return_value="range") as rng,
            patch.object(ArxivRetriever, "_retrieve_from_rss", return_value="rss") as rss,
        ):
            return r._retrieve_raw_papers(), rng, rss

    def test_lookback_days_uses_the_api_date_range(self):
        mode, rng, _ = self.dispatch(lookback_days=7)
        self.assertEqual(mode, "range")
        rng.assert_called_once_with(7)

    def test_missing_lookback_days_keeps_upstream_rss_behaviour(self):
        self.assertEqual(self.dispatch(lookback_days=None)[0], "rss")

    def test_one_day_keeps_upstream_rss_behaviour(self):
        self.assertEqual(self.dispatch(lookback_days=1)[0], "rss")


class DateRangeFilterTests(unittest.TestCase):
    def collect(self, results, **overrides):
        r = retriever(**overrides)
        with patch("arxiv.Client") as client:
            client.return_value.results.return_value = iter(results)
            return r._retrieve_by_date_range(7)

    def test_cross_lists_are_excluded_by_primary_category(self):
        # API 沒有 announce_type，只能靠 primary_category 排除 cross-list。
        got = self.collect([result("a", "physics.geo-ph"), result("b", "cs.LG")])
        self.assertEqual([p.entry_id for p in got], ["a"])

    def test_cross_lists_are_kept_when_enabled(self):
        got = self.collect(
            [result("a", "physics.geo-ph"), result("b", "cs.LG")], include_cross_list=True
        )
        self.assertEqual([p.entry_id for p in got], ["a", "b"])

    def test_duplicate_entry_ids_are_dropped(self):
        got = self.collect([result("a", "eess.SP"), result("a", "eess.SP")])
        self.assertEqual(len(got), 1)


if __name__ == "__main__":
    unittest.main()
