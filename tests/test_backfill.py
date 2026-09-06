"""驗證回填的分段與續跑行為。全部離線。"""

import sys
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from backfill import quarter_starts  # noqa: E402
from paper_pool import PaperRecord, open_pool  # noqa: E402
from zotero_arxiv_daily.retriever.arxiv_retriever import build_window_query  # noqa: E402


class QuarterSplitTests(unittest.TestCase):
    def test_a_full_year_becomes_four_quarters(self):
        spans = quarter_starts(date(2022, 1, 1), date(2022, 12, 31))
        self.assertEqual(len(spans), 4)
        self.assertEqual(spans[0], (date(2022, 1, 1), date(2022, 3, 31)))
        self.assertEqual(spans[-1], (date(2022, 10, 1), date(2022, 12, 31)))

    def test_spans_are_contiguous_and_never_overlap(self):
        spans = quarter_starts(date(2022, 1, 1), date(2026, 9, 6))
        for (_, end), (start, _) in zip(spans, spans[1:]):
            self.assertEqual(start.toordinal() - end.toordinal(), 1)

    def test_range_is_covered_exactly_end_to_end(self):
        start, end = date(2022, 2, 15), date(2026, 9, 6)
        spans = quarter_starts(start, end)
        self.assertEqual(spans[0][0], start)
        self.assertEqual(spans[-1][1], end)

    def test_partial_quarter_is_clipped_to_the_requested_range(self):
        spans = quarter_starts(date(2022, 2, 15), date(2022, 2, 20))
        self.assertEqual(spans, [(date(2022, 2, 15), date(2022, 2, 20))])

    def test_year_boundary_is_handled(self):
        spans = quarter_starts(date(2022, 11, 1), date(2023, 2, 1))
        self.assertEqual(spans[0], (date(2022, 11, 1), date(2022, 12, 31)))
        self.assertEqual(spans[1], (date(2023, 1, 1), date(2023, 2, 1)))

    def test_single_day_gives_one_span(self):
        self.assertEqual(
            quarter_starts(date(2022, 5, 5), date(2022, 5, 5)),
            [(date(2022, 5, 5), date(2022, 5, 5))],
        )

    def test_reversed_range_gives_nothing(self):
        self.assertEqual(quarter_starts(date(2023, 1, 1), date(2022, 1, 1)), [])


class WindowQueryTests(unittest.TestCase):
    def test_window_uses_the_explicit_bounds(self):
        query = build_window_query(
            ["physics.geo-ph"],
            datetime(2022, 1, 1, tzinfo=timezone.utc),
            datetime(2022, 3, 31, tzinfo=timezone.utc),
        )
        self.assertIn("submittedDate:[202201010000 TO 202203310000]", query)
        self.assertIn("cat:physics.geo-ph", query)

    def test_dates_work_as_well_as_datetimes(self):
        query = build_window_query(["eess.SP"], date(2022, 1, 1), date(2022, 3, 31))
        self.assertIn("submittedDate:[202201010000 TO 202203310000]", query)

    def test_reversed_window_is_rejected(self):
        with self.assertRaises(ValueError):
            build_window_query(["eess.SP"], date(2023, 1, 1), date(2022, 1, 1))

    def test_empty_categories_are_rejected(self):
        with self.assertRaises(ValueError):
            build_window_query([], date(2022, 1, 1), date(2022, 3, 31))


class ResumeTests(unittest.TestCase):
    """回填中斷後重跑不得產生重複資料。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = str(Path(self.tmp.name) / "pool.sqlite")

    def records(self, ids):
        return [
            PaperRecord(
                id=i,
                source="arxiv",
                title=f"Paper {i}",
                abstract="abstract",
                url=f"https://arxiv.org/abs/{i}",
                published_date="2022-01-01",
            )
            for i in ids
        ]

    def test_rerunning_after_an_interruption_skips_existing_papers(self):
        with open_pool(self.path) as pool:
            first = pool.ingest(self.records(["a", "b"]))
        self.assertEqual(first.inserted, 2)

        # 中斷後重跑：同樣的前兩筆加上後續的兩筆。
        with open_pool(self.path) as pool:
            second = pool.ingest(self.records(["a", "b", "c", "d"]))
            self.assertEqual(pool.stats()["total"], 4)
        self.assertEqual(second.inserted, 2)
        self.assertEqual(second.skipped_id, 2)

    def test_state_survives_reopening_the_database(self):
        with open_pool(self.path) as pool:
            pool.ingest(self.records(["a"]))
            pool.mark_recommended(["a"])
        with open_pool(self.path) as pool:
            row = pool._conn.execute("SELECT recommend_count FROM papers WHERE id = 'a'").fetchone()
            self.assertEqual(row["recommend_count"], 1)


if __name__ == "__main__":
    unittest.main()
