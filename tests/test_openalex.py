"""驗證 OpenAlex 客戶端。全部離線：以假的 _get_json 取代 HTTP。"""

import sys
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openalex import (  # noqa: E402
    OpenAlexClient,
    OpenAlexError,
    abstract_from_inverted_index,
    publication_window_start,
    work_to_record,
)


def work(**overrides):
    base = {
        "id": "https://openalex.org/W7135093076",
        "doi": "https://doi.org/10.1109/LGRS.2026.3673607",
        "display_name": "Tri-State Prototype Self-Distillation for SAR Ocean Imagery",
        "publication_date": "2026-01-01",
        "abstract_inverted_index": {"We": [0], "propose": [1], "a": [2], "method": [3]},
        "primary_location": {
            "landing_page_url": "https://doi.org/10.1109/lgrs.2026.3673607",
            "pdf_url": None,
        },
    }
    base.update(overrides)
    return base


class FakeClient(OpenAlexClient):
    """把 HTTP 換成預先準備好的頁面。"""

    def __init__(self, pages, **kwargs):
        super().__init__(page_size=2, **kwargs)
        self.pages = list(pages)
        self.calls = []

    def _get_json(self, params):
        self.calls.append(params)
        return self.pages.pop(0)


def page(results, next_cursor=None):
    return {"results": results, "meta": {"next_cursor": next_cursor}}


class InvertedIndexTests(unittest.TestCase):
    def test_words_are_restored_in_position_order(self):
        inverted = {"world": [1], "hello": [0], "again": [2]}
        self.assertEqual(abstract_from_inverted_index(inverted), "hello world again")

    def test_repeated_words_appear_at_every_position(self):
        inverted = {"the": [0, 2], "cat": [1], "sat": [3]}
        self.assertEqual(abstract_from_inverted_index(inverted), "the cat the sat")

    def test_missing_or_empty_index_gives_none(self):
        self.assertIsNone(abstract_from_inverted_index(None))
        self.assertIsNone(abstract_from_inverted_index({}))


class WorkToRecordTests(unittest.TestCase):
    def test_doi_becomes_the_lowercase_identifier(self):
        # 同一篇論文在不同來源要得到相同的 id，因此 DOI 統一轉小寫。
        self.assertEqual(work_to_record(work()).id, "doi:10.1109/lgrs.2026.3673607")

    def test_falls_back_to_openalex_id_without_doi(self):
        self.assertEqual(work_to_record(work(doi=None)).id, "openalex:W7135093076")

    def test_fields_are_carried_over(self):
        record = work_to_record(work())
        self.assertEqual(record.source, "openalex")
        self.assertEqual(record.abstract, "We propose a method")
        self.assertEqual(record.published_date, "2026-01-01")
        self.assertEqual(record.url, "https://doi.org/10.1109/lgrs.2026.3673607")
        # IEEE 期刊有付費牆，沒有可取得的 PDF。
        self.assertIsNone(record.pdf_url)

    def test_records_without_an_abstract_are_skipped(self):
        # 沒有摘要就無法參與相似度比對，留在池中只會佔位。
        self.assertIsNone(work_to_record(work(abstract_inverted_index=None)))

    def test_records_without_a_title_are_skipped(self):
        self.assertIsNone(work_to_record(work(display_name="   ")))

    def test_records_without_any_identifier_are_skipped(self):
        self.assertIsNone(work_to_record(work(doi=None, id="")))

    def test_missing_primary_location_does_not_crash(self):
        record = work_to_record(work(primary_location=None))
        self.assertEqual(record.url, "https://openalex.org/W7135093076")


class FetchWorksTests(unittest.TestCase):
    def collect(self, pages, **kwargs):
        client = FakeClient(pages, **kwargs)
        return list(client.fetch_works("S1", "2022-01-01")), client

    def test_walks_every_page_until_the_cursor_runs_out(self):
        records, client = self.collect(
            [
                page(
                    [work(doi="https://doi.org/10.1/a"), work(doi="https://doi.org/10.1/b")], "c2"
                ),
                page([work(doi="https://doi.org/10.1/c")], None),
            ]
        )
        self.assertEqual([r.id for r in records], ["doi:10.1/a", "doi:10.1/b", "doi:10.1/c"])
        self.assertEqual([c["cursor"] for c in client.calls], ["*", "c2"])

    def test_stops_on_an_empty_page(self):
        records, client = self.collect([page([], "c2")])
        self.assertEqual(records, [])
        self.assertEqual(len(client.calls), 1)

    def test_repeated_cursor_raises_instead_of_looping_forever(self):
        client = FakeClient([page([work()], "*"), page([work()], "*")])
        with self.assertRaises(OpenAlexError):
            list(client.fetch_works("S1", "2022-01-01"))

    def test_unusable_records_are_skipped_but_the_page_continues(self):
        records, _ = self.collect(
            [page([work(abstract_inverted_index=None), work(doi="https://doi.org/10.1/b")], None)]
        )
        self.assertEqual([r.id for r in records], ["doi:10.1/b"])

    def test_filter_includes_source_and_date_range(self):
        client = FakeClient([page([], None)])
        list(client.fetch_works("S111326731", "2022-01-01", until_date="2026-09-06"))
        filters = client.calls[0]["filter"]
        self.assertIn("primary_location.source.id:S111326731", filters)
        self.assertIn("from_publication_date:2022-01-01", filters)
        self.assertIn("to_publication_date:2026-09-06", filters)

    def test_until_date_is_omitted_when_not_given(self):
        client = FakeClient([page([], None)])
        list(client.fetch_works("S1", "2022-01-01"))
        self.assertNotIn("to_publication_date", client.calls[0]["filter"])

    def test_only_publication_date_is_used(self):
        # created / updated 日期過濾需要付費方案（HTTP 429 + "Plan upgrade required"），
        # 所以不該送出去，免得每天靜默失敗。
        client = FakeClient([page([], None)])
        list(client.fetch_works("S1", "2026-01-01"))
        filters = client.calls[0]["filter"]
        self.assertIn("from_publication_date:2026-01-01", filters)
        self.assertNotIn("created_date", filters)
        self.assertNotIn("updated_date", filters)

    def test_abstract_field_is_requested(self):
        client = FakeClient([page([], None)])
        list(client.fetch_works("S1", "2022-01-01"))
        self.assertIn("abstract_inverted_index", client.calls[0]["select"])


class PolitePoolTests(unittest.TestCase):
    def test_mailto_goes_into_the_user_agent(self):
        client = OpenAlexClient(mailto="someone@example.com")
        self.assertIn("mailto:someone@example.com", client.headers["User-Agent"])

    def test_user_agent_is_still_set_without_mailto(self):
        self.assertTrue(OpenAlexClient().headers["User-Agent"])


class PublicationWindowTests(unittest.TestCase):
    """期號日期的粒度是「年」，往回推幾天會直接把整年排除。"""

    def test_the_window_snaps_back_to_the_start_of_the_year(self):
        # 2026-09-06 往回 30 天是 2026-08-07；2026 年的論文全部標成 2026-01-01，
        # 直接拿 2026-08-07 當起點會回傳 0 筆而且不報錯。
        self.assertEqual(publication_window_start(date(2026, 9, 6), 30), "2026-01-01")

    def test_the_new_year_window_still_covers_the_previous_year(self):
        # 1 月剛跨年時，前一年最後才進 OpenAlex 的論文不能有空窗。
        self.assertEqual(publication_window_start(date(2027, 1, 10), 30), "2026-01-01")

    def test_a_zero_lookback_stays_in_the_current_year(self):
        self.assertEqual(publication_window_start(date(2026, 9, 6), 0), "2026-01-01")

    def test_a_negative_lookback_is_rejected(self):
        with self.assertRaises(ValueError):
            publication_window_start(date(2026, 9, 6), -1)


if __name__ == "__main__":
    unittest.main()
