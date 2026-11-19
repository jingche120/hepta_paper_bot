import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from heptabase_reading_history import (  # noqa: E402
    ReadingHistoryError,
    clean_card_text,
    dump_reading_history,
    load_reading_history,
    parse_created_at,
    parse_overview_card,
)

# 依真實 read_object 輸出構造：首行標頭、每行 `N\t` 行號、hepta 專用標籤。
CARD = """card "論文總覽：SC-GNN 地震烈度即時預測" [11111111-1111-4111-8111-111111111111] 20 lines
1\t# 論文總覽：SC-GNN 地震烈度即時預測
2\t
3\t來源論文：<hepta-mention type="pdfCard" id="e189cdca">Real-time Seismic Intensity Prediction using Self-supervised Contrastive GNN</hepta-mention>
4\t
5\t作者：Rafid Umayer Murshed。年份：未標示。
6\t
7\t## English Abstract
8\t
9\tThis paper studies regional earthquake early warning as a direct regression problem.
10\t
11\t## 中文摘要
12\t
13\t本研究將區域型地震預警表述為直接的烈度迴歸問題。
14\t
15\t## 關鍵字
16\t
17\tEarthquake Early Warning、Seismic Intensity Prediction、SC-GNN。
18\t
19\t## 方法概覽
20\t
21\t1. 將各測站的三分量波形編碼。"""


def parse(text=CARD, **kwargs):
    return parse_overview_card(text, **kwargs)


class CleanTests(unittest.TestCase):
    def test_object_header_and_line_numbers_are_removed(self):
        cleaned = clean_card_text(CARD)
        self.assertFalse(cleaned.startswith("card "))
        self.assertTrue(cleaned.startswith("# 論文總覽"))
        self.assertNotIn("\t", cleaned)

    def test_hepta_tags_are_removed_but_inner_text_is_kept(self):
        cleaned = clean_card_text(CARD)
        self.assertNotIn("hepta-mention", cleaned)
        self.assertIn("Real-time Seismic Intensity Prediction", cleaned)


class ParseOverviewCardTests(unittest.TestCase):
    def test_title_comes_from_source_paper_not_card_heading(self):
        paper, _ = parse()
        self.assertEqual(
            paper.title,
            "Real-time Seismic Intensity Prediction using Self-supervised Contrastive GNN",
        )

    def test_abstract_joins_english_and_chinese_sections(self):
        paper, used_fallback = parse()
        self.assertFalse(used_fallback)
        self.assertIn("regional earthquake early warning", paper.abstract)
        self.assertIn("烈度迴歸問題", paper.abstract)

    def test_abstract_excludes_other_sections(self):
        # 只有摘要進語料；方法、實驗等章節不該混入相似度比對。
        paper, _ = parse()
        self.assertNotIn("三分量波形編碼", paper.abstract)

    def test_keywords_become_paths(self):
        paper, _ = parse()
        self.assertEqual(
            paper.paths,
            ["Earthquake Early Warning", "Seismic Intensity Prediction", "SC-GNN"],
        )

    def test_english_only_card_still_parses(self):
        card = "# T\n\n來源論文：Only English\n\n## English Abstract\n\nEnglish body."
        paper, used_fallback = parse(card)
        self.assertFalse(used_fallback)
        self.assertEqual(paper.abstract, "English body.")

    def test_missing_abstract_sections_fall_back_to_whole_card(self):
        card = "# 論文總覽：X\n\n來源論文：Some Paper\n\n## 方法概覽\n\n只有方法沒有摘要。"
        paper, used_fallback = parse(card)
        self.assertTrue(used_fallback)
        self.assertIn("只有方法沒有摘要", paper.abstract)

    def test_url_like_source_paper_falls_back_to_card_heading(self):
        # 「來源論文」指到的 PDF 卡片有時以 arXiv API 查詢字串命名，那不是論文標題。
        card = (
            "# 論文總覽 PatchTST：延後混合\n\n"
            "來源論文：arXiv Query: search_query=&amp;amp;id_list=2211.14730&amp;amp;start=0\n\n"
            "## English Abstract\n\nBody."
        )
        paper, _ = parse(card)
        self.assertEqual(paper.title, "論文總覽 PatchTST：延後混合")

    def test_full_title_wins_over_source_paper(self):
        # 一級標題放短代號、論文全名寫在「完整標題：」，是卡片的標準寫法。
        card = (
            "# 論文總覽：PatchTST\n\n"
            "來源論文：PDF 卡片的名字\n\n"
            "完整標題：A Time Series Is Worth 64 Words\n\n"
            "## English Abstract\n\nBody."
        )
        paper, _ = parse(card)
        self.assertEqual(paper.title, "A Time Series Is Worth 64 Words")

    def test_full_title_drops_markdown_emphasis(self):
        card = (
            "# 論文總覽：PatchTST\n\n"
            "完整標題：*A Time Series Is Worth 64 Words*\n\n"
            "## English Abstract\n\nBody."
        )
        paper, _ = parse(card)
        self.assertEqual(paper.title, "A Time Series Is Worth 64 Words")

    def test_url_like_full_title_falls_back_to_source_paper(self):
        # 「完整標題：」貼成網址時，寧可退回來源論文，也不要拿網址當標題。
        card = (
            "# 論文總覽：PatchTST\n\n"
            "來源論文：Real Paper Name\n\n"
            "完整標題：https://arxiv.org/abs/2211.14730\n\n"
            "## English Abstract\n\nBody."
        )
        paper, _ = parse(card)
        self.assertEqual(paper.title, "Real Paper Name")

    def test_card_without_full_title_still_uses_source_paper(self):
        # 既有卡片沒有「完整標題：」那一行，退路必須維持原樣。
        card = "# 論文總覽：短代號\n\n來源論文：Real Paper Name\n\n## English Abstract\n\nBody."
        paper, _ = parse(card)
        self.assertEqual(paper.title, "Real Paper Name")

    def test_html_entities_are_unescaped(self):
        card = "# T\n\n來源論文：Attention &amp;amp; Beyond\n\n## English Abstract\n\nBody."
        paper, _ = parse(card)
        self.assertEqual(paper.title, "Attention & Beyond")

    def test_title_falls_back_to_listed_title_without_source_line(self):
        card = "## English Abstract\n\nBody."
        paper, _ = parse(card, fallback_title="清單上的標題")
        self.assertEqual(paper.title, "清單上的標題")

    def test_card_without_title_or_content_raises(self):
        with self.assertRaises(ReadingHistoryError):
            parse("", fallback_title="")

    def test_metadata_is_carried_through(self):
        paper, _ = parse(card_id="abc", created_at="2026-09-05T17:49:45.787Z", order=3)
        self.assertEqual(paper.card_id, "abc")
        self.assertEqual(paper.created_at, "2026-09-05T17:49:45.787Z")
        self.assertEqual(paper.order, 3)


class CreatedAtTests(unittest.TestCase):
    def test_iso_z_timestamp_is_parsed(self):
        parsed = parse_created_at("2026-09-05T17:49:45.787Z", 0)
        self.assertEqual(parsed.year, 2026)
        self.assertEqual(parsed.month, 9)

    def test_newer_card_sorts_after_older_one(self):
        older = parse_created_at("2026-01-01T00:00:00Z", 0)
        newer = parse_created_at("2026-09-05T00:00:00Z", 1)
        self.assertLess(older, newer)

    def test_missing_timestamp_falls_back_without_raising(self):
        self.assertIsNotNone(parse_created_at(None, 0))

    def test_malformed_timestamp_falls_back_without_raising(self):
        self.assertIsNotNone(parse_created_at("not-a-date", 0))


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "reading_history.json"

    def dump(self):
        paper, _ = parse(card_id="abc", created_at="2026-09-05T17:49:45.787Z")
        dump_reading_history([paper], self.path, "wb-id")

    def test_round_trip_preserves_fields(self):
        self.dump()
        history = load_reading_history(self.path)
        self.assertEqual(len(history), 1)
        self.assertIn("regional earthquake early warning", history[0].abstract)
        self.assertEqual(history[0].paths[0], "Earthquake Early Warning")

    def test_added_date_comes_from_card_created_at(self):
        self.dump()
        self.assertEqual(load_reading_history(self.path)[0].added_date.year, 2026)

    def test_cache_records_whiteboard_id(self):
        self.dump()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(payload["whiteboard_id"], "wb-id")

    def test_missing_cache_names_the_export_command(self):
        with self.assertRaises(ReadingHistoryError) as error:
            load_reading_history(self.path)
        self.assertIn("--export", str(error.exception))

    def test_corrupt_cache_raises_parse_error(self):
        self.path.write_text("not json", encoding="utf-8")
        with self.assertRaises(ReadingHistoryError):
            load_reading_history(self.path)


if __name__ == "__main__":
    unittest.main()
