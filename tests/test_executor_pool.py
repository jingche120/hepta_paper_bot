"""驗證 Executor 與候選池的接線。全部離線：不連網、不呼叫 API。"""

import sys
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from paper_pool import PaperRecord, open_pool  # noqa: E402
from zotero_arxiv_daily.executor import Executor  # noqa: E402
from zotero_arxiv_daily.protocol import CorpusPaper, Paper  # noqa: E402

DIM = 4


class FakeEmbedder:
    """向量只看摘要的第一個字元，方便在測試裡預測排名。"""

    model = "fake-model"
    batch_size = 8

    def embed(self, texts):
        vectors = []
        for text in texts:
            code = ord(text[0]) if text else 0
            vectors.append([1.0, float(code % 7), 0.0, 0.0])
        return vectors


def config(db_path, **pool_overrides):
    pool = {
        "db_path": str(db_path),
        "n_new": 1,
        "n_old": 1,
        "new_within_days": 7,
        "journal_lookback_days": 14,
        "openalex_sources": [],
        "openalex_mailto": "",
    }
    pool.update(pool_overrides)
    return OmegaConf.create(
        {
            "heptabase": {
                "reading_history_path": "unused",
                "include_path": None,
                "ignore_path": None,
            },
            "pool": pool,
            "executor": {
                "reranker": "api",
                "send_empty": False,
                "dry_run": False,
                "source": ["arxiv"],
                "debug": False,
            },
            "llm": {"api": {"key": "k", "base_url": None}},
        }
    )


def corpus_paper(title, abstract, days_ago):
    return CorpusPaper(
        title=title,
        abstract=abstract,
        added_date=datetime.now(timezone.utc) - timedelta(days=days_ago),
        paths=[],
    )


class ExecutorPoolTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / "pool.sqlite"
        self.config = config(self.db)
        self.executor = Executor.__new__(Executor)
        self.executor.config = self.config
        self.executor.retrievers = {}
        self.executor.openai_client = None
        self.executor.include_path_patterns = None
        self.executor.ignore_path_patterns = None
        # 閱讀紀錄從檔案讀出來是「舊到新」，排序才有事可做。
        self.history = [corpus_paper("Oldest", "bbb", 100), corpus_paper("Newest", "aaa", 0)]

    def seed(self, specs):
        with open_pool(self.db) as pool:
            pool.ingest(
                PaperRecord(
                    id=i,
                    source="arxiv",
                    title=t,
                    abstract=a,
                    url=f"https://arxiv.org/abs/{i}",
                    published_date=d,
                )
                for i, t, a, d in specs
            )

    def run_executor(self, sent):
        with (
            patch.object(Executor, "fetch_reading_history", return_value=self.history),
            patch("zotero_arxiv_daily.executor.get_embedder", return_value=FakeEmbedder()),
            patch("zotero_arxiv_daily.executor.send_email", side_effect=sent),
            patch("zotero_arxiv_daily.executor.render_email", return_value="<html/>"),
            patch.object(Executor, "fetch_full_texts"),
            patch.object(Paper, "generate_tldr"),
            patch.object(Paper, "generate_affiliations"),
        ):
            self.executor.run()


class SelectionTests(ExecutorPoolTestCase):
    def test_quota_pulls_from_both_new_and_backlog(self):
        today = date.today().isoformat()
        self.seed([("n1", "New", "aaa", today), ("o1", "Old", "aab", "2022-01-01")])
        sent = []
        self.run_executor(lambda *a: sent.append(a))
        with open_pool(self.db) as pool:
            recommended = {
                r["id"]
                for r in pool._conn.execute("SELECT id FROM papers WHERE recommend_count > 0")
            }
        self.assertEqual(recommended, {"n1", "o1"})
        self.assertEqual(len(sent), 1)

    def test_recommendations_are_recorded_only_after_the_email_is_sent(self):
        # 寄信失敗還計次，會讓這些論文白白被推到隊伍後方。
        self.seed([("o1", "Old", "aaa", "2022-01-01")])

        def boom(*_args):
            raise RuntimeError("SMTP down")

        with self.assertRaises(RuntimeError):
            self.run_executor(boom)
        with open_pool(self.db) as pool:
            count = pool._conn.execute(
                "SELECT recommend_count FROM papers WHERE id = 'o1'"
            ).fetchone()
            self.assertEqual(count["recommend_count"], 0)

    def test_dry_run_sends_nothing_and_records_nothing(self):
        self.config.executor.dry_run = True
        self.seed([("o1", "Old", "aaa", "2022-01-01")])
        sent = []
        self.run_executor(lambda *a: sent.append(a))
        self.assertEqual(sent, [])
        with open_pool(self.db) as pool:
            self.assertEqual(pool.stats()["recommended"], 0)

    def test_carded_papers_are_excluded_from_recommendations(self):
        self.seed([("o1", "Newest", "aaa", "2022-01-01"), ("o2", "Other", "aab", "2022-01-01")])
        sent = []
        self.run_executor(lambda *a: sent.append(a))
        with open_pool(self.db) as pool:
            carded = pool._conn.execute("SELECT carded FROM papers WHERE id='o1'").fetchone()
            self.assertEqual(carded["carded"], 1)
            recommended = [
                r["id"]
                for r in pool._conn.execute("SELECT id FROM papers WHERE recommend_count > 0")
            ]
        self.assertNotIn("o1", recommended)

    def test_empty_pool_sends_nothing(self):
        sent = []
        self.run_executor(lambda *a: sent.append(a))
        self.assertEqual(sent, [])

    def test_papers_persist_for_the_next_run(self):
        # v1 的候選用完即丟；池子讓沒選上的留到下次。
        self.seed([("o%d" % i, f"P{i}", chr(97 + i) * 3, "2022-01-01") for i in range(4)])
        sent = []
        self.run_executor(lambda *a: sent.append(a))
        with open_pool(self.db) as pool:
            self.assertEqual(pool.stats()["total"], 4)
            self.assertEqual(pool.stats()["recommended"], 1)


class HistoryOrderTests(ExecutorPoolTestCase):
    def test_history_is_embedded_newest_first(self):
        # time_decay_weights 假設新到舊；順序反了會讓最舊的卡片主導推薦。
        self.seed([("o1", "Old", "aaa", "2022-01-01")])
        seen = []

        class RecordingEmbedder(FakeEmbedder):
            def embed(self, texts):
                seen.append(list(texts))
                return super().embed(texts)

        with (
            patch.object(Executor, "fetch_reading_history", return_value=self.history),
            patch("zotero_arxiv_daily.executor.get_embedder", return_value=RecordingEmbedder()),
            patch("zotero_arxiv_daily.executor.send_email"),
            patch("zotero_arxiv_daily.executor.render_email", return_value="<html/>"),
            patch.object(Executor, "fetch_full_texts"),
            patch.object(Paper, "generate_tldr"),
            patch.object(Paper, "generate_affiliations"),
        ):
            self.executor.run()
        history_batch = seen[-1]
        # 傳進來是 [Oldest, Newest]，送去 embed 前必須翻成 [Newest, Oldest]。
        self.assertEqual(history_batch, ["aaa", "bbb"])


class IngestTests(ExecutorPoolTestCase):
    def test_daily_papers_land_in_the_pool_with_their_published_date(self):
        published = date.today().isoformat()

        class FakeRetriever:
            def retrieve_papers(self):
                return [
                    Paper(
                        source="arxiv",
                        title="T",
                        authors=[],
                        abstract="aaa",
                        url="https://arxiv.org/abs/2601.001",
                    )
                ]

            def published_date(self, paper):
                return published

        self.executor.retrievers = {"arxiv": FakeRetriever()}
        with open_pool(self.db) as pool:
            self.executor.ingest_daily_sources(pool)
            row = pool._conn.execute("SELECT id, published_date FROM papers").fetchone()
        self.assertEqual(row["id"], "arxiv:2601.001")
        self.assertEqual(row["published_date"], published)

    def test_journal_sources_are_skipped_when_not_configured(self):
        self.executor.config.pool.openalex_sources = []
        with open_pool(self.db) as pool:
            self.executor.ingest_daily_sources(pool)
            self.assertEqual(pool.stats()["total"], 0)


class FullTextTests(ExecutorPoolTestCase):
    def test_sources_without_a_retriever_keep_no_full_text(self):
        # IEEE 期刊有付費牆，沒有對應的 retriever；不該因此崩潰。
        paper = Paper(
            source="openalex", title="T", authors=[], abstract="a", url="https://doi.org/10.1109/x"
        )
        self.executor.retrievers = {}
        with patch("zotero_arxiv_daily.executor.sleep"):
            self.executor.fetch_full_texts([paper])
        self.assertIsNone(paper.full_text)

    def test_failure_on_one_paper_does_not_stop_the_rest(self):
        class Flaky:
            def fetch_full_text(self, paper):
                if paper.title == "bad":
                    raise RuntimeError("network")
                return "ok"

        papers = [
            Paper(source="arxiv", title="bad", authors=[], abstract="a", url="u1"),
            Paper(source="arxiv", title="good", authors=[], abstract="a", url="u2"),
        ]
        self.executor.retrievers = {"arxiv": Flaky()}
        with patch("zotero_arxiv_daily.executor.sleep"):
            self.executor.fetch_full_texts(papers)
        self.assertIsNone(papers[0].full_text)
        self.assertEqual(papers[1].full_text, "ok")


class PoolIdTests(unittest.TestCase):
    def test_arxiv_url_becomes_a_stable_id(self):
        paper = Paper(
            source="arxiv",
            title="T",
            authors=[],
            abstract="a",
            url="https://arxiv.org/abs/2211.14730v1",
        )
        self.assertEqual(Executor._pool_id(paper), "arxiv:2211.14730v1")

    def test_pooled_paper_round_trips_back_to_a_paper(self):
        pooled = SimpleNamespace(
            id="arxiv:1",
            title="T",
            abstract="a",
            url="u",
            pdf_url=None,
            published_date="2022-01-01",
            score=1.5,
            recommend_count=0,
        )
        paper = Executor._to_paper(pooled)
        self.assertEqual(paper.source, "arxiv")
        self.assertEqual(paper.score, 1.5)


if __name__ == "__main__":
    unittest.main()
