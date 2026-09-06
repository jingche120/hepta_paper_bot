"""驗證候選池的 13 條不變式。

全部離線：不連網、不呼叫任何 API，向量以固定亂數種子產生。
"""

import sqlite3
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paper_pool import (  # noqa: E402
    PaperPool,
    PaperPoolError,
    PaperRecord,
    normalize_title,
    open_pool,
    time_decay_weights,
    weighted_centroid,
)

DIM = 8
# 卡片上的真實標題，連字號是 U+2010 HYPHEN 而非 ASCII。
EDGEPHASE_CARD = "EdgePhase: A Deep Learning Model for Multi‐Station Seismic Phase Picking"
EDGEPHASE_ARXIV = "EdgePhase: A Deep Learning Model for Multi-Station Seismic Phase Picking"


def unit(seed, dim=DIM):
    rng = np.random.default_rng(seed)
    vector = rng.standard_normal(dim).astype(np.float32)
    return vector / np.linalg.norm(vector)


def raw(seed, scale, dim=DIM):
    """未正規化的向量：長度不是 1。"""
    return unit(seed, dim) * scale


def record(paper_id, title="Some Paper", published="2020-01-01", source="arxiv", abstract="a"):
    return PaperRecord(
        id=paper_id,
        source=source,
        title=title,
        abstract=abstract,
        url=f"https://example.org/{paper_id}",
        pdf_url=None,
        published_date=published,
    )


class PoolTestCase(unittest.TestCase):
    def setUp(self):
        self.pool = open_pool(":memory:")
        self.addCleanup(self.pool.close)

    def fill(self, specs, model="m1"):
        """specs: (id, title, published, seed)。seed 為 None 表示不給向量。"""
        self.pool.ingest([record(i, t, p) for i, t, p, _ in specs])
        self.pool.store_embeddings([(i, unit(s)) for i, _, _, s in specs if s is not None], model)


# 不變式 1
class NormalizeTitleTests(unittest.TestCase):
    def test_unicode_hyphen_matches_ascii_hyphen(self):
        self.assertEqual(normalize_title(EDGEPHASE_CARD), normalize_title(EDGEPHASE_ARXIV))

    def test_all_unicode_dash_variants_fold_together(self):
        base = normalize_title("Multi-Station")
        for dash in "‐‑‒–—―−－":
            with self.subTest(dash=hex(ord(dash))):
                self.assertEqual(normalize_title(f"Multi{dash}Station"), base)

    def test_case_punctuation_and_spacing_are_ignored(self):
        self.assertEqual(normalize_title("Multi Station"), normalize_title("multi-station"))
        self.assertEqual(normalize_title("A Study: Part I"), normalize_title("a study, part i"))

    def test_different_titles_stay_different(self):
        self.assertNotEqual(normalize_title("EdgePhase"), normalize_title("EQTransformer"))

    def test_cjk_is_preserved(self):
        self.assertEqual(
            normalize_title("論文總覽：PatchTST"), normalize_title("論文總覽 PatchTST")
        )
        self.assertTrue(normalize_title("論文總覽：PatchTST"))

    def test_empty_title_gives_empty_string(self):
        self.assertEqual(normalize_title(""), "")
        self.assertEqual(normalize_title(None), "")


# 不變式 2 與 3
class IngestDedupTests(PoolTestCase):
    def test_same_id_is_not_inserted_twice(self):
        first = self.pool.ingest([record("arxiv:1")])
        second = self.pool.ingest([record("arxiv:1")])
        self.assertEqual((first.inserted, second.inserted), (1, 0))
        self.assertEqual(second.skipped_id, 1)
        self.assertEqual(self.pool.stats()["total"], 1)

    def test_reingest_does_not_reset_accumulated_state(self):
        self.fill([("arxiv:1", "Paper One", "2026-09-01", 1)])
        self.pool.mark_carded(["Paper One"])
        self.pool.mark_recommended(["arxiv:1"])

        self.pool.ingest([record("arxiv:1", "Paper One")])

        row = self.pool._conn.execute(
            "SELECT recommend_count, carded, embedding FROM papers WHERE id = 'arxiv:1'"
        ).fetchone()
        self.assertEqual(row["recommend_count"], 1)
        self.assertEqual(row["carded"], 1)
        self.assertIsNotNone(row["embedding"])

    def test_same_normalized_title_under_different_ids_is_one_paper(self):
        # arXiv 預印本與 IEEE 正式版：id 不同、標題相同（且連字號寫法不同）。
        result = self.pool.ingest(
            [
                record("arxiv:2110.1", EDGEPHASE_ARXIV),
                record("doi:10.1109/TGRS.1", EDGEPHASE_CARD, source="openalex"),
            ]
        )
        self.assertEqual(result.inserted, 1)
        self.assertEqual(result.skipped_title, 1)
        self.assertEqual(self.pool.stats()["total"], 1)

    def test_genuinely_different_papers_both_land(self):
        result = self.pool.ingest([record("a", "EdgePhase"), record("b", "EQTransformer")])
        self.assertEqual(result.inserted, 2)


# 不變式 4
class PendingEmbeddingTests(PoolTestCase):
    def test_only_missing_or_stale_vectors_are_pending(self):
        self.fill([("a", "A", "2026-09-01", 1), ("b", "B", "2026-09-01", None)])
        self.assertEqual([i for i, _ in self.pool.pending_embeddings("m1")], ["b"])

    def test_changing_model_makes_everything_pending_again(self):
        self.fill([("a", "A", "2026-09-01", 1), ("b", "B", "2026-09-01", 2)])
        self.assertEqual(self.pool.pending_embeddings("m1"), [])
        self.assertEqual(len(self.pool.pending_embeddings("m2")), 2)

    def test_pending_returns_the_abstract_to_embed(self):
        self.pool.ingest([record("a", "A", abstract="the abstract text")])
        self.assertEqual(self.pool.pending_embeddings("m1"), [("a", "the abstract text")])


# 不變式 5
class MarkCardedTests(PoolTestCase):
    def test_unmatched_card_titles_are_reported(self):
        self.fill([("a", EDGEPHASE_ARXIV, "2026-09-01", 1)])
        unmatched = self.pool.mark_carded([EDGEPHASE_CARD, "論文總覽：PatchTST"])
        self.assertEqual(unmatched, ["論文總覽：PatchTST"])

    def test_matching_is_unicode_insensitive(self):
        # 卡片用 U+2010、池中是 ASCII，仍須比對成功。
        self.fill([("a", EDGEPHASE_ARXIV, "2026-09-01", 1)])
        self.assertEqual(self.pool.mark_carded([EDGEPHASE_CARD]), [])

    def test_blank_card_title_is_reported_not_crashed(self):
        self.assertEqual(self.pool.mark_carded([""]), [""])

    def test_accepts_objects_with_a_title_attribute(self):
        class Card:
            title = EDGEPHASE_CARD

        self.fill([("a", EDGEPHASE_ARXIV, "2026-09-01", 1)])
        self.assertEqual(self.pool.mark_carded([Card()]), [])


# 不變式 6 到 12
class SelectTests(PoolTestCase):
    def setUp(self):
        super().setUp()
        self.today = date(2026, 9, 6)
        self.centroid = unit(100)

    def select(self, n_new=1, n_old=1, days=7):
        return self.pool.select(self.centroid, n_new, n_old, days, today=self.today)

    def test_carded_papers_never_appear(self):
        self.fill([("a", "A", "2026-09-05", 1), ("b", "B", "2026-09-05", 2)])
        self.pool.mark_carded(["A"])
        self.assertEqual([p.id for p in self.select(n_new=5, n_old=5)], ["b"])

    def test_papers_without_embedding_never_appear(self):
        self.fill([("a", "A", "2026-09-05", 1), ("b", "B", "2026-09-05", None)])
        self.assertEqual([p.id for p in self.select(n_new=5, n_old=5)], ["a"])

    def test_quota_splits_new_and_backlog(self):
        self.fill(
            [
                ("n1", "N1", "2026-09-05", 1),
                ("n2", "N2", "2026-09-04", 2),
                ("o1", "O1", "2022-01-01", 3),
                ("o2", "O2", "2022-02-01", 4),
            ]
        )
        picked = self.select(n_new=2, n_old=1)
        self.assertEqual(len(picked), 3)
        self.assertEqual(sum(p.published_date >= "2026-08-30" for p in picked), 2)
        self.assertEqual(sum(p.published_date < "2026-08-30" for p in picked), 1)

    def test_quota_larger_than_supply_returns_what_exists(self):
        self.fill([("n1", "N1", "2026-09-05", 1), ("o1", "O1", "2022-01-01", 2)])
        self.assertEqual(len(self.select(n_new=5, n_old=5)), 2)

    def test_empty_pool_returns_empty_list(self):
        self.assertEqual(self.select(), [])

    def test_zero_quota_is_honoured(self):
        self.fill([("n1", "N1", "2026-09-05", 1), ("o1", "O1", "2022-01-01", 2)])
        self.assertEqual([p.id for p in self.select(n_new=0, n_old=1)], ["o1"])

    def test_negative_quota_is_rejected(self):
        with self.assertRaises(ValueError):
            self.select(n_new=-1)

    def test_cutoff_boundary_counts_as_new(self):
        # today - 7 天 = 2026-08-30，剛好等於 cutoff 者算「新」。
        self.fill([("edge", "Edge", "2026-08-30", 1), ("old", "Old", "2026-08-29", 2)])
        self.assertEqual([p.id for p in self.select(n_new=1, n_old=0)], ["edge"])
        self.assertEqual([p.id for p in self.select(n_new=0, n_old=1)], ["old"])

    def test_paper_without_published_date_counts_as_backlog(self):
        self.pool.ingest([record("x", "X", published=None)])
        self.pool.store_embeddings([("x", unit(1))], "m1")
        self.assertEqual([p.id for p in self.select(n_new=5, n_old=0)], [])
        self.assertEqual([p.id for p in self.select(n_new=0, n_old=5)], ["x"])

    def test_higher_score_wins_at_equal_recommend_count(self):
        self.fill([("a", "A", "2022-01-01", 1), ("b", "B", "2022-01-01", 2)])
        ordered = self.pool.select(self.centroid, 0, 2, 7, today=self.today)
        self.assertGreaterEqual(ordered[0].score, ordered[1].score)

    def test_recommend_count_outranks_score(self):
        self.fill([("a", "A", "2022-01-01", 1), ("b", "B", "2022-01-01", 2)])
        best = self.pool.select(self.centroid, 0, 1, 7, today=self.today)[0].id
        self.pool.mark_recommended([best])
        # 分數較高但已推薦過的，要讓位給還沒推薦過的。
        self.assertNotEqual(self.pool.select(self.centroid, 0, 1, 7, today=self.today)[0].id, best)

    def test_mismatched_centroid_dimension_is_rejected(self):
        self.fill([("a", "A", "2022-01-01", 1)])
        with self.assertRaises(PaperPoolError):
            self.pool.select(np.zeros(DIM + 1, dtype=np.float32), 1, 1, 7, today=self.today)


# 不變式 10 與 11
class RotationTests(PoolTestCase):
    def setUp(self):
        super().setUp()
        self.today = date(2026, 9, 6)
        self.centroid = unit(100)
        self.fill([(f"o{i}", f"O{i}", "2022-01-01", i) for i in range(1, 5)])

    def test_recommend_count_increments_by_exactly_one(self):
        self.pool.mark_recommended(["o1"])
        self.pool.mark_recommended(["o1"])
        row = self.pool._conn.execute(
            "SELECT recommend_count, last_recommended_at FROM papers WHERE id = 'o1'"
        ).fetchone()
        self.assertEqual(row["recommend_count"], 2)
        self.assertIsNotNone(row["last_recommended_at"])

    def test_consecutive_selections_rotate(self):
        first = self.pool.select(self.centroid, 0, 2, 7, today=self.today)
        self.pool.mark_recommended(first)
        second = self.pool.select(self.centroid, 0, 2, 7, today=self.today)
        self.assertNotEqual({p.id for p in first}, {p.id for p in second})

    def test_nothing_is_dropped_permanently(self):
        # 輪過一圈後，最早推薦過的論文會重新回到隊伍前端。
        seen = set()
        for _ in range(2):
            picked = self.pool.select(self.centroid, 0, 2, 7, today=self.today)
            self.pool.mark_recommended(picked)
            seen.update(p.id for p in picked)
        self.assertEqual(seen, {"o1", "o2", "o3", "o4"})
        again = self.pool.select(self.centroid, 0, 2, 7, today=self.today)
        self.assertTrue({p.id for p in again} <= seen)


# 不變式 13
class ScoreEquivalenceTests(PoolTestCase):
    def test_centroid_score_equals_weighted_mean_of_similarities(self):
        history = np.stack([unit(s) for s in (11, 12, 13, 14, 15)])
        weights = time_decay_weights(len(history))
        centroid = weighted_centroid(history)

        candidate = unit(99)
        by_centroid = float(candidate @ centroid) * 10
        by_weighted_mean = float(((candidate @ history.T) * weights).sum()) * 10
        self.assertAlmostEqual(by_centroid, by_weighted_mean, places=5)

    def test_pool_scores_match_the_reranker_formula(self):
        history = np.stack([unit(s) for s in (11, 12, 13)])
        centroid = weighted_centroid(history)
        vectors = {"a": unit(1), "b": unit(2)}
        self.pool.ingest([record(i, i.upper(), "2022-01-01") for i in vectors])
        self.pool.store_embeddings(vectors.items(), "m1")

        picked = self.pool.select(centroid, 0, 2, 7, today=date(2026, 9, 6))
        for paper in picked:
            expected = (
                float(((vectors[paper.id] @ history.T) * time_decay_weights(len(history))).sum())
                * 10
            )
            self.assertAlmostEqual(paper.score, expected, places=5)

    def test_time_decay_weights_sum_to_one_and_favour_recent(self):
        weights = time_decay_weights(5)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
        self.assertTrue(all(weights[i] > weights[i + 1] for i in range(4)))

    def test_weighted_centroid_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            weighted_centroid(np.zeros((0, DIM), dtype=np.float32))
        with self.assertRaises(ValueError):
            weighted_centroid(np.zeros((3, DIM), dtype=np.float32), weights=np.ones(2))


# Codex 審查後補：向量正規化（不變式 13 的前提）
class NormalizationTests(PoolTestCase):
    def test_unnormalized_vectors_are_normalized_on_write(self):
        # [2,0,...]·[2,0,...]*10 = 40，但餘弦分數應為 10。
        self.pool.ingest([record("a", "A", "2022-01-01")])
        self.pool.store_embeddings([("a", raw(1, 2.0))], "m1")
        centroid = weighted_centroid(np.stack([raw(1, 3.0)]))
        picked = self.pool.select(centroid, 0, 1, 7, today=date(2026, 9, 6))
        self.assertAlmostEqual(picked[0].score, 10.0, places=4)

    def test_scale_does_not_change_ranking_or_score(self):
        self.pool.ingest([record(i, i.upper(), "2022-01-01") for i in ("a", "b")])
        self.pool.store_embeddings([("a", unit(1)), ("b", raw(2, 7.0))], "m1")
        history = np.stack([unit(11), unit(12)])
        scaled = self.pool.select(weighted_centroid(history * 5), 0, 2, 7, today=date(2026, 9, 6))
        plain = self.pool.select(weighted_centroid(history), 0, 2, 7, today=date(2026, 9, 6))
        self.assertEqual([p.id for p in scaled], [p.id for p in plain])
        for a, b in zip(scaled, plain):
            self.assertAlmostEqual(a.score, b.score, places=4)

    def test_zero_vector_is_rejected(self):
        self.pool.ingest([record("a", "A")])
        with self.assertRaises(ValueError):
            self.pool.store_embeddings([("a", np.zeros(DIM, dtype=np.float32))], "m1")


# Codex 審查後補：去重由資料庫保證，而非應用層的先查再寫
class AtomicDedupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = str(Path(self.tmp.name) / "pool.sqlite")

    def open(self):
        pool = PaperPool(sqlite3.connect(self.path))
        self.addCleanup(pool.close)
        return pool

    def test_separate_connections_cannot_both_insert_the_same_title(self):
        # 回填跑到一半時每日任務觸發，兩個連線同時寫入同一篇論文。
        first, second = self.open(), self.open()
        first.ingest([record("arxiv:1", EDGEPHASE_ARXIV)])
        result = second.ingest([record("doi:10.1109/x", EDGEPHASE_CARD, source="openalex")])
        self.assertEqual(result.inserted, 0)
        self.assertEqual(result.skipped_title, 1)
        self.assertEqual(first.stats()["total"], 1)

    def test_unique_index_exists_on_norm_title(self):
        pool = self.open()
        indexes = {
            row["name"]: row["unique"] for row in pool._conn.execute("PRAGMA index_list('papers')")
        }
        self.assertEqual(indexes.get("idx_papers_norm_title"), 1)

    def test_untitled_papers_do_not_collide(self):
        # 標題無法正規化者存 NULL；UNIQUE 允許多個 NULL。
        pool = self.open()
        result = pool.ingest([record("a", ""), record("b", "   ")])
        self.assertEqual(result.inserted, 2)


# Codex 審查後補：交易邊界
class TransactionTests(PoolTestCase):
    def test_failure_mid_batch_rolls_back_the_whole_ingest(self):
        def records_then_fail():
            yield record("a", "A")
            raise RuntimeError("模擬來源中途失敗")

        with self.assertRaises(RuntimeError):
            self.pool.ingest(records_then_fail())
        self.assertEqual(self.pool.stats()["total"], 0)


# Codex 審查後補：寫入不存在的 id 要出聲
class UnknownIdTests(PoolTestCase):
    def test_storing_embedding_for_unknown_id_raises(self):
        # 靜默忽略會讓那篇永遠 pending，每次執行都重算一次向量。
        with self.assertRaises(PaperPoolError):
            self.pool.store_embeddings([("nope", unit(1))], "m1")

    def test_marking_unknown_id_as_recommended_raises(self):
        with self.assertRaises(PaperPoolError):
            self.pool.mark_recommended(["nope"])

    def test_duplicate_ids_in_one_call_count_once(self):
        self.fill([("a", "A", "2022-01-01", 1)])
        self.pool.mark_recommended(["a", "a", "a"])
        count = self.pool._conn.execute(
            "SELECT recommend_count FROM papers WHERE id = 'a'"
        ).fetchone()["recommend_count"]
        self.assertEqual(count, 1)

    def test_empty_recommendation_list_is_a_noop(self):
        self.pool.mark_recommended([])


# Codex 審查後補：不同模型的向量不可比
class EmbeddingModelTests(PoolTestCase):
    def setUp(self):
        super().setUp()
        self.today = date(2026, 9, 6)
        self.pool.ingest([record(i, i.upper(), "2022-01-01") for i in ("a", "b")])
        self.pool.store_embeddings([("a", unit(1))], "m1")
        self.pool.store_embeddings([("b", unit(2))], "m2")

    def test_mixed_models_are_rejected(self):
        with self.assertRaises(PaperPoolError):
            self.pool.select(unit(100), 0, 2, 7, today=self.today)

    def test_model_filter_selects_only_matching_vectors(self):
        picked = self.pool.select(unit(100), 0, 2, 7, today=self.today, model="m1")
        self.assertEqual([p.id for p in picked], ["a"])


class StatsTests(PoolTestCase):
    def test_stats_counts_each_state(self):
        self.fill([("a", "A", "2026-09-05", 1), ("b", "B", "2022-01-01", None)])
        self.pool.mark_carded(["A"])
        self.pool.mark_recommended(["b"])
        stats = self.pool.stats()
        self.assertEqual(stats["total"], 2)
        self.assertEqual(stats["carded"], 1)
        self.assertEqual(stats["embedded"], 1)
        self.assertEqual(stats["pending_embedding"], 1)
        self.assertEqual(stats["recommended"], 1)
        self.assertEqual(stats["by_source"], {"arxiv": 2})

    def test_stats_on_empty_pool(self):
        self.assertEqual(self.pool.stats()["total"], 0)


LEGACY_SCHEMA = """
CREATE TABLE papers(
  id TEXT PRIMARY KEY,
  norm_title TEXT NOT NULL,
  source TEXT NOT NULL,
  title TEXT NOT NULL,
  abstract TEXT NOT NULL,
  url TEXT,
  pdf_url TEXT,
  published_date TEXT,
  added_at TEXT NOT NULL,
  embedding BLOB,
  embedding_model TEXT,
  recommend_count INTEGER NOT NULL DEFAULT 0,
  last_recommended_at TEXT,
  carded INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX idx_papers_norm_title ON papers(norm_title);
CREATE INDEX idx_papers_pending ON papers(carded, embedding_model);
"""


def legacy_norm_title(title):
    """v2 早期的正規化：只小寫，保留連字號與空白。"""
    return title.lower()


class MigrationTests(unittest.TestCase):
    """舊資料庫必須真的被升級。

    `CREATE UNIQUE INDEX IF NOT EXISTS` 只看名稱，不看 UNIQUE 旗標；早期版本
    建的是同名的非 UNIQUE 索引，於是既有資料庫會靜默沿用舊索引，
    A3 的原子去重保證形同虛設。
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = str(Path(self.tmp.name) / "legacy.sqlite")

    def legacy(self, rows):
        conn = sqlite3.connect(self.db)
        conn.executescript(LEGACY_SCHEMA)
        conn.executemany(
            "INSERT INTO papers (id, norm_title, source, title, abstract, url, added_at,"
            " embedding, embedding_model, recommend_count, carded)"
            " VALUES (?, ?, 'arxiv', ?, 'a', '', '2026-01-01', ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
        conn.close()

    def index_is_unique(self):
        conn = sqlite3.connect(self.db)
        try:
            return next(
                row[2]
                for row in conn.execute("PRAGMA index_list(papers)")
                if row[1] == "idx_papers_norm_title"
            )
        finally:
            conn.close()

    def test_a_non_unique_index_is_upgraded(self):
        self.legacy([("a1", legacy_norm_title("Alpha"), "Alpha", None, None, 0, 0)])
        self.assertEqual(self.index_is_unique(), 0)
        with open_pool(self.db):
            pass
        self.assertEqual(self.index_is_unique(), 1)

    def test_the_upgraded_index_actually_blocks_duplicates(self):
        self.legacy([("a1", legacy_norm_title("Alpha Beta"), "Alpha Beta", None, None, 0, 0)])
        with open_pool(self.db) as pool:
            # 同一篇論文的另一個版本，標點不同但正規化後相同。
            result = pool.ingest(
                [PaperRecord(id="doi:1", source="openalex", title="Alpha-Beta", abstract="a")]
            )
            self.assertEqual(result.inserted, 0)
            self.assertEqual(result.skipped_title, 1)

    def test_norm_title_becomes_nullable(self):
        # 舊 schema 的 NOT NULL 會讓「標題無法正規化」的論文整批寫入失敗。
        self.legacy([("a1", legacy_norm_title("Alpha"), "Alpha", None, None, 0, 0)])
        with open_pool(self.db) as pool:
            result = pool.ingest([PaperRecord(id="x1", source="s", title="。。。", abstract="a")])
            self.assertEqual(result.inserted, 1)

    def test_norm_titles_are_recomputed_with_the_current_rules(self):
        # 舊值是 "edgephase: ... multi‐station ..."，與現行算法對不起來，
        # 比對已讀卡片會失敗而重複推薦。
        self.legacy([("a1", legacy_norm_title(EDGEPHASE_ARXIV), EDGEPHASE_ARXIV, None, None, 0, 0)])
        with open_pool(self.db) as pool:
            self.assertEqual(pool.mark_carded([EDGEPHASE_CARD]), [])
            row = pool._conn.execute("SELECT carded FROM papers WHERE id='a1'").fetchone()
            self.assertEqual(row["carded"], 1)

    def test_duplicates_are_resolved_keeping_the_most_complete_row(self):
        blob = unit(1).tobytes()
        self.legacy(
            [
                ("plain", "alpha", "Alpha", None, None, 0, 0),
                ("embedded", "alpha", "Alpha", blob, "m", 0, 0),
                ("carded", "alpha", "Alpha", None, None, 3, 1),
            ]
        )
        with open_pool(self.db) as pool:
            ids = [r["id"] for r in pool._conn.execute("SELECT id FROM papers")]
        # 已寫成卡片的那筆狀態最完整，必須留下來，否則那篇論文會被重新推薦。
        self.assertEqual(ids, ["carded"])

    def test_accumulated_state_survives_the_upgrade(self):
        blob = unit(2).tobytes()
        self.legacy([("a1", "alpha", "Alpha", blob, "m", 5, 1)])
        with open_pool(self.db) as pool:
            row = pool._conn.execute("SELECT * FROM papers WHERE id='a1'").fetchone()
        self.assertEqual(row["recommend_count"], 5)
        self.assertEqual(row["carded"], 1)
        self.assertEqual(row["embedding_model"], "m")
        self.assertEqual(row["embedding"], blob)

    def test_reopening_a_current_database_changes_nothing(self):
        with open_pool(self.db) as pool:
            pool.ingest([PaperRecord(id="a1", source="s", title="Alpha", abstract="a")])
        with open_pool(self.db) as pool:
            self.assertEqual(pool.stats()["total"], 1)
        self.assertEqual(self.index_is_unique(), 1)


class NonFiniteVectorTests(unittest.TestCase):
    """NaN 與 Infinity 除下去仍是 NaN，會讓分數變成沒有意義的亂序卻不報錯。"""

    def test_infinity_is_rejected(self):
        with open_pool(":memory:") as pool:
            pool.ingest([PaperRecord(id="p1", source="s", title="T", abstract="a")])
            with self.assertRaises(ValueError):
                pool.store_embeddings([("p1", [float("inf")] + [0.0] * (DIM - 1))], "m")
            self.assertEqual(pool.stats()["embedded"], 0)

    def test_nan_is_rejected(self):
        with open_pool(":memory:") as pool:
            pool.ingest([PaperRecord(id="p1", source="s", title="T", abstract="a")])
            with self.assertRaises(ValueError):
                pool.store_embeddings([("p1", [float("nan")] + [1.0] * (DIM - 1))], "m")

    def test_a_non_finite_history_vector_is_rejected(self):
        with self.assertRaises(ValueError):
            weighted_centroid([[float("nan")] * DIM])


class StoreEmbeddingsRollbackTests(unittest.TestCase):
    def test_an_unknown_id_rolls_back_the_whole_batch(self):
        # 拋錯前先 commit 有效的那幾筆，會讓呼叫端以為整批沒寫入，實際卻寫了一半。
        with open_pool(":memory:") as pool:
            pool.ingest([PaperRecord(id="known", source="s", title="T", abstract="a")])
            with self.assertRaises(PaperPoolError):
                pool.store_embeddings([("known", unit(3)), ("missing", unit(4))], "m")
            self.assertEqual(pool.stats()["embedded"], 0)


if __name__ == "__main__":
    unittest.main()
