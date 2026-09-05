"""驗證候選池的 13 條不變式。

全部離線：不連網、不呼叫任何 API，向量以固定亂數種子產生。
"""

import sys
import unittest
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paper_pool import (  # noqa: E402
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


if __name__ == "__main__":
    unittest.main()
