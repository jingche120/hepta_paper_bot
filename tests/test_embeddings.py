"""驗證向量計算。以假的 client 取代 API，全部離線。"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from embeddings import (  # noqa: E402
    MAX_CHARS,
    ApiEmbedder,
    EmbeddingResponseError,
    embed_pending,
    embed_texts,
)
from paper_pool import PaperRecord, open_pool  # noqa: E402

DIM = 4


class FakeClient:
    def __init__(self, shuffle=False):
        self.calls = []
        self.shuffle = shuffle
        self.embeddings = SimpleNamespace(create=self._create)

    def _create(self, input, model):
        self.calls.append({"input": list(input), "model": model})
        data = [
            SimpleNamespace(index=i, embedding=[float(len(t)), 1.0, 0.0, 0.0])
            for i, t in enumerate(input)
        ]
        if self.shuffle:
            data.reverse()
        return SimpleNamespace(data=data)


class EmbedTextsTests(unittest.TestCase):
    def test_batches_respect_the_batch_size(self):
        client = FakeClient()
        embed_texts(client, "m", ["a", "b", "c", "d", "e"], batch_size=2)
        self.assertEqual([len(c["input"]) for c in client.calls], [2, 2, 1])

    def test_output_order_matches_input_even_if_api_reorders(self):
        # API 不保證回傳順序；錯位會把向量配到錯的論文上。
        client = FakeClient(shuffle=True)
        vectors = embed_texts(client, "m", ["a", "bb", "ccc"], batch_size=3)
        self.assertEqual([v[0] for v in vectors], [1.0, 2.0, 3.0])

    def test_overlong_text_is_truncated(self):
        client = FakeClient()
        embed_texts(client, "m", ["x" * (MAX_CHARS + 500)])
        self.assertEqual(len(client.calls[0]["input"][0]), MAX_CHARS)

    def test_model_is_passed_through(self):
        client = FakeClient()
        embed_texts(client, "text-embedding-3-small", ["a"])
        self.assertEqual(client.calls[0]["model"], "text-embedding-3-small")

    def test_empty_input_makes_no_calls(self):
        client = FakeClient()
        self.assertEqual(embed_texts(client, "m", []), [])
        self.assertEqual(client.calls, [])

    def test_zero_batch_size_is_rejected(self):
        with self.assertRaises(ValueError):
            embed_texts(FakeClient(), "m", ["a"], batch_size=0)


class EmbedPendingTests(unittest.TestCase):
    def setUp(self):
        self.pool = open_pool(":memory:")
        self.addCleanup(self.pool.close)
        self.pool.ingest(
            [
                PaperRecord(id=f"p{i}", source="arxiv", title=f"T{i}", abstract=f"abstract {i}")
                for i in range(5)
            ]
        )

    def test_all_pending_papers_get_vectors(self):
        client = FakeClient()
        self.assertEqual(
            embed_pending(self.pool, ApiEmbedder(client, "m", 2), show_progress=False), 5
        )
        self.assertEqual(self.pool.stats()["pending_embedding"], 0)

    def test_second_run_computes_nothing(self):
        client = FakeClient()
        embed_pending(self.pool, ApiEmbedder(client, "m", 2), show_progress=False)
        before = len(client.calls)
        self.assertEqual(
            embed_pending(self.pool, ApiEmbedder(client, "m", 2), show_progress=False), 0
        )
        # 沒有新的 API 呼叫：這是每天的成本控制關鍵。
        self.assertEqual(len(client.calls), before)

    def test_changing_model_recomputes_everything(self):
        client = FakeClient()
        embed_pending(self.pool, ApiEmbedder(client, "m1", 5), show_progress=False)
        self.assertEqual(
            embed_pending(self.pool, ApiEmbedder(client, "m2", 5), show_progress=False), 5
        )

    def test_vectors_land_on_the_right_papers(self):
        # abstract "abstract 0" 長度 10、"abstract 4" 也是 10，改用不同長度來分辨。
        pool = open_pool(":memory:")
        self.addCleanup(pool.close)
        pool.ingest(
            [
                PaperRecord(id="short", source="arxiv", title="A", abstract="ab"),
                PaperRecord(id="long", source="arxiv", title="B", abstract="abcdef"),
            ]
        )
        embed_pending(pool, ApiEmbedder(FakeClient(shuffle=True), "m", 2), show_progress=False)
        picked = {p.id: p for p in pool.select([1.0, 0.0, 0.0, 0.0], 0, 2, 7)}
        # 向量第一維等於摘要長度，正規化後長度較長者的分數較高。
        self.assertGreater(picked["long"].score, picked["short"].score)

    def test_empty_pool_is_a_noop(self):
        pool = open_pool(":memory:")
        self.addCleanup(pool.close)
        client = FakeClient()
        self.assertEqual(embed_pending(pool, ApiEmbedder(client, "m"), show_progress=False), 0)
        self.assertEqual(client.calls, [])


class GetEmbedderTests(unittest.TestCase):
    def config(self, kind):
        from omegaconf import OmegaConf

        return OmegaConf.create(
            {
                "executor": {"reranker": kind},
                "reranker": {
                    "api": {
                        "key": "k",
                        "base_url": None,
                        "model": "text-embedding-3-small",
                        "batch_size": 32,
                    },
                    "local": {"model": "some/model", "encode_kwargs": {"task": "retrieval"}},
                },
            }
        )

    def test_api_embedder_uses_the_embedding_credentials(self):
        from embeddings import ApiEmbedder, get_embedder

        client = FakeClient()
        embedder = get_embedder(self.config("api"), client)
        self.assertIsInstance(embedder, ApiEmbedder)
        self.assertEqual(embedder.model, "text-embedding-3-small")
        self.assertEqual(embedder.batch_size, 32)

    def test_local_embedder_is_namespaced_to_avoid_clashing_with_api_models(self):
        # 池中以模型名判斷向量是否需要重算；本機與 API 同名模型不可混為一談。
        from embeddings import LocalEmbedder, get_embedder

        embedder = get_embedder(self.config("local"))
        self.assertIsInstance(embedder, LocalEmbedder)
        self.assertEqual(embedder.model, "local:some/model")

    def test_unknown_kind_is_rejected(self):
        from embeddings import get_embedder

        with self.assertRaises(ValueError):
            get_embedder(self.config("nope"))


class BrokenClient:
    """回傳的 index 與送出的批次對不起來。"""

    def __init__(self, indices):
        self.indices = indices
        self.embeddings = SimpleNamespace(create=self._create)

    def _create(self, input, model):
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=i, embedding=[float(i), 1.0, 0.0, 0.0]) for i in self.indices
            ]
        )


class ResponseIntegrityTests(unittest.TestCase):
    """只排序不夠：缺號、重複或數量不符都會靜默把向量配到錯的論文上。"""

    def test_a_missing_index_is_rejected(self):
        # 送 2 筆只回 index 1：zip 會把它配給第 0 篇，且不會有任何錯誤。
        with self.assertRaises(EmbeddingResponseError):
            embed_texts(BrokenClient([1]), "m", ["a", "b"])

    def test_a_duplicate_index_is_rejected(self):
        with self.assertRaises(EmbeddingResponseError):
            embed_texts(BrokenClient([0, 0]), "m", ["a", "b"])

    def test_an_out_of_range_index_is_rejected(self):
        with self.assertRaises(EmbeddingResponseError):
            embed_texts(BrokenClient([0, 5]), "m", ["a", "b"])

    def test_extra_vectors_are_rejected(self):
        with self.assertRaises(EmbeddingResponseError):
            embed_texts(BrokenClient([0, 1, 2]), "m", ["a", "b"])

    def test_a_complete_response_still_passes(self):
        self.assertEqual(len(embed_texts(BrokenClient([1, 0]), "m", ["a", "b"])), 2)


class ShortEmbedderTests(unittest.TestCase):
    """embed_pending 的 zip 會靜默截斷，數量不符必須擋下來。"""

    class Short:
        model = "m"
        batch_size = 8

        def embed(self, texts):
            return [[1.0, 0.0, 0.0, 0.0]] * (len(texts) - 1)

    def test_too_few_vectors_raises_and_writes_nothing(self):
        with open_pool(":memory:") as pool:
            pool.ingest(
                PaperRecord(id=f"p{i}", source="s", title=f"T{i}", abstract="a") for i in range(2)
            )
            with self.assertRaises(EmbeddingResponseError):
                embed_pending(pool, self.Short(), show_progress=False)
            self.assertEqual(pool.stats()["embedded"], 0)


if __name__ == "__main__":
    unittest.main()
