"""把論文摘要轉成向量。

backfill 與每日流程共用同一份邏輯，避免兩邊的批次大小、截斷規則或錯誤處理
各自演化。實際的正規化在 `PaperPool.store_embeddings()` 完成。
"""

from loguru import logger
from tqdm import tqdm

DEFAULT_BATCH_SIZE = 64
# embedding 模型單筆輸入上限約 8,191 tokens。論文摘要通常在 400 tokens 左右，
# 但異常長的紀錄會讓整批請求失敗，因此保守截斷。
MAX_CHARS = 20000


def embed_texts(client, model, texts, batch_size=DEFAULT_BATCH_SIZE):
    """分批向 OpenAI 相容 API 取得向量，回傳順序與輸入一致。"""
    if batch_size < 1:
        raise ValueError("batch_size 必須大於 0。")
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = [text[:MAX_CHARS] for text in texts[start : start + batch_size]]
        response = client.embeddings.create(input=batch, model=model)
        # API 不保證回傳順序，依 index 排回原順序。錯位會把向量配到錯的論文上，
        # 而且不會有任何錯誤訊息。
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors.extend(item.embedding for item in ordered)
    return vectors


class ApiEmbedder:
    """走 OpenAI 相容的 embedding API。"""

    def __init__(self, client, model, batch_size=DEFAULT_BATCH_SIZE):
        self.client = client
        self.model = model
        self.batch_size = batch_size

    def embed(self, texts):
        return embed_texts(self.client, self.model, texts, self.batch_size)


class LocalEmbedder:
    """走本機 sentence-transformers 模型，不呼叫 API、不花錢。

    模型名稱前綴 `local:`，避免與同名的 API 模型在池中被誤認為同一種向量。
    """

    def __init__(self, model_name, encode_kwargs=None, batch_size=DEFAULT_BATCH_SIZE):
        self.model = f"local:{model_name}"
        self._model_name = model_name
        self._encode_kwargs = dict(encode_kwargs or {})
        self.batch_size = batch_size
        self._encoder = None

    def _load(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self._model_name, trust_remote_code=True)
        return self._encoder

    def embed(self, texts):
        if not texts:
            return []
        encoder = self._load()
        truncated = [text[:MAX_CHARS] for text in texts]
        return [list(vector) for vector in encoder.encode(truncated, **self._encode_kwargs)]


def get_embedder(config, openai_client=None):
    """依 `executor.reranker` 建立 embedder，沿用既有的設定鍵。"""
    kind = config.executor.reranker
    if kind == "api":
        if openai_client is None:
            from openai import OpenAI

            openai_client = OpenAI(
                api_key=config.reranker.api.key, base_url=config.reranker.api.base_url
            )
        return ApiEmbedder(
            openai_client,
            config.reranker.api.model,
            config.reranker.api.get("batch_size") or DEFAULT_BATCH_SIZE,
        )
    if kind == "local":
        return LocalEmbedder(
            config.reranker.local.model, config.reranker.local.get("encode_kwargs")
        )
    raise ValueError(f"未知的 executor.reranker：{kind}")


def embed_pending(pool, embedder, batch_size=None, show_progress=True):
    """把池中還沒有向量的論文補齊，回傳實際計算的筆數。

    只處理 `pending_embeddings()` 回傳的列，因此重複執行不會重算已有的向量——
    少了這層，每天都會重算整個池子（19,200 篇約 $0.15/天）。
    """
    model = embedder.model
    batch_size = batch_size or getattr(embedder, "batch_size", DEFAULT_BATCH_SIZE)
    pending = pool.pending_embeddings(model)
    if not pending:
        return 0
    logger.info(f"Computing embeddings for {len(pending)} papers with {model}")

    batches = range(0, len(pending), batch_size)
    if show_progress:
        batches = tqdm(
            batches, total=(len(pending) + batch_size - 1) // batch_size, desc="Embedding"
        )
    done = 0
    for start in batches:
        chunk = pending[start : start + batch_size]
        vectors = embedder.embed([abstract for _, abstract in chunk])
        # 每批寫入後即提交，中斷時已算好的向量不會白費。
        pool.store_embeddings(
            ((paper_id, vector) for (paper_id, _), vector in zip(chunk, vectors)), model
        )
        done += len(chunk)
    return done
