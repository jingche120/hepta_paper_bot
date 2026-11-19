"""把論文摘要轉成向量。

backfill 與每日流程共用同一份邏輯，避免兩邊的批次大小、截斷規則或錯誤處理
各自演化。實際的正規化在 `PaperPool.store_embeddings()` 完成。
"""

import time
from collections import deque

from loguru import logger
from tqdm import tqdm

DEFAULT_BATCH_SIZE = 64
# embedding API 的限制是 tokens per minute，不是請求數。1M TPM 是 OpenAI 目前
# 常見的預設值；你的方案不同就用 --tpm 覆蓋。
DEFAULT_TPM = 1_000_000
# 只用來估算節流額度，不需要精確：英文摘要大約 4 個字元 1 個 token。
CHARS_PER_TOKEN = 4
RETRY_ATTEMPTS = 6
RETRY_BASE_DELAY = 2.0
# embedding 模型單筆輸入上限約 8,191 tokens。論文摘要通常在 400 tokens 左右，
# 但異常長的紀錄會讓整批請求失敗，因此保守截斷。
MAX_CHARS = 20000


class EmbeddingResponseError(RuntimeError):
    """embedding 回應與送出的批次對不起來。"""


class TokenBudget:
    """把送出的 token 數控制在每分鐘上限之內。

    回填一次要送 660 萬 tokens。不節流的話實測是每秒 1.25 批、每批約 24,000
    tokens，等於每分鐘 185 萬——1M TPM 上限的 1.85 倍，跑幾十秒就會 429。

    重試能救回來，但整趟回填會變成不斷撞牆再等待；主動配速才是對的作法。
    """

    def __init__(self, tokens_per_minute=DEFAULT_TPM, window=60.0, clock=None, sleeper=None):
        if tokens_per_minute <= 0:
            raise ValueError("tokens_per_minute 必須大於 0。")
        self.limit = tokens_per_minute
        self.window = window
        self._clock = clock or time.monotonic
        self._sleep = sleeper or time.sleep
        self._events = deque()
        self._spent = 0

    def _expire(self, now):
        while self._events and now - self._events[0][0] >= self.window:
            self._spent -= self._events.popleft()[1]

    def spend(self, tokens):
        """在送出前呼叫。額度不足就睡到最舊的紀錄滑出視窗，回傳總共睡了幾秒。

        單一批次本身就超過上限時直接放行——睡再久也不會夠，讓 API 自己拒絕，
        免得無限迴圈。
        """
        now = self._clock()
        self._expire(now)
        slept = 0.0
        while self._events and self._spent + tokens > self.limit:
            wait = self.window - (now - self._events[0][0])
            if wait > 0:
                self._sleep(wait)
                slept += wait
            now = self._clock()
            self._expire(now)
        self._events.append((now, tokens))
        self._spent += tokens
        return slept


def _is_rate_limit(exc):
    """用鴨子型別判斷，避免在模組層匯入 openai（本機模式不需要它）。"""
    return getattr(exc, "status_code", None) == 429


def _create_with_retry(client, model, batch, sleeper=None):
    """撞到速率上限時指數退避重試。

    配速只是估算（token 數用字元數推算），仍可能偏差；而回填跑到一半死掉
    的代價是重跑整趟，所以還是要有重試兜底。
    """
    sleeper = sleeper or time.sleep
    for attempt in range(RETRY_ATTEMPTS):
        try:
            return client.embeddings.create(input=batch, model=model)
        except Exception as exc:
            if not _is_rate_limit(exc) or attempt == RETRY_ATTEMPTS - 1:
                raise
            delay = RETRY_BASE_DELAY * 2**attempt
            logger.warning(
                f"embedding 撞到速率上限，{delay:.0f} 秒後重試"
                f"（第 {attempt + 1}/{RETRY_ATTEMPTS - 1} 次）。"
            )
            sleeper(delay)


def _ordered_embeddings(data, expected):
    """把回應排回輸入順序，並確認每一筆都回來了。

    API 不保證回傳順序，所以要依 index 排序；但只排序不夠。少回一筆、index 重複
    或缺號時，呼叫端的 zip 會把向量配到**錯的論文**上，而且完全不會報錯——
    錯誤的向量會就此永久存進池子，之後每天都用它來評分。

    因此這裡要求 index 恰好是 0..n-1 的完整集合，不符就整批失敗。
    重跑一批的代價遠低於靜默寫入錯誤資料。
    """
    items = list(data)
    indices = sorted(item.index for item in items)
    if indices != list(range(expected)):
        raise EmbeddingResponseError(
            f"embedding API 送出 {expected} 筆卻回傳 index {indices}，無法確定向量對應哪一篇論文。"
        )
    return [item.embedding for item in sorted(items, key=lambda item: item.index)]


def embed_texts(client, model, texts, batch_size=DEFAULT_BATCH_SIZE, budget=None):
    """分批向 OpenAI 相容 API 取得向量，回傳順序與輸入一致。"""
    if batch_size < 1:
        raise ValueError("batch_size 必須大於 0。")
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = [text[:MAX_CHARS] for text in texts[start : start + batch_size]]
        if budget is not None:
            budget.spend(sum(len(text) for text in batch) // CHARS_PER_TOKEN)
        response = _create_with_retry(client, model, batch)
        vectors.extend(_ordered_embeddings(response.data, len(batch)))
    return vectors


class ApiEmbedder:
    """走 OpenAI 相容的 embedding API。"""

    def __init__(self, client, model, batch_size=DEFAULT_BATCH_SIZE, tokens_per_minute=DEFAULT_TPM):
        self.client = client
        self.model = model
        self.batch_size = batch_size
        self.budget = TokenBudget(tokens_per_minute)

    def embed(self, texts):
        return embed_texts(self.client, self.model, texts, self.batch_size, self.budget)


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
            config.reranker.api.get("tokens_per_minute") or DEFAULT_TPM,
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
    outdated = {m: n for m, n in pool.embedding_models().items() if m != model}
    if outdated:
        # 一個拼錯的模型名稱就會讓整池重算並重新計費，這不該悄悄發生。
        listed = "、".join(f"{m}（{n} 篇）" for m, n in sorted(outdated.items()))
        logger.warning(
            f"池中已有其他模型算好的向量：{listed}。改用 {model} 會把它們全部重算並重新計費。"
        )
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
        # 下面的 zip 會靜默截斷，數量不符時前面的論文會拿到別人的向量。
        # 這道檢查同時涵蓋本機 embedder，不只 API。
        if len(vectors) != len(chunk):
            raise EmbeddingResponseError(
                f"這批有 {len(chunk)} 篇論文，embedder 卻回傳 {len(vectors)} 個向量。"
            )
        # 每批寫入後即提交，中斷時已算好的向量不會白費。
        pool.store_embeddings(
            ((paper_id, vector) for (paper_id, _), vector in zip(chunk, vectors)), model
        )
        done += len(chunk)
    return done
