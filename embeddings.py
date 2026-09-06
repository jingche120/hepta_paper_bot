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
    """分批取得向量，回傳順序與輸入一致。"""
    if batch_size < 1:
        raise ValueError("batch_size 必須大於 0。")
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = [text[:MAX_CHARS] for text in texts[start : start + batch_size]]
        response = client.embeddings.create(input=batch, model=model)
        # API 不保證回傳順序，依 index 排回原順序。
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors.extend(item.embedding for item in ordered)
    return vectors


def embed_pending(pool, client, model, batch_size=DEFAULT_BATCH_SIZE, show_progress=True):
    """把池中還沒有向量的論文補齊，回傳實際計算的筆數。

    只處理 `pending_embeddings()` 回傳的列，因此重複執行不會重算已有的向量——
    少了這層，每天都會重算整個池子（19,200 篇約 $0.15/天）。
    """
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
        vectors = embed_texts(client, model, [abstract for _, abstract in chunk], batch_size)
        # 每批寫入後即提交，中斷時已算好的向量不會白費。
        pool.store_embeddings(
            ((paper_id, vector) for (paper_id, _), vector in zip(chunk, vectors)), model
        )
        done += len(chunk)
    return done
