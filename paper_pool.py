"""候選論文池：儲存、去重、評分、選取。

與 v1 的差別是候選論文不再用完即丟。所有檢索到的論文都存進 SQLite，
每天從整個池子挑出最貼近閱讀紀錄的幾篇；沒被選上的留在池中等下次機會。

本模組**刻意不連網**——embedding 由呼叫端算好傳進來，因此可完整離線測試。
"""

import sqlite3
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

EMBEDDING_DTYPE = np.float32
# 與 reranker/base.py 的分數尺度一致，方便兩邊的數值互相比對。
SCORE_SCALE = 10

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers(
  id TEXT PRIMARY KEY,
  norm_title TEXT,
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
-- UNIQUE 讓「同一篇論文只進池一次」由資料庫保證，而不是靠應用層的先查再寫。
-- 標題無法正規化者存 NULL；SQLite 的 UNIQUE 允許多個 NULL，因此不會互撞。
CREATE UNIQUE INDEX IF NOT EXISTS idx_papers_norm_title ON papers(norm_title);
CREATE INDEX IF NOT EXISTS idx_papers_pending ON papers(carded, embedding_model);
"""


class PaperPoolError(RuntimeError):
    """候選池的資料狀態不正確。"""


def normalize_title(title):
    """跨來源比對用的標題正規化。

    同一篇論文在 Heptabase 卡片、arXiv 與 OpenAlex 之間常有標點差異。
    真實案例：卡片的 "Multi‐Station" 用 U+2010 HYPHEN，arXiv 用 ASCII U+002D，
    直接字串比對會失敗，導致已讀論文被重複推薦。

    因此先做 NFKD 分解與大小寫折疊，再只保留英數與 CJK 等文字字元；
    連字號、冒號、空白等一律去除，`multi-station` 與 `multi station` 皆成為
    `multistation`。
    """
    folded = unicodedata.normalize("NFKD", title or "").casefold()
    return "".join(ch for ch in folded if ch.isalnum())


def _l2_normalize(array):
    """把向量正規化成單位長度。

    分數公式 `(向量 · 質心) * 10` 只有在雙方都是單位向量時才等於餘弦相似度。
    OpenAI 的 embedding 恰好已正規化，但本機模型與未來的來源不保證如此，
    因此在寫入與合成質心時強制正規化，讓這個前提由結構保證而非呼叫端自律。
    """
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("零向量無法正規化。")
    return (array / norms).astype(EMBEDDING_DTYPE)


def time_decay_weights(count):
    """與 reranker/base.py 相同的時間衰減權重，順序為新到舊。"""
    if count <= 0:
        raise ValueError("time_decay_weights 需要至少一筆閱讀紀錄。")
    weights = 1 / (1 + np.log10(np.arange(count) + 1))
    return (weights / weights.sum()).astype(EMBEDDING_DTYPE)


def weighted_centroid(vectors, weights=None):
    """把閱讀紀錄的向量合成單一質心向量。

    對 L2 正規化的向量，「逐篇算相似度再加權平均」與「先合成質心再算一次內積」
    恆等，因為內積是線性的：sum_i w_i (c·h_i) = c·(sum_i w_i h_i)。
    用質心可少算 len(history) 倍的乘法，排序與分數都不變。
    """
    matrix = _l2_normalize(np.asarray(vectors, dtype=EMBEDDING_DTYPE))
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("weighted_centroid 需要形狀為 (n, dim) 且非空的向量。")
    if weights is None:
        weights = time_decay_weights(matrix.shape[0])
    weights = np.asarray(weights, dtype=EMBEDDING_DTYPE)
    if weights.shape != (matrix.shape[0],):
        raise ValueError("權重數量與向量數量不一致。")
    return matrix.T @ weights


@dataclass(frozen=True)
class PaperRecord:
    """要寫進池子的一篇論文。"""

    id: str
    source: str
    title: str
    abstract: str
    url: str = ""
    pdf_url: str | None = None
    published_date: str | None = None


@dataclass
class PooledPaper:
    """從池中選出的一篇論文，附帶本次評分。"""

    id: str
    title: str
    abstract: str
    url: str
    pdf_url: str | None
    published_date: str | None
    score: float
    recommend_count: int


@dataclass(frozen=True)
class IngestResult:
    inserted: int
    skipped_id: int
    skipped_title: int

    @property
    def skipped(self):
        return self.skipped_id + self.skipped_title


def _utcnow_iso():
    return datetime.now(timezone.utc).isoformat()


class PaperPool:
    """候選池。所有寫入都在交易中完成，中途失敗不會留下半寫入狀態。"""

    def __init__(self, connection):
        self._conn = connection
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False

    def close(self):
        self._conn.close()

    # ---------- 寫入 ----------

    def ingest(self, records):
        """寫入論文，以 id 與正規化標題雙重去重。

        已存在的論文**不會**被覆蓋——`recommend_count`、`carded`、`embedding`
        都是累積的狀態，重新檢索到同一篇不該把它們清掉。
        """
        inserted = skipped_id = skipped_title = 0
        now = _utcnow_iso()
        with self._conn:
            for record in records:
                # 先寫入再判斷結果，而不是「先查再寫」——後者在兩個程序同時
                # ingest 時（例如回填正在跑，每日任務剛好觸發）會讓同一篇論文
                # 兩個版本都進池。ON CONFLICT 由 PRIMARY KEY 與 norm_title 的
                # UNIQUE 索引共同把關，是原子的。
                cursor = self._conn.execute(
                    "INSERT INTO papers (id, norm_title, source, title, abstract, url,"
                    " pdf_url, published_date, added_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
                    (
                        record.id,
                        normalize_title(record.title) or None,
                        record.source,
                        record.title,
                        record.abstract,
                        record.url,
                        record.pdf_url,
                        record.published_date,
                        now,
                    ),
                )
                if cursor.rowcount == 1:
                    inserted += 1
                elif self._conn.execute(
                    "SELECT 1 FROM papers WHERE id = ?", (record.id,)
                ).fetchone():
                    skipped_id += 1
                else:
                    # id 沒撞到卻寫不進去，代表撞的是正規化標題：
                    # 同一篇論文的 arXiv 預印本與 IEEE 正式版。
                    skipped_title += 1
        return IngestResult(inserted, skipped_id, skipped_title)

    def mark_carded(self, reading_history):
        """把已寫成 Heptabase 卡片的論文標記為已讀，回傳沒比對上的卡片標題。

        回傳未匹配清單而不是靜默略過，是因為卡片標題可能根本不是論文名
        （例如 `來源論文：` 指向 arXiv 查詢網址時），那種卡片對應的論文會被
        當成未讀而重複推薦，使用者需要知道。
        """
        unmatched = []
        with self._conn:
            for item in reading_history:
                title = item if isinstance(item, str) else item.title
                norm_title = normalize_title(title)
                if not norm_title:
                    unmatched.append(title)
                    continue
                cursor = self._conn.execute(
                    "UPDATE papers SET carded = 1 WHERE norm_title = ?", (norm_title,)
                )
                if cursor.rowcount == 0:
                    unmatched.append(title)
        return unmatched

    def store_embeddings(self, items, model):
        """寫入向量。items 為 (paper_id, vector) 的可迭代物件。

        向量一律先正規化成單位長度，`select()` 的分數才等於餘弦相似度。
        寫到不存在的 id 會拋錯而非靜默忽略——否則那篇會永遠留在
        `pending_embeddings()` 裡，每次執行都重算一次，是實際的成本漏洞。
        """
        missing = []
        with self._conn:
            for paper_id, vector in items:
                array = np.asarray(vector, dtype=EMBEDDING_DTYPE)
                if array.ndim != 1 or array.size == 0:
                    raise ValueError(f"{paper_id} 的向量必須是非空的一維陣列。")
                cursor = self._conn.execute(
                    "UPDATE papers SET embedding = ?, embedding_model = ? WHERE id = ?",
                    (_l2_normalize(array).tobytes(), model, paper_id),
                )
                if cursor.rowcount == 0:
                    missing.append(paper_id)
        if missing:
            raise PaperPoolError(f"這些 id 不在池中，向量沒有寫入：{missing}")

    def mark_recommended(self, papers, when=None):
        """把選出的論文計入推薦次數。這是輪替機制的依據。

        同一次呼叫中重複出現的 id 只計一次；重複計數會讓該篇被錯誤地推到
        隊伍更後方，扭曲輪替順序。
        """
        when = when or _utcnow_iso()
        unique_ids = list(dict.fromkeys(p if isinstance(p, str) else p.id for p in papers))
        if not unique_ids:
            return
        with self._conn:
            cursor = self._conn.executemany(
                "UPDATE papers SET recommend_count = recommend_count + 1,"
                " last_recommended_at = ? WHERE id = ?",
                [(when, paper_id) for paper_id in unique_ids],
            )
            if cursor.rowcount != len(unique_ids):
                found = {
                    row["id"]
                    for row in self._conn.execute(
                        "SELECT id FROM papers WHERE id IN (%s)" % ",".join("?" * len(unique_ids)),
                        unique_ids,
                    )
                }
                raise PaperPoolError(
                    f"這些 id 不在池中，推薦次數沒有更新：{sorted(set(unique_ids) - found)}"
                )

    # ---------- 讀取 ----------

    def pending_embeddings(self, model, limit=None):
        """回傳還需要計算向量的 (id, abstract)。

        已有相同模型向量的論文**不會**回傳——這是成本控制的關鍵，
        少了這道過濾就會每天重算整個池子的 embedding。
        """
        sql = (
            "SELECT id, abstract FROM papers"
            " WHERE embedding IS NULL OR embedding_model IS NULL OR embedding_model <> ?"
            " ORDER BY added_at, id"
        )
        params = [model]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return [(row["id"], row["abstract"]) for row in self._conn.execute(sql, params)]

    def select(self, centroid, n_new, n_old, new_within_days, today=None, model=None):
        """依配額與輪替排序選出要推薦的論文。

        指定 `model` 只採用該模型算出的向量。未指定時，若池中存在多個模型的
        向量會直接拋錯——不同模型的向量不可比，混在一起評分會靜默給出錯誤排名。

        排序鍵是 `(recommend_count ASC, score DESC)`：推薦過的排到隊伍後方，
        等其他高分論文都輪過一遍才會再出現，因此不會連續多天重複同一篇，
        但也不會有論文被永久丟棄——只有寫成卡片才會被排除。
        """
        if n_new < 0 or n_old < 0:
            raise ValueError("n_new 與 n_old 不得為負數。")
        sql = (
            "SELECT id, title, abstract, url, pdf_url, published_date, embedding,"
            " embedding_model, recommend_count"
            " FROM papers WHERE carded = 0 AND embedding IS NOT NULL"
        )
        params = []
        if model is not None:
            sql += " AND embedding_model = ?"
            params.append(model)
        rows = self._conn.execute(sql, params).fetchall()
        if not rows:
            return []

        models = {row["embedding_model"] for row in rows}
        if len(models) > 1:
            raise PaperPoolError(
                f"池中混用了多個 embedding 模型 {sorted(models)}，向量不可比。"
                " 請指定 model 參數，或重新計算整個池子的向量。"
            )

        blobs = [row["embedding"] for row in rows]
        dim = len(blobs[0]) // EMBEDDING_DTYPE().itemsize
        if any(len(b) != len(blobs[0]) for b in blobs):
            raise PaperPoolError("池中存在維度不一致的向量，請重新計算 embedding。")
        matrix = np.frombuffer(b"".join(blobs), dtype=EMBEDDING_DTYPE).reshape(len(rows), dim)

        vector = np.asarray(centroid, dtype=EMBEDDING_DTYPE).reshape(-1)
        if vector.shape[0] != dim:
            raise PaperPoolError(f"質心維度 {vector.shape[0]} 與池中向量維度 {dim} 不符。")
        scores = (matrix @ vector) * SCORE_SCALE

        papers = [
            PooledPaper(
                id=row["id"],
                title=row["title"],
                abstract=row["abstract"],
                url=row["url"],
                pdf_url=row["pdf_url"],
                published_date=row["published_date"],
                score=float(score),
                recommend_count=row["recommend_count"],
            )
            for row, score in zip(rows, scores)
        ]

        today = today or date.today()
        cutoff = (today - timedelta(days=new_within_days)).isoformat()
        # 發表日剛好等於 cutoff 者算「新」；沒有發表日的一律歸為「舊」。
        fresh = [p for p in papers if p.published_date and p.published_date >= cutoff]
        backlog = [p for p in papers if not (p.published_date and p.published_date >= cutoff)]

        def rotation_key(paper):
            return (paper.recommend_count, -paper.score)

        fresh.sort(key=rotation_key)
        backlog.sort(key=rotation_key)
        return fresh[:n_new] + backlog[:n_old]

    def stats(self):
        row = self._conn.execute(
            "SELECT COUNT(*) AS total,"
            " SUM(carded) AS carded,"
            " SUM(embedding IS NOT NULL) AS embedded,"
            " SUM(recommend_count > 0) AS recommended FROM papers"
        ).fetchone()
        by_source = {
            r["source"]: r["n"]
            for r in self._conn.execute(
                "SELECT source, COUNT(*) AS n FROM papers GROUP BY source ORDER BY source"
            )
        }
        total = row["total"] or 0
        return {
            "total": total,
            "carded": row["carded"] or 0,
            "embedded": row["embedded"] or 0,
            "pending_embedding": total - (row["embedded"] or 0),
            "recommended": row["recommended"] or 0,
            "by_source": by_source,
        }


def open_pool(db_path):
    """開啟（必要時建立）候選池。傳入 ":memory:" 可用於測試。"""
    if db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return PaperPool(sqlite3.connect(db_path))
