"""從 OpenAlex 取得期刊論文。

IEEE TGRS 與 GRSL 不在 arXiv 上，而 IEEE Xplore 的 API 需要訂閱。
OpenAlex 免費、不需金鑰，且這兩本期刊的摘要覆蓋率實測為 100%
（各抽樣 100 篇，平均長度 1,300–1,600 字元）。

摘要在 OpenAlex 是以 inverted index 儲存的，需要還原成文字才能算 embedding。

**注意 IEEE 的日期語意**：這兩本期刊的 `publication_date` 是「期號日期」而非實際
上線日。實測 2026 年的 GRSL 與 TGRS 各抽樣 200 篇，`publication_date` 全部是
`2026-01-01`。因此每日增量必須用 `created_date`（進入 OpenAlex 的日期），
用 `publication_date` 會抓不到任何新論文。回填則兩者皆可，用 publication 較直觀。
"""

from collections.abc import Iterator
from time import sleep

import requests
from loguru import logger

from paper_pool import PaperRecord

API_URL = "https://api.openalex.org/works"
PAGE_SIZE = 200
# 只取需要的欄位，回應體積小很多；abstract_inverted_index 一定要在裡面。
SELECT_FIELDS = "id,doi,display_name,publication_date,abstract_inverted_index,primary_location"

# 已知的期刊 source id。
TGRS = "S111326731"  # IEEE Transactions on Geoscience and Remote Sensing
GRSL = "S126920919"  # IEEE Geoscience and Remote Sensing Letters


class OpenAlexError(RuntimeError):
    """OpenAlex 回應無法使用。"""


def abstract_from_inverted_index(inverted):
    """把 `{詞: [位置, ...]}` 還原成文字；沒有摘要時回傳 None。"""
    if not inverted:
        return None
    positions = {}
    for word, indexes in inverted.items():
        for index in indexes:
            positions[index] = word
    if not positions:
        return None
    return " ".join(positions[i] for i in sorted(positions))


def _paper_id(work):
    """優先用 DOI 當識別碼，這樣同一篇論文在不同來源會得到相同的 id。"""
    doi = work.get("doi")
    if doi:
        return "doi:" + doi.removeprefix("https://doi.org/").lower()
    openalex_id = (work.get("id") or "").rsplit("/", 1)[-1]
    return f"openalex:{openalex_id}" if openalex_id else ""


def work_to_record(work):
    """把一筆 OpenAlex work 轉成 PaperRecord；資料不完整時回傳 None。

    沒有摘要的論文無法參與相似度比對，留在池中只會佔位，因此直接略過。
    """
    paper_id = _paper_id(work)
    title = (work.get("display_name") or "").strip()
    abstract = abstract_from_inverted_index(work.get("abstract_inverted_index"))
    if not paper_id or not title or not abstract:
        return None
    location = work.get("primary_location") or {}
    return PaperRecord(
        id=paper_id,
        source="openalex",
        title=title,
        abstract=abstract,
        url=location.get("landing_page_url") or work.get("id") or "",
        # IEEE 期刊有付費牆，通常沒有可取得的 PDF；後續抓全文會自動回退成摘要。
        pdf_url=location.get("pdf_url"),
        published_date=work.get("publication_date"),
    )


class OpenAlexClient:
    """OpenAlex 唯讀客戶端，使用 cursor 分頁。"""

    def __init__(self, mailto="", timeout=60, retries=5, retry_delay=10, page_size=PAGE_SIZE):
        # 帶 mailto 會進入 OpenAlex 的 polite pool，速率上限較寬鬆。
        contact = f" (mailto:{mailto})" if mailto else ""
        self.headers = {
            "User-Agent": f"daily-paper-bot{contact}",
            "Accept": "application/json",
        }
        self.timeout = timeout
        self.retries = retries
        self.retry_delay = retry_delay
        self.page_size = page_size

    def _get_json(self, params):
        for attempt in range(self.retries):
            try:
                response = requests.get(
                    API_URL, params=params, headers=self.headers, timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
            except Exception as exc:
                if attempt == self.retries - 1:
                    raise OpenAlexError(f"OpenAlex 請求失敗：{exc}") from exc
                logger.warning(f"OpenAlex 請求失敗（{exc}），{self.retry_delay} 秒後重試。")
                sleep(self.retry_delay)
        raise OpenAlexError("OpenAlex 請求失敗。")

    def fetch_works(
        self, source_id, from_date, until_date=None, date_field="publication"
    ) -> Iterator[PaperRecord]:
        """走訪某本期刊在指定日期區間內的論文。

        `date_field` 為 `"publication"`（期號日期）或 `"created"`（進入 OpenAlex 的日期）。
        每日增量務必用 `"created"`：IEEE 的 publication_date 是期號日期，
        整年的論文都標成同一天，用它做增量會抓不到東西。

        缺標題或摘要的紀錄會被略過，因此產出的數量會少於 OpenAlex 回報的總數。
        """
        if date_field not in ("publication", "created"):
            raise ValueError("date_field 必須是 'publication' 或 'created'。")
        filters = [
            f"primary_location.source.id:{source_id}",
            f"from_{date_field}_date:{from_date}",
        ]
        if until_date:
            filters.append(f"to_{date_field}_date:{until_date}")

        cursor = "*"
        seen_cursors = {cursor}
        while cursor:
            payload = self._get_json(
                {
                    "filter": ",".join(filters),
                    "select": SELECT_FIELDS,
                    "per-page": self.page_size,
                    "cursor": cursor,
                }
            )
            results = payload.get("results") or []
            for work in results:
                record = work_to_record(work)
                if record is not None:
                    yield record
            if not results:
                return
            cursor = (payload.get("meta") or {}).get("next_cursor")
            if cursor is None:
                return
            if cursor in seen_cursors:
                # 游標沒有前進代表回應異常，繼續下去會無限迴圈。
                raise OpenAlexError("OpenAlex 的分頁游標沒有前進，已中止。")
            seen_cursors.add(cursor)
