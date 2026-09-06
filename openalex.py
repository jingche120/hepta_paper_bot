"""從 OpenAlex 取得期刊論文。

IEEE TGRS 與 GRSL 不在 arXiv 上，而 IEEE Xplore 的 API 需要訂閱。
OpenAlex 免費、不需金鑰，且這兩本期刊的摘要覆蓋率實測為 100%
（各抽樣 100 篇，平均長度 1,300–1,600 字元）。

摘要在 OpenAlex 是以 inverted index 儲存的，需要還原成文字才能算 embedding。

**注意 IEEE 的日期語意**：這兩本期刊的 `publication_date` 是「期號日期」而非實際
上線日。實測 2026 年的 GRSL 與 TGRS 各抽樣 200 篇，`publication_date` 全部是
`2026-01-01`——粒度其實是「年」。

因此 `from_publication_date:yesterday` 這種每日增量會回傳 0 筆而且不報錯。
而 OpenAlex 唯一能表達「最近新增」的 `from_created_date` 與 `from_updated_date`
**需要付費方案**（免費金鑰收到 HTTP 429 與 `"Plan upgrade required"`，
狀態碼看起來像速率限制，實際不是）。

免費方案下唯一可行的增量作法，是**重抓整個當年度**再靠候選池去重，
起始日由 `publication_window_start()` 決定。代價是每天多幾十次請求，換到的是
不必付費。
"""

from collections.abc import Iterator
from datetime import date, timedelta
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


def year_start(day):
    """把日期對齊到所在年份的 1 月 1 日。

    期號日期的粒度是「年」（2026 年的論文全部標成 2026-01-01），所以任何年中的
    起始日都會把整年排除：`from_publication_date:2026-07-01` 對 2026 年的論文
    回傳 0 筆，而且不報錯。查詢期刊的起始日一律要先過這個函式。
    """
    return date(day.year, 1, 1)


def publication_window_start(today, lookback_days):
    """每日增量要從哪一天開始抓期刊：`today - lookback_days` 所在年份的年初。

    跨年那幾天會自然同時涵蓋前一年，不會有空窗。
    """
    if lookback_days < 0:
        raise ValueError("lookback_days 不得為負數。")
    return year_start(today - timedelta(days=lookback_days)).isoformat()


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

    def fetch_works(self, source_id, from_date, until_date=None) -> Iterator[PaperRecord]:
        """走訪某本期刊在指定發表日區間內的論文。

        只支援 `publication_date`。`created_date` 與 `updated_date` 要付費方案
        （見模組開頭），所以這裡不提供那個選項——留著會讓人以為免費就能用。

        缺標題或摘要的紀錄會被略過，因此產出的數量會少於 OpenAlex 回報的總數。
        """
        filters = [
            f"primary_location.source.id:{source_id}",
            f"from_publication_date:{from_date}",
        ]
        if until_date:
            filters.append(f"to_publication_date:{until_date}")

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
