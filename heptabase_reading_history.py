"""把 Heptabase 白板上的「論文總覽」卡片轉成閱讀紀錄，供推薦系統比對。

白板上每張卡片是一篇讀過的論文總覽，內容依 Markdown 章節組織：

    # 論文總覽：<自訂短代號>
    來源論文：<hepta-mention ...>指向的 PDF 卡片名稱</hepta-mention>
    完整標題：<論文全名>
    ## English Abstract
    ...
    ## 中文摘要
    ...
    ## 關鍵字
    A、B、C。

本模組只做純文字處理與檔案讀寫，不連線，可離線測試。
"""

import html
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# read_object 的輸出首行是 `card "標題" [id] N lines`，其後每行帶 `N\t` 行號。
OBJECT_HEADER = re.compile(r'^card\s+".*"\s+\[[0-9a-fA-F-]{36}\].*$')
LINE_NUMBER_PREFIX = re.compile(r"^\s*\d+\t")
# Heptabase 專用標籤，例如 <hepta-mention type="pdfCard" id="...">標題</hepta-mention>。
# 標籤要移除，但夾在中間的文字要保留。
HEPTA_TAG = re.compile(r"</?hepta-[^>]*>")
HEADING = re.compile(r"^(#{1,6})\s+(?P<name>.*?)\s*$")
SOURCE_PAPER = re.compile(
    r"^(?:來源論文|原始論文|論文出處|source paper)\s*[:：]\s*(?P<title>.+)$", re.I
)
# 卡片裡人工寫下的論文全名。有了這一行，一級標題就能自由寫成短代號
# （`# 論文總覽：PatchTST`），不必為了讓解析器抓對而塞進論文全名。
FULL_TITLE = re.compile(r"^(?:完整標題|完整題目|full title)\s*[:：]\s*(?P<title>.+)$", re.I)
# 章節名稱正規化後的對應；同一種內容可能有多個寫法。
ENGLISH_ABSTRACT_SECTIONS = frozenset({"english abstract", "abstract", "英文摘要"})
CHINESE_ABSTRACT_SECTIONS = frozenset({"中文摘要", "摘要", "chinese abstract"})
KEYWORD_SECTIONS = frozenset({"關鍵字", "關鍵詞", "keywords", "keyword"})
# 關鍵字分隔符：頓號、逗號、分號，中英文都收。
KEYWORD_SEPARATORS = re.compile(r"[、,，;；]")
# 「來源論文」指到的 PDF 卡片有時是以網址或 arXiv API 查詢字串命名，而不是論文標題。
# 這種值不能當標題，要退回卡片自己的一級標題。
NOT_A_TITLE = re.compile(r"https?://|search_query=|id_list=|max_results=", re.I)

ORDER_EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)

PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_path(path):
    """相對路徑一律以專案根目錄為基準，避免匯出與讀取因工作目錄不同而指到不同檔案。"""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


class ReadingHistoryError(ValueError):
    """卡片內容無法解析成閱讀紀錄。"""


@dataclass
class CardRecord:
    title: str
    abstract: str
    order: int
    paths: list[str] = field(default_factory=list)
    card_id: str | None = None
    created_at: str | None = None


@dataclass
class ReadingRecord:
    """對應 zotero_arxiv_daily.protocol.CorpusPaper 的欄位，但不依賴該模組。

    這樣匯出流程與解析測試都不必安裝 tiktoken、openai 等推薦流程才需要的套件。
    """

    title: str
    abstract: str
    added_date: datetime
    paths: list[str]


def clean_card_text(text):
    """去掉 read_object 的標頭列、行號前綴與 Heptabase 專用標籤。"""
    lines = []
    for index, line in enumerate(text.splitlines()):
        if index == 0 and OBJECT_HEADER.match(line.strip()):
            continue
        lines.append(LINE_NUMBER_PREFIX.sub("", line))
    # MCP 會把內容做 HTML 跳脫（有時還跳脫兩層），還原後才是原始文字。
    text = HEPTA_TAG.sub("", "\n".join(lines))
    for _ in range(2):
        unescaped = html.unescape(text)
        if unescaped == text:
            break
        text = unescaped
    return text


def split_sections(text):
    """依 Markdown 標題切分；回傳 (前言, {正規化章節名: 內容})。"""
    preamble = []
    sections = {}
    current = None
    body = []
    for line in text.splitlines():
        heading = HEADING.match(line)
        if heading is None:
            (body if current is not None else preamble).append(line)
            continue
        if current is not None:
            sections[current] = "\n".join(body).strip()
        current = heading.group("name").strip().lower()
        body = []
    if current is not None:
        sections[current] = "\n".join(body).strip()
    return "\n".join(preamble).strip(), sections


def _first_section(sections, names):
    for name in names:
        if sections.get(name):
            return sections[name]
    return ""


def _strip_emphasis(text):
    """去掉整段包住標題的 Markdown 強調符號；標題內部的星號不受影響。"""
    return text.strip().strip("*_").strip()


def _title_candidate(pattern, line):
    """取出欄位值當標題；值不像標題時回傳空字串，交給呼叫端往下退。"""
    match = pattern.match(line)
    if match is None:
        return ""
    candidate = _strip_emphasis(match.group("title"))
    # PDF 卡片若以網址或 arXiv 查詢字串命名，這裡拿到的不是論文標題。
    # `完整標題：` 也一樣要擋：那行貼成網址時，寧可往下退，也不要拿網址當標題。
    return "" if NOT_A_TITLE.search(candidate) else candidate


def _extract_title(cleaned, fallback_title):
    """依序取「完整標題：」、「來源論文：」、卡片一級標題，最後才用 list_cards 的標題。

    `完整標題：` 是卡片裡人工寫下的論文全名，最可信，所以排在最前面。
    這些欄位通常寫在一級標題底下，會被歸進該章節的內文而非前言，
    因此這裡掃描整份文字而不是只看前言。

    一級標題刻意不做 `_strip_emphasis`：它的強調符號在 `論文總覽：` 之後
    （`# 論文總覽：*A...*`），只去頭尾會變成不對稱的 `論文總覽：*A...`，比不動更糟。
    """
    full_title = ""
    source_title = ""
    first_heading = ""
    for line in cleaned.splitlines():
        stripped = line.strip()
        if not full_title:
            full_title = _title_candidate(FULL_TITLE, stripped)
        if not source_title:
            source_title = _title_candidate(SOURCE_PAPER, stripped)
        if not first_heading and stripped.startswith("# "):
            first_heading = stripped[2:].strip()
    return full_title or source_title or first_heading or fallback_title


def _extract_keywords(sections):
    body = _first_section(sections, KEYWORD_SECTIONS)
    if not body:
        return []
    # 只取第一段，避免把後面的說明文字也當成關鍵字。
    body = body.split("\n\n")[0].strip().rstrip("。.")
    keywords = [part.strip() for part in KEYWORD_SEPARATORS.split(body)]
    return [k for k in keywords if k]


def parse_overview_card(text, fallback_title="", card_id=None, created_at=None, order=0):
    """把一張論文總覽卡片解析成語料紀錄。

    摘要優先用「English Abstract + 中文摘要」——排序階段只比對 abstract，
    英文原文與 arXiv 摘要同語言，訊號最強，中文摘要則補上你關注的角度。
    兩個章節都沒有時，退回整張卡片內文，並由呼叫端提示。
    """
    cleaned = clean_card_text(text)
    _, sections = split_sections(cleaned)

    parts = [
        _first_section(sections, ENGLISH_ABSTRACT_SECTIONS),
        _first_section(sections, CHINESE_ABSTRACT_SECTIONS),
    ]
    abstract = "\n\n".join(part for part in parts if part).strip()
    used_fallback = False
    if not abstract:
        # 沒有摘要章節就用整張卡片，總比讓這篇論文從語料中消失好。
        abstract = cleaned.strip()
        used_fallback = True

    title = _extract_title(cleaned, fallback_title)
    if not title or not abstract:
        raise ReadingHistoryError(f"卡片 {card_id or fallback_title} 缺少標題或內容。")

    paper = CardRecord(
        title=title,
        abstract=abstract,
        order=order,
        paths=_extract_keywords(sections),
        card_id=card_id,
        created_at=created_at,
    )
    return paper, used_fallback


def dump_reading_history(papers, path, whiteboard_id):
    """把解析結果寫成 JSON 快取。"""
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "whiteboard_id": whiteboard_id,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "papers": [asdict(paper) for paper in papers],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_created_at(value, order):
    """卡片建立時間；缺少或格式不正確時，退回以順序推導的相對日期。"""
    if value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
    return ORDER_EPOCH.replace(year=ORDER_EPOCH.year + order // 365)


def load_reading_history(path):
    """讀取 JSON 快取，回傳閱讀紀錄。"""
    path = resolve_path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        papers = payload["papers"]
    except FileNotFoundError:
        # 用 sys.executable 而非寫死 .venv/bin/python：Windows 的 venv 版面不同。
        raise ReadingHistoryError(
            f"找不到閱讀紀錄 {path}。請先執行："
            f"\n    {sys.executable} get_hepthabase_bot.py --export {path}"
        ) from None
    except (ValueError, KeyError, TypeError):
        raise ReadingHistoryError(f"閱讀紀錄 {path} 格式不正確，請重新匯出。") from None

    return [
        ReadingRecord(
            title=paper["title"],
            abstract=paper["abstract"],
            added_date=parse_created_at(paper.get("created_at"), paper.get("order", index)),
            paths=paper.get("paths", []),
        )
        for index, paper in enumerate(papers)
    ]
