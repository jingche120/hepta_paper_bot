"""唯讀卡片工具；顯示 MCP 傳回的原始內容，不經 LLM 改寫。"""

import json
import re


class CardReadError(RuntimeError):
    """讀取失敗，不將遠端私密錯誤內容帶入日誌。"""


async def call_read_tool(session, name, arguments):
    result = await session.call_tool(name, arguments)
    if result.isError:
        raise CardReadError("MCP 工具回報錯誤。")
    payload = result.structuredContent
    if payload is None:
        try:
            payload = json.loads(
                "\n".join(block.text for block in result.content if block.type == "text")
            )
        except (ValueError, TypeError):
            raise CardReadError("MCP 工具回應格式不正確。") from None
    if (
        not isinstance(payload, dict)
        or payload.get("status") != "succeeded"
        or not isinstance(payload.get("content"), str)
    ):
        raise CardReadError("無法讀取卡片，請確認 ID、類型與權限。")
    return payload


CARD_LINE = re.compile(r'^card\s+"(?P<title>.*)"\s+\[(?P<card_id>[0-9a-fA-F-]{36})\](?P<rest>.*)$')
CREATED_AT = re.compile(r"created:\s*(?P<created>\S+?)(?:;|\s|$)")


def parse_card_lines(content):
    """從 list_cards 的文字回應取出 (card_id, title, created_at)。

    list_cards 回傳的是人類可讀文字而非結構化 JSON，因此需要逐行比對。
    """
    cards = []
    for line in content.splitlines():
        match = CARD_LINE.match(line.strip())
        if match is None:
            continue
        created = CREATED_AT.search(match.group("rest"))
        cards.append(
            {
                "card_id": match.group("card_id"),
                "title": match.group("title"),
                "created_at": created.group("created") if created else None,
            }
        )
    return cards


async def list_whiteboard_cards(session, whiteboard_id, page_size=100):
    """列出白板上的所有卡片，依建立時間由舊到新。"""
    offset = 0
    cards = []
    while True:
        payload = await call_read_tool(
            session,
            "list_cards",
            {
                "whiteboardIds": [whiteboard_id],
                "objectTypes": ["card"],
                "include": ["timestamps"],
                "sortBy": "createdTime",
                "sortDirection": "ascending",
                "offset": offset,
                "limit": page_size,
            },
        )
        page = parse_card_lines(payload["content"])
        cards.extend(page)
        # list_cards 不回報 hasMore，改以「這頁是否填滿」判斷還有沒有下一頁。
        if len(page) < page_size:
            return cards
        if len(cards) >= 10000:
            raise CardReadError("白板卡片數量異常，請確認白板 ID。")
        offset += page_size


async def search_cards(session, title):
    offset = 0
    pages = []
    while True:
        payload = await call_read_tool(
            session,
            "search_by_keyword",
            {
                "query": title,
                "objectTypes": ["card"],
                "contextLines": 0,
                "offset": offset,
                "limit": 20,
            },
        )
        pages.append(payload["content"])
        if payload.get("hasMore") is False:
            return "\n\n".join(pages)
        if payload.get("hasMore") is not True or len(pages) >= 100:
            raise CardReadError("搜尋未完整完成，請縮小關鍵字範圍。")
        offset += 20


async def read_card(session, card_id):
    offset = 0
    pages = []
    while True:
        payload = await call_read_tool(
            session,
            "read_object",
            {
                "objectId": card_id,
                "objectType": "card",
                "offset": offset,
                "limit": 200,
            },
        )
        pages.append(payload["content"])
        if payload.get("hasMore") is not True:
            return "\n\n".join(pages)
        if not payload["content"] or len(pages) >= 100:
            raise CardReadError("卡片未完整讀取，請分段縮小內容。")
        offset += 200
