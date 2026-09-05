import unittest
from unittest.mock import AsyncMock

from mcp.types import CallToolResult, TextContent

from heptabase_cards import (
    CardReadError,
    list_whiteboard_cards,
    parse_card_lines,
    read_card,
    search_cards,
)


def result(content, **fields):
    return CallToolResult(
        content=[], structuredContent={"status": "succeeded", "content": content, **fields}
    )


class CardTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_reads_all_pages_and_does_not_read_arbitrary_candidate(self):
        session = AsyncMock()
        session.call_tool.side_effect = [
            result("first", hasMore=True),
            result("last", hasMore=False),
        ]
        self.assertEqual(await search_cards(session, "exact title"), "first\n\nlast")
        calls = session.call_tool.call_args_list
        self.assertEqual([call.args[1]["offset"] for call in calls], [0, 20])
        self.assertTrue(all(call.args[0] == "search_by_keyword" for call in calls))

    async def test_read_uses_card_id_and_reads_all_pages(self):
        session = AsyncMock()
        session.call_tool.side_effect = [
            result("first", hasMore=True),
            result("last", hasMore=False),
        ]
        self.assertEqual(await read_card(session, "known-id"), "first\n\nlast")
        calls = session.call_tool.call_args_list
        self.assertEqual([call.args[1]["offset"] for call in calls], [0, 200])
        self.assertTrue(all(call.args[1]["objectId"] == "known-id" for call in calls))

    async def test_failed_tool_does_not_return_error_as_card_content(self):
        session = AsyncMock()
        session.call_tool.return_value = CallToolResult(
            content=[], structuredContent={"status": "failed", "content": "private-error"}
        )
        with self.assertRaises(CardReadError) as error:
            await read_card(session, "known-id")
        self.assertNotIn("private-error", str(error.exception))

    async def test_text_json_fallback(self):
        session = AsyncMock()
        session.call_tool.return_value = CallToolResult(
            content=[
                TextContent(
                    type="text", text='{"status":"succeeded","content":"card body","hasMore":false}'
                )
            ]
        )
        self.assertEqual(await read_card(session, "known-id"), "card body")

    async def test_incomplete_search_fails(self):
        session = AsyncMock()
        session.call_tool.return_value = result("partial")
        with self.assertRaises(CardReadError):
            await search_cards(session, "title")

    async def test_empty_continuation_fails_instead_of_looping(self):
        session = AsyncMock()
        session.call_tool.return_value = result("", hasMore=True)
        with self.assertRaises(CardReadError):
            await read_card(session, "known-id")


LISTED = (
    "Cards:\n"
    'card "論文總覽：SC-GNN 地震烈度即時預測" [11111111-1111-4111-8111-111111111111] '
    "created: 2026-09-05T17:49:45.787Z; updated: 2026-09-05T17:49:45.787Z"
)


class ListCardsTests(unittest.IsolatedAsyncioTestCase):
    def test_parses_title_id_and_created_at(self):
        cards = parse_card_lines(LISTED)
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0]["title"], "論文總覽：SC-GNN 地震烈度即時預測")
        self.assertEqual(cards[0]["card_id"], "11111111-1111-4111-8111-111111111111")
        self.assertEqual(cards[0]["created_at"], "2026-09-05T17:49:45.787Z")

    def test_ignores_the_heading_line(self):
        self.assertEqual(parse_card_lines("Cards:\nnot a card line"), [])

    def test_missing_timestamps_do_not_break_parsing(self):
        cards = parse_card_lines('card "T" [11111111-1111-4111-8111-111111111111]')
        self.assertIsNone(cards[0]["created_at"])

    async def test_pagination_stops_on_short_page(self):
        session = AsyncMock()
        session.call_tool.side_effect = [result(LISTED), result("Cards:")]
        cards = await list_whiteboard_cards(session, "wb", page_size=1)
        self.assertEqual(len(cards), 1)
        offsets = [call.args[1]["offset"] for call in session.call_tool.call_args_list]
        self.assertEqual(offsets, [0, 1])

    async def test_requests_cards_sorted_oldest_first(self):
        session = AsyncMock()
        session.call_tool.return_value = result("Cards:")
        await list_whiteboard_cards(session, "wb")
        args = session.call_tool.call_args.args[1]
        self.assertEqual(args["whiteboardIds"], ["wb"])
        self.assertEqual(args["sortBy"], "createdTime")
        self.assertEqual(args["sortDirection"], "ascending")
        self.assertIn("timestamps", args["include"])


if __name__ == "__main__":
    unittest.main()
