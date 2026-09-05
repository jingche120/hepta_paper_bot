"""連線 Heptabase 並讀取白板「讀過的論文總覽們」上的論文總覽卡片。"""

import argparse
import asyncio
import logging
from pathlib import Path

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from heptabase_cards import CardReadError, list_whiteboard_cards, read_card
from heptabase_keychain import KeychainStorageError
from heptabase_oauth import AuthorizationRequiredError, oauth_connection
from heptabase_reading_history import (
    ReadingHistoryError,
    dump_reading_history,
    parse_overview_card,
    resolve_path,
)

MCP_URL = "https://api.heptabase.com/mcp"
# 白板「讀過的論文總覽們」，每張卡片是一篇讀過的論文總覽。
WHITEBOARD_ID = "00000000-0000-4000-8000-000000000000"
DEFAULT_EXPORT_PATH = Path("data/reading_history.json")


async def export_whiteboard(session, export_path):
    """讀取白板上每張卡片，解析成閱讀紀錄，並保留原始全文供人工核對。"""
    export_path = resolve_path(export_path)
    raw_path = export_path.with_name("heptabase_cards_raw.txt")
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    listed = await list_whiteboard_cards(session, WHITEBOARD_ID)
    if not listed:
        raise ReadingHistoryError(f"白板 {WHITEBOARD_ID} 沒有卡片，請確認白板 ID 與卡片位置。")
    print(f"白板上有 {len(listed)} 張卡片，開始逐張讀取…")

    papers = []
    raw_blocks = []
    fallbacks = []
    for order, card in enumerate(listed):
        content = await read_card(session, card["card_id"])
        raw_blocks.append(content)
        paper, used_fallback = parse_overview_card(
            content,
            fallback_title=card["title"],
            card_id=card["card_id"],
            created_at=card["created_at"],
            order=order,
        )
        papers.append(paper)
        if used_fallback:
            fallbacks.append(paper.title)
        print(
            f"  [{order + 1}/{len(listed)}] {paper.title[:50]}"
            f"｜摘要 {len(paper.abstract)} 字｜關鍵字 {len(paper.paths)} 個"
        )

    raw_path.write_text("\n\n".join(raw_blocks), encoding="utf-8")
    dump_reading_history(papers, export_path, WHITEBOARD_ID)
    print(f"\n已匯出 {len(papers)} 篇論文到 {export_path}。")
    print(f"原始卡片全文已存到 {raw_path}。")
    if fallbacks:
        print("\n以下卡片找不到「English Abstract」或「中文摘要」章節，已改用整張卡片內文：")
        for title in fallbacks:
            print(f"  - {title}")
        print("補上摘要章節可讓相似度比對更準確。")


async def connect(*, no_browser=False, export_path=None):
    # 1. 準備 OAuth：需要登入時，自動開啟瀏覽器並接收授權回呼。
    async with oauth_connection(MCP_URL, allow_browser=not no_browser) as oauth:
        # 2. HTTP 用戶端交由 OAuth 元件處理授權及存取權杖。
        # Heptabase 的 OAuth 探索網址會回傳 302，必須跟隨轉址。
        async with httpx.AsyncClient(auth=oauth, timeout=60, follow_redirects=True) as client:
            # 3. 以 HTTP 建立 MCP 的接收（read）與傳送（write）通道。
            async with streamable_http_client(MCP_URL, http_client=client) as (read, write, _):
                # 4. 建立工作階段並完成 MCP 握手，成功後才印出訊息。
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    print("Heptabase MCP 連線成功。")
                    if export_path is None:
                        # 不加 --export 時只列出卡片標題，作為連線檢查，不輸出卡片內文。
                        listed = await list_whiteboard_cards(session, WHITEBOARD_ID)
                        print(f"白板「讀過的論文總覽們」上有 {len(listed)} 張卡片：")
                        for card in listed:
                            print(f"  - {card['title']}（建立於 {card['created_at']}）")
                        print("\n加上 --export 可解析成閱讀紀錄。")
                    else:
                        await export_whiteboard(session, export_path)
                    if no_browser:
                        print("免 OAuth 互動驗證成功：本次未開啟瀏覽器授權。")
    # 離開 async with 後，自動關閉連線與本機回呼服務。


def report_error(error):
    """展開 SDK 的群組例外，只顯示類型及狀態碼，不印私密回應。"""
    if isinstance(error, BaseExceptionGroup):
        for child in error.exceptions:
            report_error(child)
    elif isinstance(error, AuthorizationRequiredError):
        print("需要重新 OAuth 授權；本次未開啟瀏覽器。請移除 --no-browser 後執行。")
    elif isinstance(error, CardReadError):
        print("卡片讀取失敗或不完整，請確認卡片 ID、類型、權限與連線。")
    elif isinstance(error, ReadingHistoryError):
        print(f"卡片解析失敗：{error}")
    elif isinstance(error, KeychainStorageError):
        print(
            "鑰匙圈存取失敗。請確認 macOS 鑰匙圈已解鎖、允許此 Python 存取，且已安裝 requirements.txt。"
        )
    elif isinstance(error, httpx.HTTPStatusError):
        print(f"HTTP 請求失敗：{error.response.status_code}。")
    elif isinstance(error, (TimeoutError, httpx.TimeoutException)):
        print("連線或授權等候逾時，請重新執行。")
    elif isinstance(error, OSError):
        print(f"本機或檔案存取錯誤（errno={error.errno}）。連線失敗請檢查 8765 埠是否被占用；")
        print("匯出失敗請確認 --export 指定的路徑可寫入。")
    else:
        print(f"連線失敗（{type(error).__name__}），請將這則訊息提供給開發者。")


def main():
    parser = argparse.ArgumentParser(description="讀取 Heptabase 白板上的論文總覽卡片")
    parser.add_argument("--no-browser", action="store_true", help="禁止開啟 OAuth；需要授權即失敗")
    parser.add_argument(
        "--export",
        nargs="?",
        const=DEFAULT_EXPORT_PATH,
        type=Path,
        metavar="PATH",
        help=f"解析卡片並匯出閱讀紀錄，預設 {DEFAULT_EXPORT_PATH}；不指定則只列出卡片標題",
    )
    args = parser.parse_args()
    # SDK 錯誤日誌可能含授權網址，因此不直接印出完整例外或日誌。
    logging.disable(logging.CRITICAL)
    try:
        asyncio.run(connect(no_browser=args.no_browser, export_path=args.export))
    except KeyboardInterrupt:
        print("已取消連線。")
        return 130
    except Exception as error:
        report_error(error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
