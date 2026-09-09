"""OAuth 輔助元件：不讀取 .env，授權存入作業系統的憑證儲存。"""

import asyncio
import secrets
import webbrowser
from contextlib import asynccontextmanager
from urllib.parse import parse_qs, urlsplit

from mcp.shared.auth import OAuthClientMetadata
from pydantic import AnyUrl

from heptabase_keychain import KeychainStorage, PersistentOAuthClientProvider

CALLBACK_URL = "http://127.0.0.1:8765/callback"


class AuthorizationRequiredError(RuntimeError):
    """禁止互動時，明確指出需要重新授權。"""


class BrowserCallback:
    def __init__(self, allow_browser=True):
        self.result = asyncio.get_running_loop().create_future()
        self.state = None
        self.allow_browser = allow_browser

    async def open_browser(self, url):
        if not self.allow_browser:
            raise AuthorizationRequiredError("需要互動授權。")
        # state 是本次登入的隨機識別值，用來確認回呼屬於這次授權。
        self.state = parse_qs(urlsplit(url).query)["state"][0]
        print("請在瀏覽器登入 Heptabase 並允許授權。")
        if not await asyncio.to_thread(webbrowser.open, url):
            print("無法自動開啟瀏覽器，請在本機瀏覽器開啟此授權網址：")
            print(url)

    async def wait_for_callback(self):
        # 瀏覽器回呼抵達時，handle_request 會填入 result。
        return await asyncio.wait_for(asyncio.shield(self.result), timeout=300)

    def accept_callback(self, target):
        parsed = urlsplit(target)
        params = parse_qs(parsed.query)
        states = params.get("state", [])
        if parsed.path != "/callback" or self.result.done():
            return False
        if not self.state or len(states) != 1 or not secrets.compare_digest(states[0], self.state):
            return False
        if "error" in params:
            self.result.set_exception(RuntimeError("使用者未完成授權"))
            return False
        codes = params.get("code", [])
        if len(codes) != 1:
            return False
        # SDK 會再次驗證 state，並用授權碼換取存取權杖。
        self.result.set_result((codes[0], states[0]))
        return True

    async def handle_request(self, reader, writer):
        """僅接收本機瀏覽器的 OAuth 回呼，不記錄網址或授權碼。"""
        accepted = False
        try:
            line = await asyncio.wait_for(reader.readline(), timeout=5)
            method, target, _ = line.decode("ascii").strip().split(" ")
            if method == "GET":
                accepted = self.accept_callback(target)
        except (ValueError, TimeoutError, ConnectionError):
            pass
        status = "200 OK" if accepted else "400 Bad Request"
        body = (
            b"Response received. Return to the terminal to check the connection."
            if accepted
            else b"Invalid or denied authorization. Return to the terminal."
        )
        try:
            writer.write(
                (
                    f"HTTP/1.1 {status}\r\nContent-Type: text/plain\r\n"
                    f"Content-Length: {len(body)}\r\nCache-Control: no-store\r\n"
                    "Connection: close\r\n\r\n"
                ).encode("ascii")
                + body
            )
            await writer.drain()
        except ConnectionError:
            pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionError:
                pass


@asynccontextmanager
async def oauth_connection(server_url, *, allow_browser=True):
    """先準備回呼接收服務，再把 OAuth 元件交給 HTTP 用戶端。"""
    storage = KeychainStorage(server_url)
    callback = BrowserCallback(allow_browser=allow_browser)
    server = await asyncio.start_server(callback.handle_request, "127.0.0.1", 8765, limit=16384)
    async with server:
        yield PersistentOAuthClientProvider(
            server_url=server_url,
            client_metadata=OAuthClientMetadata(
                client_name="Heptabase MCP Client",
                redirect_uris=[AnyUrl(CALLBACK_URL)],
                grant_types=["authorization_code", "refresh_token"],
                response_types=["code"],
                token_endpoint_auth_method="none",
            ),
            storage=storage,
            redirect_handler=callback.open_browser,
            callback_handler=callback.wait_for_callback,
        )
