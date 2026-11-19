import tempfile
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

from get_hepthabase_bot import (
    WhiteboardConfigError,
    connect,
    load_whiteboard_id,
    report_error,
)
from heptabase_oauth import AuthorizationRequiredError, BrowserCallback


class MemoryStorage:
    """僅供離線探索測試使用，不接觸鑰匙圈。"""

    def __init__(self):
        self.tokens = None
        self.client_info = None

    async def get_tokens(self):
        return self.tokens

    async def set_tokens(self, tokens):
        self.tokens = tokens

    async def get_client_info(self):
        return self.client_info

    async def set_client_info(self, client_info):
        self.client_info = client_info


class ConnectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_connection_reads_and_prints_fixed_card(self):
        session = AsyncMock()
        session.__aenter__.return_value = session
        auth = httpx.Auth()

        @asynccontextmanager
        async def oauth(*args, **kwargs):
            yield auth

        @asynccontextmanager
        async def transport(*args, **kwargs):
            self.assertIs(kwargs["http_client"].auth, auth)
            self.assertTrue(kwargs["http_client"].follow_redirects)
            yield None, None, None

        with (
            patch("get_hepthabase_bot.oauth_connection", oauth),
            patch("get_hepthabase_bot.streamable_http_client", transport),
            patch("get_hepthabase_bot.ClientSession", return_value=session),
            patch(
                "get_hepthabase_bot.list_whiteboard_cards",
                new_callable=AsyncMock,
                return_value=[{"card_id": "id", "title": "某篇論文", "created_at": "2026-09-05"}],
            ) as listed,
            patch("builtins.print") as output,
        ):
            await connect("board-under-test")
        session.initialize.assert_awaited_once()
        session.list_tools.assert_not_called()
        session.call_tool.assert_not_called()
        # 不加 --export 時只列出標題，不讀取也不輸出卡片內文。
        listed.assert_awaited_once_with(session, "board-under-test")
        printed = [call.args[0] for call in output.call_args_list]
        self.assertEqual(printed[0], "Heptabase MCP 連線成功。")
        self.assertTrue(any("某篇論文" in line for line in printed))


class WhiteboardConfigTests(unittest.TestCase):
    """白板 ID 不寫在原始碼裡，改由不進版控的 config/custom.yaml 提供。"""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.config_dir = Path(self.dir.name)
        (self.config_dir / "base.yaml").write_text(
            "heptabase:\n  whiteboard_id: null\n", encoding="utf-8"
        )

    def tearDown(self):
        self.dir.cleanup()

    def write_custom(self, text):
        (self.config_dir / "custom.yaml").write_text(text, encoding="utf-8")

    def test_reads_whiteboard_from_custom_config(self):
        self.write_custom("heptabase:\n  whiteboard_id: ' board-from-config '\n")
        self.assertEqual(load_whiteboard_id(config_dir=self.config_dir), "board-from-config")

    def test_command_line_overrides_config(self):
        self.write_custom("heptabase:\n  whiteboard_id: board-from-config\n")
        self.assertEqual(
            load_whiteboard_id("board-from-cli", config_dir=self.config_dir), "board-from-cli"
        )

    def test_missing_custom_config_is_an_error_not_a_silent_default(self):
        with self.assertRaises(WhiteboardConfigError):
            load_whiteboard_id(config_dir=self.config_dir)

    def test_blank_whiteboard_is_an_error(self):
        self.write_custom("heptabase:\n  whiteboard_id: '  '\n")
        with self.assertRaises(WhiteboardConfigError):
            load_whiteboard_id(config_dir=self.config_dir)

    def test_unresolved_env_placeholders_elsewhere_are_not_touched(self):
        # 其他鍵的 ${oc.env:...} 不應被解析；匯出流程不該依賴 .env。
        self.write_custom(
            "heptabase:\n  whiteboard_id: board\nemail:\n  sender: ${oc.env:NOT_SET_ANYWHERE}\n"
        )
        self.assertEqual(load_whiteboard_id(config_dir=self.config_dir), "board")


class CallbackTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.callback = BrowserCallback()
        self.callback.state = "known"

    async def test_browser_opens_authorization_url(self):
        with (
            patch("heptabase_oauth.webbrowser.open", return_value=True) as browser,
            patch("builtins.print"),
        ):
            await self.callback.open_browser("https://example.com/authorize?state=new")
        browser.assert_called_once_with("https://example.com/authorize?state=new")
        self.assertEqual(self.callback.state, "new")

    async def test_no_browser_fails_before_opening_or_printing_authorization_url(self):
        callback = BrowserCallback(allow_browser=False)
        with patch("heptabase_oauth.webbrowser.open") as browser, patch("builtins.print") as output:
            with self.assertRaises(AuthorizationRequiredError):
                await callback.open_browser("https://example.com/authorize?state=private")
        browser.assert_not_called()
        output.assert_not_called()

    async def test_valid_callback_returns_code_and_state(self):
        self.assertTrue(self.callback.accept_callback("/callback?code=example&state=known"))
        self.assertEqual(await self.callback.wait_for_callback(), ("example", "known"))
        self.assertFalse(self.callback.accept_callback("/callback?code=example&state=known"))

    async def test_invalid_callback_keeps_waiting(self):
        for target in (
            "/wrong?code=x&state=known",
            "/callback?code=x",
            "/callback?code=x&state=wrong",
            "/callback?state=known",
            "/callback?code=x&state=known&state=known",
        ):
            with self.subTest(target=target):
                self.assertFalse(self.callback.accept_callback(target))
                self.assertFalse(self.callback.result.done())

    async def test_denial_stops_authorization(self):
        self.assertFalse(self.callback.accept_callback("/callback?error=access_denied&state=known"))
        with self.assertRaises(RuntimeError):
            await self.callback.wait_for_callback()

    async def test_timeout(self):
        with patch("heptabase_oauth.asyncio.wait_for", side_effect=TimeoutError):
            with self.assertRaises(TimeoutError):
                await self.callback.wait_for_callback()

    async def test_http_callback_response_does_not_expose_code(self):
        reader = AsyncMock()
        reader.readline.return_value = b"GET /callback?code=private-code&state=known HTTP/1.1\r\n"
        writer = AsyncMock()
        from unittest.mock import Mock

        writer.write = Mock()
        writer.close = Mock()
        await self.callback.handle_request(reader, writer)
        response = writer.write.call_args.args[0]
        self.assertIn(b"200 OK", response)
        self.assertNotIn(b"private-code", response)
        writer.close.assert_called_once()

    async def test_storage_is_not_shared_between_runs(self):
        storage = MemoryStorage()
        await storage.set_tokens("test-token")
        self.assertEqual(await storage.get_tokens(), "test-token")
        self.assertIsNone(await MemoryStorage().get_tokens())


class OAuthDiscoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_redirected_discovery_reaches_browser_handler(self):
        from mcp.client.auth import OAuthClientProvider
        from mcp.shared.auth import OAuthClientMetadata
        from pydantic import AnyUrl

        class BrowserReached(Exception):
            pass

        paths = []

        def respond(request):
            paths.append(request.url.path)
            if request.url.path == "/mcp":
                return httpx.Response(
                    401,
                    headers={
                        "WWW-Authenticate": 'Bearer resource_metadata="https://example.com/resource", scope="space:read"'
                    },
                )
            if request.url.path == "/resource":
                return httpx.Response(
                    200,
                    json={
                        "resource": "https://example.com/mcp",
                        "authorization_servers": ["https://example.com"],
                    },
                )
            if request.url.path == "/.well-known/oauth-authorization-server":
                return httpx.Response(
                    302, headers={"Location": "/.well-known/openid-configuration"}
                )
            if request.url.path == "/.well-known/openid-configuration":
                return httpx.Response(
                    200,
                    json={
                        "issuer": "https://example.com",
                        "authorization_endpoint": "https://example.com/auth",
                        "token_endpoint": "https://example.com/token",
                        "registration_endpoint": "https://example.com/v1/oauth/register",
                        "response_types_supported": ["code"],
                        "code_challenge_methods_supported": ["S256"],
                    },
                )
            if request.url.path == "/v1/oauth/register":
                return httpx.Response(
                    201,
                    json={
                        "client_id": "test-client",
                        "redirect_uris": ["http://127.0.0.1:8765/callback"],
                        "token_endpoint_auth_method": "none",
                    },
                )
            raise AssertionError(f"Unexpected path: {request.url.path}")

        browser = AsyncMock(side_effect=BrowserReached)
        auth = OAuthClientProvider(
            server_url="https://example.com/mcp",
            storage=MemoryStorage(),
            client_metadata=OAuthClientMetadata(
                redirect_uris=[AnyUrl("http://127.0.0.1:8765/callback")],
                token_endpoint_auth_method="none",
            ),
            redirect_handler=browser,
            callback_handler=AsyncMock(),
        )
        with patch("logging.Logger.exception"):
            async with httpx.AsyncClient(
                auth=auth, follow_redirects=True, transport=httpx.MockTransport(respond)
            ) as client:
                with self.assertRaises(BrowserReached):
                    await client.post("https://example.com/mcp")
        browser.assert_awaited_once()
        self.assertIn("/.well-known/openid-configuration", paths)
        self.assertIn("/v1/oauth/register", paths)
        self.assertTrue(browser.call_args.args[0].startswith("https://example.com/auth?"))

    async def test_error_reporting_hides_private_details(self):
        error = ExceptionGroup("private-group", [RuntimeError("private-token")])
        with patch("builtins.print") as output:
            report_error(error)
        self.assertIn("RuntimeError", output.call_args.args[0])
        self.assertNotIn("private", output.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
