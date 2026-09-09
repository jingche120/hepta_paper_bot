import sys
import time
import unittest
from unittest.mock import AsyncMock, Mock, patch
from urllib.parse import parse_qs

import httpx
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthMetadata

from get_hepthabase_bot import report_error
from heptabase_keychain import (
    SERVICE,
    KeychainStorage,
    KeychainStorageError,
    PersistentOAuthClientProvider,
    _load_backend,
)

URL = "https://example.com/mcp"
CALLBACK = "http://127.0.0.1:8765/callback"


class KeychainTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.values = {}
        self.backend = Mock()
        self.backend.get_password.side_effect = lambda service, account: self.values.get(
            (service, account)
        )
        self.backend.set_password.side_effect = lambda service, account, value: self.values.update(
            {(service, account): value}
        )
        # 取代整個後端解析，這些測試才能在 macOS 與 Windows 上跑同一份程式碼：
        # 直接 patch keyring.backends.macOS 會在 Windows 上因為模組無法匯入而失敗。
        factory = patch("heptabase_keychain._load_backend", return_value=self.backend)
        factory.start()
        self.addCleanup(factory.stop)

    def provider(self, url=URL):
        return PersistentOAuthClientProvider(
            server_url=url,
            storage=KeychainStorage(url),
            client_metadata=OAuthClientMetadata(
                redirect_uris=[CALLBACK], token_endpoint_auth_method="none"
            ),
            redirect_handler=AsyncMock(side_effect=AssertionError("Unexpected browser")),
            callback_handler=AsyncMock(),
        )

    async def seed(self, expires_in=3600):
        auth = self.provider()
        auth.context.oauth_metadata = OAuthMetadata(
            issuer="https://auth.example.com",
            authorization_endpoint="https://auth.example.com/authorize",
            token_endpoint="https://auth.example.com/custom/token",
            response_types_supported=["code"],
        )
        auth.context.auth_server_url = "https://auth.example.com"
        await auth.context.storage.set_client_info(
            OAuthClientInformationFull(
                client_id="fake-client", redirect_uris=[CALLBACK], token_endpoint_auth_method="none"
            )
        )
        await auth._handle_token_response(
            httpx.Response(
                200,
                json={
                    "access_token": "fake-access",
                    "refresh_token": "fake-refresh",
                    "token_type": "Bearer",
                    "expires_in": expires_in,
                },
            )
        )
        return auth

    async def test_empty_storage_and_server_isolation(self):
        self.assertIsNone(await KeychainStorage(URL).get_tokens())
        await self.seed()
        self.assertIsNone(await KeychainStorage("https://other.example/mcp").get_tokens())

    async def test_restart_reuses_tokens_without_browser_or_registration(self):
        previous = await self.seed()
        restarted = self.provider()
        requests = []

        def respond(request):
            requests.append(request)
            self.assertEqual(request.headers.get("Authorization"), "Bearer fake-access")
            return httpx.Response(200, json={"ok": True})

        async with httpx.AsyncClient(
            auth=restarted, transport=httpx.MockTransport(respond)
        ) as client:
            response = await client.get(URL)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(requests), 1)
        self.assertEqual(restarted.context.client_info.client_id, "fake-client")
        self.assertEqual(restarted.context.token_expiry_time, previous.context.token_expiry_time)
        restarted.context.redirect_handler.assert_not_called()

    async def test_expired_restart_refreshes_at_saved_endpoint_and_persists_rotation(self):
        await self.seed(expires_in=-60)
        restarted = self.provider()
        paths = []

        def respond(request):
            paths.append(str(request.url))
            if str(request.url) == "https://auth.example.com/custom/token":
                form = parse_qs(request.content.decode())
                self.assertEqual(form["grant_type"], ["refresh_token"])
                self.assertEqual(form["refresh_token"], ["fake-refresh"])
                return httpx.Response(
                    200,
                    json={
                        "access_token": "rotated-access",
                        "refresh_token": "rotated-refresh",
                        "expires_in": 3600,
                        "token_type": "Bearer",
                    },
                )
            self.assertEqual(str(request.url), URL)
            self.assertEqual(request.headers["Authorization"], "Bearer rotated-access")
            return httpx.Response(200)

        async with httpx.AsyncClient(
            auth=restarted, transport=httpx.MockTransport(respond)
        ) as client:
            self.assertEqual((await client.get(URL)).status_code, 200)
        self.assertEqual(paths, ["https://auth.example.com/custom/token", URL])
        again = self.provider()
        await again._initialize()
        self.assertEqual(again.context.current_tokens.refresh_token, "rotated-refresh")
        self.assertGreater(again.context.token_expiry_time, time.time())
        restarted.context.redirect_handler.assert_not_called()

    async def test_refresh_without_new_refresh_token_preserves_previous(self):
        await self.seed()
        auth = self.provider()
        await auth._initialize()
        self.assertTrue(
            await auth._handle_refresh_response(
                httpx.Response(
                    200,
                    json={
                        "access_token": "new-access",
                        "token_type": "Bearer",
                        "expires_in": 3600,
                    },
                )
            )
        )
        tokens = await KeychainStorage(URL).get_tokens()
        self.assertEqual(tokens.refresh_token, "fake-refresh")
        self.assertEqual(tokens.access_token, "new-access")

    async def test_denied_read_and_corrupt_data_fail_closed(self):
        self.backend.get_password.side_effect = RuntimeError("private-secret")
        with self.assertRaises(KeychainStorageError) as error:
            await KeychainStorage(URL).get_tokens()
        self.assertNotIn("private-secret", str(error.exception))
        self.backend.get_password.side_effect = None
        self.backend.get_password.return_value = '{"tokens": "private-secret"}'
        with self.assertRaises(KeychainStorageError):
            await KeychainStorage(URL).get_tokens()
        self.backend.set_password.assert_not_called()

    async def test_write_failure_is_not_reported_as_saved(self):
        await self.seed()
        auth = self.provider()
        await auth._initialize()
        previous = dict(self.values)
        self.backend.set_password.side_effect = RuntimeError("private-secret")
        with self.assertRaises(KeychainStorageError) as error:
            await auth._handle_refresh_response(
                httpx.Response(
                    200,
                    json={
                        "access_token": "new-secret",
                        "token_type": "Bearer",
                    },
                )
            )
        self.assertEqual(previous, self.values)
        with patch("builtins.print") as output:
            report_error(error.exception)
        self.assertNotIn("private-secret", output.call_args.args[0])
        self.assertNotIn("new-secret", output.call_args.args[0])

    async def test_unknown_record_version_fails_closed(self):
        self.values[(SERVICE, f"{URL}:tokens")] = '{"version":2,"tokens":{"access_token":"fake"}}'
        with self.assertRaises(KeychainStorageError):
            await KeychainStorage(URL).get_tokens()

    async def test_rejected_refresh_clears_active_tokens(self):
        await self.seed()
        auth = self.provider()
        await auth._initialize()
        self.assertFalse(await auth._handle_refresh_response(httpx.Response(400)))
        self.assertIsNone(auth.context.current_tokens)


class BackendSelectionTests(unittest.TestCase):
    """後端必須由平台指名，不能退回 keyring 自動選擇（可能是明文後端）。

    把假模組塞進 sys.modules，`from ... import` 就會拿到它，
    於是在任一平台都能測到另一平台的分支。
    """

    def test_macos_uses_keychain(self):
        backend = Mock()
        with patch.dict(
            sys.modules, {"keyring.backends.macOS": Mock(Keyring=Mock(return_value=backend))}
        ):
            with patch("heptabase_keychain.sys.platform", "darwin"):
                self.assertIs(_load_backend(), backend)

    def test_windows_uses_credential_manager(self):
        backend = Mock()
        with patch.dict(
            sys.modules,
            {"keyring.backends.Windows": Mock(WinVaultKeyring=Mock(return_value=backend))},
        ):
            with patch("heptabase_keychain.sys.platform", "win32"):
                self.assertIs(_load_backend(), backend)

    def test_unsupported_platform_fails_without_fallback(self):
        with patch("heptabase_keychain.sys.platform", "linux"):
            with self.assertRaises(KeychainStorageError):
                _load_backend()
            with self.assertRaises(KeychainStorageError):
                KeychainStorage(URL)

    def test_backend_failure_is_reported_without_details(self):
        module = Mock(Keyring=Mock(side_effect=RuntimeError("private-secret")))
        with patch.dict(sys.modules, {"keyring.backends.macOS": module}):
            with patch("heptabase_keychain.sys.platform", "darwin"):
                with self.assertRaises(KeychainStorageError) as error:
                    _load_backend()
        self.assertNotIn("private-secret", str(error.exception))


if __name__ == "__main__":
    unittest.main()
