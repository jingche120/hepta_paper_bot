"""macOS 鑰匙圈儲存；不使用 .env、一般檔案或自動選擇的後端。"""

import asyncio
import sys

import httpx
from mcp.client.auth import OAuthClientProvider
from mcp.shared.auth import (
    OAuthClientInformationFull,
    OAuthMetadata,
    OAuthToken,
    ProtectedResourceMetadata,
)
from pydantic import BaseModel, ConfigDict, Field

SERVICE = "daily-paper-bot.heptabase.oauth"


class KeychainStorageError(RuntimeError):
    """僅向呼叫端提供不含秘密的錯誤。"""


class SavedAuthorization(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    version: int = Field(default=1, ge=1, le=1)
    tokens: OAuthToken
    expires_at: float | None = None
    oauth_metadata: OAuthMetadata | None = None
    resource_metadata: ProtectedResourceMetadata | None = None
    auth_server_url: str | None = None


class KeychainStorage:
    """每個 MCP URL 各自儲存用戶端與權杖；拒絕明文降級。"""

    def __init__(self, server_url):
        if sys.platform != "darwin":
            raise KeychainStorageError("此儲存方式需要 macOS 鑰匙圈。")
        try:
            from keyring.backends.macOS import Keyring

            self.backend = Keyring()
        except Exception:
            raise KeychainStorageError("無法載入 macOS 鑰匙圈後端。") from None
        self.server_url = server_url
        self.saved = None
        self.context = None

    async def _read(self, name, model):
        try:
            raw = await asyncio.to_thread(
                self.backend.get_password, SERVICE, f"{self.server_url}:{name}"
            )
            return None if raw is None else model.model_validate_json(raw)
        except Exception:
            raise KeychainStorageError("鑰匙圈讀取失敗或資料格式不正確。") from None

    async def _write(self, name, value):
        try:
            await asyncio.to_thread(
                self.backend.set_password,
                SERVICE,
                f"{self.server_url}:{name}",
                value.model_dump_json(),
            )
        except Exception:
            raise KeychainStorageError("鑰匙圈寫入失敗；授權資料未可靠保存。") from None

    async def get_tokens(self):
        self.saved = await self._read("tokens", SavedAuthorization)
        return self.saved.tokens if self.saved else None

    async def set_tokens(self, tokens):
        if self.context is None:
            raise KeychainStorageError("尚未連接 OAuth 授權狀態。")
        saved = SavedAuthorization(
            tokens=tokens,
            expires_at=self.context.token_expiry_time,
            oauth_metadata=self.context.oauth_metadata,
            resource_metadata=self.context.protected_resource_metadata,
            auth_server_url=self.context.auth_server_url,
        )
        await self._write("tokens", saved)
        self.saved = saved

    async def get_client_info(self):
        return await self._read("client", OAuthClientInformationFull)

    async def set_client_info(self, client_info):
        await self._write("client", client_info)


class PersistentOAuthClientProvider(OAuthClientProvider):
    """補足 mcp==1.26.0 重啟後的到期時間、探索資訊與 refresh token 保留。"""

    def __init__(self, *, storage, **kwargs):
        super().__init__(storage=storage, **kwargs)
        storage.context = self.context

    async def _initialize(self):
        await super()._initialize()
        saved = self.context.storage.saved
        if saved:
            self.context.token_expiry_time = saved.expires_at
            self.context.oauth_metadata = saved.oauth_metadata
            self.context.protected_resource_metadata = saved.resource_metadata
            self.context.auth_server_url = saved.auth_server_url

    async def _handle_refresh_response(self, response):
        # OAuth 允許更新回應省略 refresh_token，此時保留原值。
        if response.status_code == 200 and self.context.current_tokens:
            try:
                data = response.json()
                if isinstance(data, dict) and "refresh_token" not in data:
                    data["refresh_token"] = self.context.current_tokens.refresh_token
                    response = httpx.Response(200, json=data)
            except ValueError:
                pass  # 無效回應仍交給 SDK 處理。
        return await super()._handle_refresh_response(response)
