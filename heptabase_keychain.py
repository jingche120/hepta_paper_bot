"""作業系統憑證儲存；不使用 .env、一般檔案或自動選擇的後端。"""

import asyncio
import sys
from uuid import uuid4

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

# Windows 認證管理員每筆的上限是 CRED_MAX_CREDENTIAL_BLOB_SIZE（2560 bytes），而 keyring
# 以 UTF-16 編碼寫入，實際只放得下 1280 個字元。授權資料含兩份 OAuth 探索資訊，實測約
# 3000~4000 字元，整筆寫入會得到 WinError 1783，且錯誤發生在瀏覽器授權**之後**——權杖
# 已經拿到卻存不下來，每次重試都白費一輪授權。因此一律分段。macOS 鑰匙圈沒有這個上限，
# 但兩邊共用同一段程式碼，分段對它無害。
CHUNK_CHARS = 1024
# 索引值的前綴。一般酬載是 JSON、必定以 `{` 開頭，不會與這個前綴相撞，所以分段之前
# 以單筆格式存進去的資料仍然讀得回來。
INDEX_PREFIX = "v1:"


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


def _load_backend():
    """依平台指名後端。不呼叫 keyring 的自動選擇，因為它可能落到明文後端。"""
    try:
        if sys.platform == "darwin":
            from keyring.backends.macOS import Keyring

            return Keyring()
        if sys.platform == "win32":
            from keyring.backends.Windows import WinVaultKeyring

            return WinVaultKeyring()
    except Exception:
        raise KeychainStorageError("無法載入系統憑證後端。") from None
    raise KeychainStorageError("此儲存方式需要 macOS 鑰匙圈或 Windows 認證管理員。")


class KeychainStorage:
    """每個 MCP URL 各自儲存用戶端與權杖；拒絕明文降級。"""

    def __init__(self, server_url):
        self.backend = _load_backend()
        self.server_url = server_url
        self.saved = None
        self.context = None

    def _key(self, name):
        return f"{self.server_url}:{name}"

    def _chunk_key(self, name, generation, position):
        return f"{self.server_url}:{name}#{generation}#{position}"

    async def _read(self, name, model):
        try:
            raw = await asyncio.to_thread(self.backend.get_password, SERVICE, self._key(name))
            if raw is None:
                return None
            if raw.startswith(INDEX_PREFIX):
                raw = await self._join(name, raw)
                if raw is None:
                    # 世代不完整，只可能來自被中斷的寫入。當成沒有授權、重跑一次流程，
                    # 比拋錯好：使用者重新授權即可，不必自己去清認證管理員。
                    return None
            return model.model_validate_json(raw)
        except Exception:
            raise KeychainStorageError("鑰匙圈讀取失敗或資料格式不正確。") from None

    async def _join(self, name, index):
        """把索引指到的那個世代接回完整酬載；缺任何一段就視為不完整。"""
        _, generation, count = index.split(":", 2)
        parts = []
        for position in range(int(count)):
            part = await asyncio.to_thread(
                self.backend.get_password, SERVICE, self._chunk_key(name, generation, position)
            )
            if part is None:
                return None
            parts.append(part)
        return "".join(parts)

    async def _write(self, name, value):
        payload = value.model_dump_json()
        chunks = [payload[at : at + CHUNK_CHARS] for at in range(0, len(payload), CHUNK_CHARS)] or [
            ""
        ]
        generation = uuid4().hex[:8]
        try:
            previous = await asyncio.to_thread(self.backend.get_password, SERVICE, self._key(name))
        except Exception:
            previous = None  # 讀不到舊索引只是無法清理，不該擋下這次寫入。
        try:
            for position, chunk in enumerate(chunks):
                await asyncio.to_thread(
                    self.backend.set_password,
                    SERVICE,
                    self._chunk_key(name, generation, position),
                    chunk,
                )
            # 索引是單筆小寫入，也是整次更新的原子切換點：寫成功之前讀到的仍是舊世代，
            # 寫成功之後才整份換新。中途失敗不會讓讀取端看到半套資料。
            await asyncio.to_thread(
                self.backend.set_password,
                SERVICE,
                self._key(name),
                f"{INDEX_PREFIX}{generation}:{len(chunks)}",
            )
        except Exception:
            raise KeychainStorageError("鑰匙圈寫入失敗；授權資料未可靠保存。") from None
        await self._discard(name, previous)

    async def _discard(self, name, index):
        """切換完成後清掉舊世代。清不掉只會留下孤兒項目，不影響正確性。"""
        if not index or not index.startswith(INDEX_PREFIX):
            return
        try:
            _, generation, count = index.split(":", 2)
            for position in range(int(count)):
                await asyncio.to_thread(
                    self.backend.delete_password,
                    SERVICE,
                    self._chunk_key(name, generation, position),
                )
        except Exception:
            pass

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
