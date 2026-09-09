# 每日推薦論文 bot

以 Heptabase 白板「讀過的論文總覽們」當成你的閱讀紀錄 —— 白板上每張卡片是一篇
讀過的論文總覽 —— 每天從 arXiv 等來源檢索新論文，依與閱讀紀錄的相似度排序後寄出清單。

推薦引擎來自 [TideDra/zotero-arxiv-daily](https://github.com/TideDra/zotero-arxiv-daily)，
原本讀 Zotero，本專案把資料來源換成 Heptabase。流程分兩步：

```
Heptabase 白板 --匯出--> data/reading_history.json --讀取--> 檢索 → 排序 → 寄信
 (get_hepthabase_bot.py --export)                         (main.py)
```

分兩步而不是一次跑完，是因為 Heptabase 讀取需要 OAuth 瀏覽器授權，
把它與可排程、可離線重跑的推薦流程分開比較穩定。

`src/zotero_arxiv_daily/` 是上游程式碼的複製，**刻意維持原樣**（連命名與格式都不動），
以便日後與上游比對合併；因此 ruff 也把該目錄排除在檢查之外。
我們的修改集中在 `heptabase_reading_history.py`、`executor.py` 的閱讀紀錄來源，以及 `config/`。

想了解設計理由、各檔案職責與已知限制，看 [docs/專案說明.md](docs/專案說明.md)；
本文件只講安裝與操作。

**平台**：macOS 與 Windows 都可執行。本文件的指令以 macOS 寫法為準，Windows 請換成
venv 的對應路徑：`.venv/bin/python` → `.venv\Scripts\python.exe`、
`.venv/bin/ruff` → `.venv\Scripts\ruff.exe`。OAuth 授權在 macOS 存進鑰匙圈、
在 Windows 存進認證管理員，兩者都由作業系統加密保管，不會落到明文檔案。
每天自動執行的設定見〈[每日排程](#每日排程)〉。

## Heptabase 連線：如何執行

在專案目錄執行：

```bash
.venv/bin/python get_hepthabase_bot.py
```

首次執行請在自動開啟的瀏覽器登入 Heptabase 並允許授權，然後回到終端機查看結果。
macOS 也可能要求允許 Python 存取鑰匙圈；請確認提示來自本專案使用的 Python。
Windows 改用認證管理員，通常不會另外提示，但防火牆可能詢問是否允許本機回呼埠。
再次執行通常不需開啟瀏覽器；授權遭撤銷或 refresh token 失效時仍可能需要重新登入。
只有 MCP 初始化成功才會顯示「Heptabase MCP 連線成功」。
回呼頁顯示已收到回應，不代表權杖交換及 MCP 初始化已成功，請以終端機為準。

需要 macOS 或 Windows，以及 Python 3.11 以上。首次安裝（既有使用者也需重新安裝依賴）：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Windows 用 `py -3.12 -m venv .venv` 建立環境（`.venv/` 不可跨平台複製，一定要重建）。
`keyring` 會一併帶入 Windows 需要的 `pywin32-ctypes`，不必另外安裝。

瀏覽器與程式需在同一台電腦。本機回呼使用 `127.0.0.1:8765`，請勿同時啟動兩份程式。
授權等候上限為 5 分鐘；可按 Ctrl+C 取消。若瀏覽器無法自動開啟，程式會顯示授權網址供你在本機開啟。

## 如何讀懂主程式

先讀 `get_hepthabase_bot.py`，依下列順序理解：

1. `if __name__ == "__main__"`：直接執行這個檔案時，呼叫 `main()`。
2. `main()`：使用 `asyncio.run(connect())` 啟動非同步連線流程，並處理失敗或取消。
3. `oauth_connection(MCP_URL)`：準備 OAuth 元件及本機回呼接收服務，這時還未完成登入。
4. `httpx.AsyncClient(auth=oauth, follow_redirects=True)`：建立 HTTP 用戶端，將授權處理交給 OAuth 元件。Heptabase 的 OAuth 探索網址會回傳 302 轉址，因此必須啟用 `follow_redirects=True` 才能取得真正的授權端點。
5. `streamable_http_client(...)`：建立 MCP 的 HTTP 傳輸通道，得到接收端 `read` 與傳送端 `write`；`_` 表示此處不使用的第三個回傳值。
6. `ClientSession(read, write)`：在傳輸通道上建立 MCP 工作階段。
7. `await session.initialize()`：送出 MCP 初始化請求並等待結果。需要授權時，HTTP 層會啟動 OAuth、開啟瀏覽器、取得權杖，再繼續請求。
8. `list_whiteboard_cards(session, WHITEBOARD_ID)`：列出白板上的卡片。不加 `--export` 時只印出標題清單；加了就進入 `export_whiteboard()`，逐張 `read_card()` 並解析成閱讀紀錄。離開 `async with` 區塊時，自動關閉連線。

`async def` 宣告可以等待的非同步函式；`await` 表示等候結果，同時讓其他非同步工作繼續，例如接收瀏覽器回呼。
`async with` 負責資源的建立與清理，即使中途發生錯誤也會執行清理。

## OAuth 細節放在哪裡

`heptabase_oauth.py` 包含主程式需要的授權元件：

- `heptabase_keychain.py` 的 `KeychainStorage`：將權杖、到期時間、授權伺服器資訊及用戶端註冊資料存入 macOS 鑰匙圈。直接指定 macOS 後端，不自動降級為明文或其他後端。
- `PersistentOAuthClientProvider`：針對固定版本 `mcp==1.26.0`，恢復重啟後的到期時間與探索資訊；若更新回應未附新 refresh token，保留舊值。升級 MCP SDK 時需重新檢查這些覆寫方法。
- `BrowserCallback.open_browser()`：使用 SDK 建立的授權網址開啟瀏覽器。
- `handle_request()`：接收瀏覽器送到本機 `/callback` 的 HTTP 請求。
- `accept_callback()`：檢查回呼路徑、授權碼與 `state`，防止接受其他登入流程的回呼。
- `wait_for_callback()`：等待回呼，將授權碼與 `state` 交回 SDK。
- `oauth_connection()`：組合上述元件。OAuth 探索、註冊、PKCE 及權杖交換由官方 SDK 處理。

本機 HTTP 服務只用來接收 OAuth 回呼；MCP 伺服器仍由 Heptabase 提供。

## 私密資料與驗證

程式不讀取 `.env`，也不將授權寫入一般 JSON 檔或原始碼。鑰匙圈 service 名稱為
`daily-paper-bot.heptabase.oauth`，帳號項目為 MCP URL 加上 `:tokens` 或 `:client`。
權杖與其到期／探索狀態會一併保存，避免重啟後將過期權杖誤判為有效。
資料格式損壞、鑰匙圈被鎖定或存取遭拒時，程式會停止並顯示不含秘密的錯誤。

這保護的是授權資料；未來匯出的卡片 JSON 仍需另外保護。鑰匙圈也不是對同一個 Python
執行檔執行的所有程式完全隔離，請勿用它執行不可信任的程式。

若要清除本程式的登入資料，在「鑰匙圈存取」搜尋上述 service 名稱，只刪除對應 Heptabase
URL 的 `:tokens` 與 `:client` 兩個項目；下次執行會重新授權。這不等於在 Heptabase 撤銷授權。

實際登入仍需你在瀏覽器完成；離線測試無法證明帳號授權或真實權杖更新成功。

```bash
.venv/bin/python -m unittest discover -s tests -v
```

測試模擬鑰匙圈、瀏覽器、HTTP 傳輸及 MCP 工作階段，不存取真實鑰匙圈、Heptabase 或 `.env`。
涵蓋跨儲存實例的授權重用、過期後向已保存端點更新、refresh token 輪替與省略、讀寫失敗及錯誤訊息遮蔽。
其中 OAuth 探索測試模擬 302 轉址、註冊回應，確認 SDK 能繼續到開啟瀏覽器的步驟。
如果仍連線失敗，請提供終端機顯示的錯誤類型或 HTTP 狀態碼；程式不印出完整私密回應。

## 手動驗收與排程

1. 執行 `.venv/bin/python get_hepthabase_bot.py`，完成首次 OAuth 與可能的鑰匙圈提示。
2. 確認終端機顯示「Heptabase MCP 連線成功」。
3. 再次執行同一命令，確認不需瀏覽器授權且仍連線成功。
4. 等待服務端權杖自然到期後再執行，驗證真實 refresh token 是否可用。

### 明確驗證免 OAuth 互動

```bash
.venv/bin/python get_hepthabase_bot.py --no-browser
```

此模式禁止開啟 OAuth 授權頁；若需要重新授權會立即失敗，以非零狀態結束。
成功時顯示「免 OAuth 互動驗證成功：本次未開啟瀏覽器授權。」。
它允許使用 refresh token 自動更新；macOS 鑰匙圈的系統權限提示與 OAuth 是不同機制。
此測試確認當次執行不需 OAuth，並不保證授權永不失效。

### 固定讀取的白板

白板 ID 固定在 `get_hepthabase_bot.py` 的 `WHITEBOARD_ID`：

```python
# 白板「讀過的論文總覽們」，每張卡片是一篇讀過的論文總覽。
WHITEBOARD_ID = "00000000-0000-4000-8000-000000000000"
```

直接執行程式或按編輯器的執行按鈕，會列出白板上的卡片標題與建立時間，作為連線檢查；
**不會輸出卡片內文**。加上 `--export` 才會逐張讀取、解析並存檔（見下一節）。
`--no-browser` 僅保留作為可選的免 OAuth 驗證模式。
讀取會處理長內容分頁，不呼叫 LLM，也不修改 Heptabase。
請勿將含私密內容的終端機輸出貼到公開日誌。

請勿同時啟動多份程式。排程需使用同一個作業系統使用者與同一個 Python，並確保電腦開機；
首次授權與權限設定先互動完成。設定方式見〈[每日排程](#每日排程)〉。

## 匯出閱讀紀錄

```bash
.venv/bin/python get_hepthabase_bot.py --export
```

依建立時間由舊到新列出白板上的卡片，逐張讀取解析成 `data/reading_history.json`，
並把所有卡片原始全文另存成 `data/heptabase_cards_raw.txt`。
加路徑可改變輸出位置：`--export 其他/路徑.json`。
相對路徑一律以專案根目錄為基準，因此在哪個目錄執行都會寫到同一個檔案。

執行時會逐張回報解析結果，可立即看出哪張卡片的摘要異常短：

```
白板上有 1 張卡片，開始逐張讀取…
  [1/1] Real-time Seismic Intensity Prediction using Self-｜摘要 1857 字｜關鍵字 7 個
```

`data/` 已列入 `.gitignore`：匯出內容是你的私人筆記，不進版控。

### 卡片格式與解析規則

白板上每張卡片是一篇論文總覽，依 Markdown 章節組織：

```markdown
# 論文總覽：SC-GNN 地震烈度即時預測

來源論文：Real-time Seismic Intensity Prediction using Self-supervised Contrastive GNN

完整標題：Real-time Seismic Intensity Prediction using Self-supervised Contrastive GNN

## English Abstract

This paper studies regional earthquake early warning as a direct regression problem...

## 中文摘要

本研究將區域型地震預警表述為直接的烈度迴歸問題...

## 關鍵字

Earthquake Early Warning、Seismic Intensity Prediction、SC-GNN。

## 方法概覽
（以下章節不進閱讀紀錄，可自由增減）
```

| 卡片內容 | 對應閱讀紀錄欄位 | 作用 |
|---|---|---|
| `完整標題：` 那一行（沒有則 `來源論文：`） | `title` | **不參與計分**，只用於顯示與日誌 |
| `## English Abstract` + `## 中文摘要` | `abstract` | **唯一決定推薦品質的欄位** |
| `## 關鍵字` | `paths` | 供 `include_path` / `ignore_path` 做 glob 篩選 |
| 卡片建立時間（來自 `list_cards`） | `added_date` | 時間衰減權重，越新的卡片權重越高 |

細節：

- **只有摘要兩節進閱讀紀錄。** 方法概覽、實驗設定、主要結果等章節會被排除，
  避免實驗數據稀釋語意訊號。所以摘要要寫得完整，別只寫一行。
- **英文摘要請保留原文。** 要比對的是 arXiv 的英文摘要，同語言訊號最強；
  中文摘要則補上你關注的角度。兩節都寫效果最好，只寫一節也能運作。
- **章節名稱有彈性**：英文摘要接受 `English Abstract`／`Abstract`／`英文摘要`；
  中文摘要接受 `中文摘要`／`摘要`；關鍵字接受 `關鍵字`／`關鍵詞`／`Keywords`。
- **標題來源**：依序取 `完整標題：`、`來源論文：`、卡片的一級標題，最後才用
  `list_cards` 回報的卡片標題。`完整標題：` 是你手寫的論文全名，最可信，所以排最前面
  —— 有了它，一級標題就能自由寫成短代號（`# 論文總覽：PatchTST`）。
  `來源論文：` 指向的 PDF 卡片有時以 arXiv 查詢字串命名，那不是標題，會被自動跳過；
  `完整標題：` 同樣受這道檢查保護。兩者的前後 `*`／`_` 強調符號會去掉，
  `<hepta-mention>` 標籤會被移除、內文保留。
- **關鍵字分隔符**：頓號、逗號、分號都可以，結尾句號會自動去掉。
- **找不到摘要章節時**，會退回使用整張卡片內文，並在匯出結束時列出這些卡片提醒你補寫 ——
  不會靜默漏掉整篇論文。
- 卡片內文可以放編號清單、項目符號、粗體、LaTeX、連結，這些都能正常讀出。
  （表格儲存格內的巢狀內容則會變成 `[complex]`／`[list]` 佔位符，所以摘要別寫在表格裡。）

匯出後可用下列指令核對解析結果，必要時對照 `data/heptabase_cards_raw.txt`：

```bash
.venv/bin/python -c "import json;d=json.load(open('data/reading_history.json'));print('筆數:',len(d['papers']));[print(i,'|',p['title'][:40],'|',len(p['abstract']),'字 |',p['paths']) for i,p in enumerate(d['papers'][:5])]"
```

## 每日推薦流程

```bash
.venv/bin/python main.py
```

讀取 `data/reading_history.json` 當閱讀紀錄，檢索當日新論文，依相似度排序後寄信。
設定用 [Hydra](https://hydra.cc)，預設值與各項說明在 `config/base.yaml`，
本機覆寫寫在 `config/custom.yaml`（不進版控），祕密值放 `.env`（參考 `.env.example`）。
命令列也可臨時覆寫：`.venv/bin/python main.py executor.max_paper_num=5 executor.debug=true`。

### 執行前必填

**`source.arxiv.category` 必須設定**，否則程式會直接報
`category must be specified for arxiv.`。在 `config/custom.yaml` 填入你的研究領域
（代號查 [arXiv category taxonomy](https://arxiv.org/category_taxonomy)）：

```yaml
source:
  arxiv:
    category: ["cs.AI", "cs.LG", "cs.CL"]
```

寄信則需在 `.env` 填好 `SENDER_EMAIL`、`SENDER_APP_PASSWORD`、`SMTP_RECEIVER`、
`SMTP_SERVER` 四個變數（連接埠預設 465，要改請在 `config/custom.yaml` 寫 `email.smtp_port`）。

### 檢索範圍：當日或多天

預設每天執行、只讀 arXiv 當日 RSS 公告（`lookback_days: null`）。

要一次抓多天，把 `source.arxiv.lookback_days` 設成天數，會改走 arXiv API 的
`submittedDate` 區間查詢。實測以 `physics.geo-ph + eess.SP + stat.ML` 抓 7 天
會得到 178 篇，檢索本身只要 21 秒。

論文全文改成**排序後才抓**（只抓前 `executor.max_paper_num` 篇），
所以下載量由該設定決定，不隨候選數成長 —— 加大檢索範圍不會讓執行時間爆掉。

### SSL 憑證

macOS 的 python.org 版 Python 預設沒有 CA 憑證，抓 arXiv RSS 會以
`CERTIFICATE_VERIFY_FAILED` 失敗。`main.py` 已自動指向 `certifi` 的憑證庫，
不需額外處理。若想改為修正 Python 安裝本身，可執行
`/Applications/Python 3.12/Install Certificates.command`。

### 依賴安裝

```bash
.venv/bin/python -m pip install -r requirements.txt            # 基本流程
.venv/bin/python -m pip install -r requirements-reranker.txt   # 本機排序模型，額外數百 MB
```

### 關於 LLM API

尚未申請 OpenAI 金鑰時也能執行：

- **排序**用本機 embedding 模型（`executor.reranker: local`），不需要 API。
- **TLDR 與作者單位**會呼叫 LLM 失敗，程式自動回退 —— TLDR 顯示原始摘要，作者單位留空。
  整條流程仍會跑完並寄出信件。

之後申請到金鑰，把 `OPENAI_API_KEY` 寫進 `.env` 即可啟用 TLDR。
若也想改用 API embedding 省下 torch，再把 `executor.reranker` 改成 `api`
並填好 `.env` 的 `EMBEDDING_*` 變數。

## 每日排程

只有 `main.py` 適合排程。`get_hepthabase_bot.py --export` 需要瀏覽器授權，
無法無人值守執行 —— 但排程讀的是已匯出的 `data/reading_history.json`，
所以閱讀紀錄隔一段時間手動更新一次即可，不影響每天的推薦。

### Windows（工作排程器）

`scripts/run_daily.bat` 已備妥：它會切換到專案目錄、設定 `PYTHONUTF8=1`，
再把輸出附加到 `logs/daily.log`。`PYTHONUTF8=1` 是必要的 —— 排程非互動執行時
主控台會退回 ANSI 代碼頁（繁中 Windows 是 cp950），論文標題若含代碼頁外的字元
會在半途丟 `UnicodeEncodeError` 中斷整場執行。

開啟「工作排程器」→ **建立工作**（不是「建立基本工作」，選項不夠）：

| 頁籤 | 設定 |
| --- | --- |
| 一般 | 只在使用者登入時執行 |
| 觸發程序 | 每天，開始時間 09:00 |
| 動作 → 程式或指令碼 | `<專案路徑>\scripts\run_daily.bat` |
| 動作 → 起始於 | `<專案路徑>` |
| 設定 | 失敗時每 30 分鐘重試、最多 3 次；執行超過 1 小時則停止 |

建立後先按「執行」手動觸發一次，確認結果碼為 `0x0`，不要等到隔天才發現問題。

### macOS（launchd）

尚未建立。可用 `launchd` 的 `StartCalendarInterval`，指向 `.venv/bin/python main.py`
並把 `WorkingDirectory` 設為專案目錄。

### 排程時的路徑

`main.py` 啟動時會 `os.chdir()` 到專案根目錄，所以 `config` 裡的相對路徑
（`pool.db_path`、Hydra 的 `outputs/`）不受排程器的工作目錄影響。
少了這一步，排程器會用自己的預設工作目錄，程式**會默默建立一個空的候選池**
並推薦出無意義的結果，且不會報錯 —— 驗收時請確認 log 顯示的池中筆數不是 0。

## 開發檢查

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/python -m unittest discover -s tests -v
```

格式與 lint 規則定義於 `pyproject.toml`；`src/zotero_arxiv_daily/`（上游複製）排除在外。

測試涵蓋 OAuth／鑰匙圈流程（`test_client.py`、`test_keychain.py`）、卡片讀取分頁與白板卡片列舉
（`test_cards.py`）、卡片解析與閱讀紀錄（`test_reading_history.py`）、閱讀紀錄載入與路徑篩選
（`test_executor_history.py`），以及 glob 相容層（`test_utils_glob.py`）。全部不觸網。
