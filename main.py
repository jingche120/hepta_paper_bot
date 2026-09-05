"""推薦流程的進入點；設定模組搜尋路徑後交給實作模組。

`src/` 放上游 zotero-arxiv-daily 的程式碼，專案根目錄放 Heptabase 客戶端
（`heptabase_reading_history` 等）。executor 需要匯入後者，所以兩個路徑都要加進來。
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# macOS 的 python.org 版 Python 預設沒有 CA 憑證，urllib（feedparser 用）會以
# CERTIFICATE_VERIFY_FAILED 失敗。指向 certifi 的憑證庫，讓排程執行也不必手動處理。
if "SSL_CERT_FILE" not in os.environ:
    import certifi

    os.environ["SSL_CERT_FILE"] = certifi.where()

from zotero_arxiv_daily.main import main  # noqa: E402

if __name__ == "__main__":
    main()
