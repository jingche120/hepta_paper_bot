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

# 設定裡的相對路徑（pool.db_path、hydra.run.dir）以工作目錄為基準。排程器用自己的
# 預設工作目錄啟動時，候選池會指到別處並默默建一個空的，不會報錯。釘住工作目錄，
# 讓從哪裡執行都讀寫同一份資料。
os.chdir(ROOT)

# macOS 的 python.org 版 Python 預設沒有 CA 憑證，urllib（feedparser 用）會以
# CERTIFICATE_VERIFY_FAILED 失敗。指向 certifi 的憑證庫，讓排程執行也不必手動處理。
if "SSL_CERT_FILE" not in os.environ:
    import certifi

    os.environ["SSL_CERT_FILE"] = certifi.where()

from zotero_arxiv_daily.main import main  # noqa: E402

if __name__ == "__main__":
    main()
