"""檢查新機器上的安裝是否完整，列出還缺什麼。

搬到新電腦後在跑 main.py 之前執行，可省下逐一試錯的時間：

    .venv/bin/python scripts/check_setup.py           # macOS
    .\\.venv\\Scripts\\python.exe scripts\\check_setup.py   # Windows

不連網、不寄信、不寫入資料庫。祕密值只回報「有沒有填」，不會印出內容，
因此輸出可以安全貼給別人看。
"""

import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

# 只回報有無填值，永遠不印內容。
SECRETS = ("OPENAI_API_KEY", "EMBEDDING_API_KEY", "SENDER_APP_PASSWORD")
SETTINGS = ("SENDER_EMAIL", "SMTP_RECEIVER", "SMTP_SERVER")

# 搬遷時最常漏掉的檔案；都不進版控，git clone 不會帶過來。
REQUIRED_FILES = (
    (".env", "祕密值。從 .env.example 複製後填入。"),
    ("config/custom.yaml", "本機設定。缺少時 arxiv 分類為空，main.py 會直接報錯。"),
    ("data/reading_history.json", "閱讀紀錄。由 get_hepthabase_bot.py --export 產生。"),
    ("data/paper_pool.sqlite", "候選池。重建要重算向量並付 API 費用，務必從舊機器複製。"),
)

# 用 ASCII 標記：cp950 沒有勾號等符號，Windows 排程輸出到檔案時會炸掉。
OK = "[ OK ]"
BAD = "[FAIL]"
WARN = "[WARN]"


def check_python(report):
    version = sys.version_info
    text = f"Python {version.major}.{version.minor}.{version.micro}（{sys.executable}）"
    if version < (3, 11):
        report(BAD, text, "需要 3.11 以上。請用 py -3.12 -m venv .venv 重建虛擬環境。")
    elif version[:2] != (3, 12):
        report(WARN, text, "專案在 3.12 上開發；其他版本可跑，但與開發環境不一致。")
    else:
        report(OK, text)


def check_packages(report):
    from importlib.metadata import PackageNotFoundError, version

    names = []
    for line in Path("requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line:
            names.append(requirement_name(line))

    missing = []
    for name in names:
        try:
            version(name)
        except PackageNotFoundError:
            missing.append(name)

    if missing:
        report(
            BAD,
            f"依賴套件缺少 {len(missing)} 個：{', '.join(missing)}",
            "請重跑 pip install -r requirements.txt。",
        )
    else:
        report(OK, f"依賴套件 {len(names)} 個都已安裝")


def requirement_name(line):
    """取出套件名稱，去掉版本條件。"""
    for separator in ("==", ">=", "<=", "~=", ">", "<"):
        if separator in line:
            return line.split(separator)[0].strip()
    return line


def check_files(report):
    missing = False
    for name, why in REQUIRED_FILES:
        path = Path(name)
        if path.exists():
            size = path.stat().st_size
            report(OK, f"{name}（{size / 1_048_576:.1f} MB）" if size > 1_048_576 else name)
        else:
            missing = True
            report(BAD, f"缺少 {name}", why)
    return not missing


def check_pool(report):
    path = Path("data/paper_pool.sqlite")
    if not path.exists():
        return
    try:
        # 唯讀開啟，不觸發 schema 升級，也不會寫入。
        connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        total, embedded = connection.execute(
            "select count(*), count(embedding) from papers"
        ).fetchone()
        connection.close()
    except sqlite3.Error as error:
        report(
            BAD,
            f"候選池無法讀取：{type(error).__name__}",
            "檔案可能在傳輸中損壞，請重新複製並比對 SHA256。",
        )
        return

    if total == 0:
        report(
            BAD,
            "候選池是空的（0 篇）",
            "多半是複製到錯的位置，或程式曾在別的工作目錄下另建了一個空池。",
        )
    else:
        report(OK, f"候選池 {total} 篇，其中 {embedded} 篇已有向量")


def check_reading_history(report):
    if not Path("data/reading_history.json").exists():
        return
    try:
        from heptabase_reading_history import load_reading_history

        records = load_reading_history("data/reading_history.json")
    except Exception as error:
        report(BAD, f"閱讀紀錄無法載入：{type(error).__name__}", "請重新匯出或重新複製檔案。")
        return
    report(OK, f"閱讀紀錄 {len(records)} 篇")


def check_config(report):
    if not Path("config/custom.yaml").exists():
        return
    try:
        from omegaconf import OmegaConf

        # 只讀這幾個鍵，不會觸發 ${oc.env:...} 解析，因此不需要環境變數。
        config = OmegaConf.merge(
            OmegaConf.load("config/base.yaml"), OmegaConf.load("config/custom.yaml")
        )
        category = config.source.arxiv.category
        reranker = config.executor.reranker
        whiteboard = OmegaConf.select(config, "heptabase.whiteboard_id")
    except Exception as error:
        report(BAD, f"設定檔無法解析：{type(error).__name__}")
        return

    if not category:
        report(
            BAD,
            "source.arxiv.category 未設定",
            "main.py 會直接報 category must be specified for arxiv。",
        )
    else:
        report(OK, f"arxiv 分類：{list(category)}")

    if not whiteboard or not str(whiteboard).strip():
        report(
            BAD,
            "heptabase.whiteboard_id 未設定",
            "get_hepthabase_bot.py 不知道要讀哪個白板。在 config/custom.yaml 填入白板 ID。",
        )
    else:
        # 只回報有填，不印 ID：它是個人識別碼，輸出可能被貼給別人看。
        report(OK, "heptabase.whiteboard_id 已填")

    if reranker == "local":
        report(
            WARN,
            "executor.reranker = local",
            "需要 requirements-reranker.txt（torch，數百 MB）。用 API 排序請改成 api。",
        )
    else:
        report(OK, f"排序方式：{reranker}")


def check_env(report):
    if not Path(".env").exists():
        return
    import dotenv

    dotenv.load_dotenv()
    for name in SECRETS + SETTINGS:
        if os.getenv(name):
            report(OK, f"{name} 已填")
        else:
            report(BAD, f"{name} 是空的", "在 .env 填入。搬機器時祕密值要手動補回。")


def main():
    failures = []

    def report(status, text, hint=None):
        print(f"{status} {text}")
        if hint:
            print(f"       -> {hint}")
        if status == BAD:
            failures.append(text)

    print(f"專案目錄：{ROOT}\n")
    check_python(report)
    check_packages(report)
    print()
    check_files(report)
    check_pool(report)
    check_reading_history(report)
    print()
    check_config(report)
    check_env(report)

    print()
    if failures:
        print(f"{len(failures)} 項待處理，請照上面的提示補齊後重跑本腳本。")
        return 1
    print("檢查通過。下一步：main.py executor.dry_run=true（不寄信，確認選出的論文）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
