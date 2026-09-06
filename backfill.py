"""把 2022 年以來的相關論文回填進候選池。

一次性作業，但**可以重複執行**：已存在的論文會被 `PaperPool.ingest()` 以
id 與正規化標題去重，因此中斷後直接重跑即可，不會產生重複資料。

兩個來源的取法不同：

- arXiv 走官方 API 的 `submittedDate` 區間查詢。單次查詢有結果數上限，
  因此**按季分段**，每段各自寫入。
- IEEE TGRS / GRSL 走 OpenAlex。它們不在 arXiv 上，而 IEEE Xplore 的 API 需要訂閱。

向量計算另外用 `--embed` 觸發，因為那會呼叫付費 API（約 $0.15 一次性）。
"""

import argparse
import os
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from embeddings import embed_pending  # noqa: E402
from openalex import GRSL, TGRS, OpenAlexClient  # noqa: E402
from paper_pool import PaperRecord, open_pool  # noqa: E402

DEFAULT_DB = "data/paper_pool.sqlite"
DEFAULT_FROM = "2022-01-01"
DEFAULT_ARXIV_CATEGORIES = ["physics.geo-ph"]
DEFAULT_JOURNALS = {TGRS: "IEEE TGRS", GRSL: "IEEE GRSL"}


def quarter_starts(start, end):
    """把日期區間切成每季的起訖點，避開 arXiv 單次查詢的結果上限。"""
    if start > end:
        return []
    spans = []
    cursor = date(start.year, ((start.month - 1) // 3) * 3 + 1, 1)
    while cursor <= end:
        year, month = cursor.year, cursor.month + 3
        if month > 12:
            year, month = year + 1, month - 12
        following = date(year, month, 1)
        spans.append((max(cursor, start), min(following - timedelta(days=1), end)))
        cursor = following
    return spans


def arxiv_records(categories, start, end, config_overrides=None):
    """用既有的 arXiv 檢索器逐季取回論文，轉成候選池的紀錄格式。"""
    from omegaconf import OmegaConf

    from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever

    config = OmegaConf.create(
        {
            "source": {
                "arxiv": {
                    "category": list(categories),
                    # 回填要涵蓋跨領域掛號的論文，例如主分類在 cs.LG、
                    # 同時掛到 physics.geo-ph 的機器學習地震論文。
                    "include_cross_list": True,
                    "lookback_days": None,
                    "max_results": 2000,
                }
            },
            "executor": {"debug": False},
        }
    )
    if config_overrides:
        config = OmegaConf.merge(config, config_overrides)
    retriever = ArxivRetriever(config)

    for span_start, span_end in quarter_starts(start, end):
        days = (span_end - span_start).days + 1
        results = retriever._retrieve_by_date_range_between(span_start, span_end)
        print(f"  arXiv {span_start} ~ {span_end}（{days} 天）：{len(results)} 篇")
        for result in results:
            paper = retriever.convert_to_paper(result)
            yield PaperRecord(
                id="arxiv:" + paper.url.rsplit("/", 1)[-1],
                source="arxiv",
                title=paper.title,
                abstract=paper.abstract,
                url=paper.url,
                pdf_url=paper.pdf_url,
                published_date=result.published.date().isoformat()
                if getattr(result, "published", None)
                else None,
            )


def journal_records(client, sources, start):
    for source_id, name in sources.items():
        count = 0
        for record in client.fetch_works(source_id, start.isoformat()):
            count += 1
            yield record
        print(f"  {name}：{count} 篇（已扣除無摘要者）")


def embed(pool, model, base_url, api_key, batch_size):
    """把池中缺向量的論文補齊。這一步會呼叫付費 API。"""
    from openai import OpenAI

    if not api_key:
        raise SystemExit(
            "缺少 API 金鑰。請在 .env 設定 EMBEDDING_API_KEY（或 OPENAI_API_KEY），"
            "或改用 --embed-key 傳入。"
        )
    client = OpenAI(api_key=api_key, base_url=base_url or None)
    done = embed_pending(pool, client, model, batch_size)
    print(f"  計算了 {done} 篇的向量")


def run(args):
    pool = open_pool(args.db)
    try:
        if args.embed_only:
            print(f"計算向量（模型 {args.embed_model}）…")
            embed(pool, args.embed_model, args.embed_base_url, args.embed_key, args.batch_size)
            print(f"\n池中共 {pool.stats()['total']} 篇，全部已有向量。")
            return 0

        start = date.fromisoformat(args.since)
        end = date.fromisoformat(args.until) if args.until else date.today()

        if not args.skip_arxiv:
            print(f"回填 arXiv {args.categories}…")
            result = pool.ingest(arxiv_records(args.categories, start, end))
            print(f"  寫入 {result.inserted}，跳過 {result.skipped}")

        if not args.skip_journals:
            print("回填 IEEE 期刊…")
            client = OpenAlexClient(mailto=args.mailto)
            result = pool.ingest(journal_records(client, DEFAULT_JOURNALS, start))
            print(f"  寫入 {result.inserted}，跳過 {result.skipped}")

        if args.embed:
            print(f"計算向量（模型 {args.embed_model}）…")
            embed(pool, args.embed_model, args.embed_base_url, args.embed_key, args.batch_size)

        stats = pool.stats()
        print(
            f"\n池中共 {stats['total']} 篇 {stats['by_source']}；"
            f"待計算向量 {stats['pending_embedding']} 篇"
        )
        if stats["pending_embedding"]:
            print("計算向量請執行：.venv/bin/python backfill.py --embed-only")
    finally:
        pool.close()
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="把歷史論文回填進候選池")
    parser.add_argument("--db", default=DEFAULT_DB, help=f"候選池路徑，預設 {DEFAULT_DB}")
    parser.add_argument("--since", default=DEFAULT_FROM, help=f"起始日期，預設 {DEFAULT_FROM}")
    parser.add_argument("--until", default=None, help="結束日期，預設今天")
    parser.add_argument(
        "--categories", nargs="+", default=DEFAULT_ARXIV_CATEGORIES, help="arXiv 分類"
    )
    parser.add_argument("--mailto", default="", help="OpenAlex polite pool 用的聯絡信箱")
    parser.add_argument("--skip-arxiv", action="store_true", help="只回填期刊")
    parser.add_argument("--skip-journals", action="store_true", help="只回填 arXiv")
    parser.add_argument("--embed", action="store_true", help="回填後接著計算向量（會呼叫付費 API）")
    parser.add_argument("--embed-only", action="store_true", help="只計算向量，不回填")
    parser.add_argument(
        "--embed-model", default=None, help="embedding 模型，預設讀 EMBEDDING_MODEL"
    )
    parser.add_argument("--embed-key", default=None, help="API 金鑰，預設讀 .env")
    parser.add_argument("--embed-base-url", default=None, help="API 位址，預設讀 .env")
    parser.add_argument("--batch-size", type=int, default=64, help="每批送出的摘要數")
    args = parser.parse_args(argv)

    if args.embed or args.embed_only:
        import dotenv

        dotenv.load_dotenv()
        args.embed_model = args.embed_model or os.environ.get(
            "EMBEDDING_MODEL", "text-embedding-3-small"
        )
        args.embed_key = (
            args.embed_key
            or os.environ.get("EMBEDDING_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        args.embed_base_url = args.embed_base_url or os.environ.get("EMBEDDING_BASE_URL")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
