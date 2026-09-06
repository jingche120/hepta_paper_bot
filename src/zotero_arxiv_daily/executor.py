from loguru import logger
from omegaconf import DictConfig, ListConfig
from embeddings import embed_pending, get_embedder
from heptabase_reading_history import load_reading_history
from openalex import OpenAlexClient, publication_window_start
from paper_pool import PaperRecord, open_pool, weighted_centroid
from .utils import glob_match
from .retriever import get_retriever_cls
from .protocol import CorpusPaper, Paper
import random
from datetime import date
from time import sleep
from .construct_email import render_email
from .utils import send_email
from openai import OpenAI
from tqdm import tqdm


def normalize_path_patterns(patterns: list[str] | ListConfig | None, config_key: str) -> list[str] | None:
    if patterns is None:
        return None

    if not isinstance(patterns, (list, ListConfig)):
        raise TypeError(
            f"config.heptabase.{config_key} must be a list of glob patterns or null, "
            'for example ["2026/survey/**"]. Single strings are not supported.'
        )

    if any(not isinstance(pattern, str) for pattern in patterns):
        raise TypeError(f"config.heptabase.{config_key} must contain only glob pattern strings.")

    return list(patterns)


class Executor:
    def __init__(self, config:DictConfig):
        self.config = config
        self.include_path_patterns = normalize_path_patterns(config.heptabase.include_path, "include_path")
        self.ignore_path_patterns = normalize_path_patterns(config.heptabase.ignore_path, "ignore_path")
        self.retrievers = {
            source: get_retriever_cls(source)(config) for source in config.executor.source
        }
        self.openai_client = OpenAI(api_key=config.llm.api.key, base_url=config.llm.api.base_url)
    def fetch_reading_history(self) -> list[CorpusPaper]:
        reading_history_path = self.config.heptabase.reading_history_path
        logger.info(f"Loading reading history from {reading_history_path}")
        records = load_reading_history(reading_history_path)
        logger.info(f"Loaded {len(records)} papers you have read")
        return [CorpusPaper(
            title=r.title,
            abstract=r.abstract,
            added_date=r.added_date,
            paths=r.paths,
        ) for r in records]

    def filter_reading_history(self, history:list[CorpusPaper]) -> list[CorpusPaper]:
        if self.include_path_patterns:
            logger.info(f"Selecting read papers matching include_path: {self.include_path_patterns}")
            history = [
                c for c in history
                if any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.include_path_patterns
                )
            ]
        if self.ignore_path_patterns:
            logger.info(f"Excluding read papers matching ignore_path: {self.ignore_path_patterns}")
            history = [
                c for c in history
                if not any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.ignore_path_patterns
                )
            ]
        if self.include_path_patterns or self.ignore_path_patterns:
            samples = random.sample(history, min(5, len(history)))
            samples = '\n'.join([c.title + ' - ' + '\n'.join(c.paths) for c in samples])
            logger.info(f"Selected {len(history)} papers you have read:\n{samples}\n...")
        return history


    def ingest_daily_sources(self, pool) -> None:
        """把當日檢索到的論文存進池子，而不是用完即丟。

        v1 每天寄完信就把候選丟掉，沒讀完的推薦隔天就消失。存進池子後，
        沒被選上的論文留在池中等下次機會，直到寫成卡片為止。
        """
        for source, retriever in self.retrievers.items():
            logger.info(f"Retrieving {source} papers...")
            papers = retriever.retrieve_papers()
            result = pool.ingest(
                PaperRecord(
                    id=self._pool_id(p),
                    source=p.source,
                    title=p.title,
                    abstract=p.abstract,
                    url=p.url,
                    pdf_url=p.pdf_url,
                    published_date=retriever.published_date(p),
                )
                for p in papers
            )
            logger.info(f"{source}: {len(papers)} retrieved, {result.inserted} new to the pool")

        journals = self.config.pool.get("openalex_sources") or []
        if journals:
            client = OpenAlexClient(mailto=self.config.pool.get("openalex_mailto") or "")
            # 沒有真正的「增量」可用：IEEE 的 publication_date 是期號日期（整年標同一天），
            # 而 OpenAlex 的 created/updated 日期過濾要付費方案。因此每天重抓整個
            # 當年度，靠池子的 id 與標題去重擋掉已經有的，只有真正新增的會寫入。
            since = publication_window_start(date.today(), self.config.pool.journal_lookback_days)
            for source_id in journals:
                works = list(client.fetch_works(source_id, since))
                result = pool.ingest(works)
                logger.info(f"{source_id}: {len(works)} fetched, {result.inserted} new to the pool")

    @staticmethod
    def _pool_id(paper:Paper) -> str:
        """把論文網址轉成池中的識別碼。"""
        return f"{paper.source}:{(paper.url or paper.title).rsplit('/', 1)[-1]}"

    @staticmethod
    def _to_paper(pooled) -> Paper:
        source = pooled.id.split(":", 1)[0]
        return Paper(
            source=source,
            title=pooled.title,
            authors=[],
            abstract=pooled.abstract,
            url=pooled.url,
            pdf_url=pooled.pdf_url,
            score=pooled.score,
        )

    def fetch_full_texts(self, papers:list[Paper]) -> None:
        """Download full text for the papers that survived selection.

        Only TLDR and affiliation generation read full_text, and only these few papers
        reach that stage, so fetching earlier would download a whole day of papers to
        throw most away. Sources without a retriever (journal articles behind a paywall)
        simply keep full_text None and fall back to the abstract.
        """
        logger.info(f"Downloading full text of {len(papers)} papers...")
        for paper in tqdm(papers, desc="Fetching full text"):
            retriever = self.retrievers.get(paper.source)
            if retriever is None:
                continue
            try:
                paper.full_text = retriever.fetch_full_text(paper)
            except Exception as exc:
                logger.warning(f"Failed to fetch full text of {paper.url}: {exc}")
                paper.full_text = None
            sleep(1)

    def run(self):
        history = self.fetch_reading_history()
        history = self.filter_reading_history(history)
        if len(history) == 0:
            logger.error(f"No reading history found. Please check your heptabase settings:\n{self.config.heptabase}")
            return

        pool = open_pool(self.config.pool.db_path)
        try:
            self.ingest_daily_sources(pool)

            unmatched = pool.mark_carded(history)
            if unmatched:
                logger.warning(
                    "These cards matched no paper in the pool, so those papers may be "
                    f"recommended again. Fix the 來源論文 line on each: {unmatched}"
                )

            # embedding 走 reranker.api.*（EMBEDDING_*），與 LLM 的憑證是分開的。
            embedder = get_embedder(self.config)
            computed = embed_pending(pool, embedder)
            logger.info(f"Embedded {computed} new papers; pool now {pool.stats()}")

            # time_decay_weights 假設順序是新到舊；不排序會讓最舊的卡片拿到最高權重。
            recent_first = sorted(history, key=lambda h: h.added_date, reverse=True)
            centroid = weighted_centroid(embedder.embed([h.abstract for h in recent_first]))
            selected = pool.select(
                centroid,
                n_new=self.config.pool.n_new,
                n_old=self.config.pool.n_old,
                new_within_days=self.config.pool.new_within_days,
                model=embedder.model,
            )
            logger.info(f"Selected {len(selected)} papers")
            if len(selected) == 0 and not self.config.executor.send_empty:
                logger.info("Nothing to recommend. No email will be sent.")
                return

            papers = [self._to_paper(p) for p in selected]
            self.fetch_full_texts(papers)
            logger.info("Generating TLDR and affiliations...")
            for p in tqdm(papers):
                p.generate_tldr(self.openai_client, self.config.llm)
                p.generate_affiliations(self.openai_client, self.config.llm)

            if self.config.executor.get("dry_run"):
                for p in papers:
                    logger.info(f"{p.score:7.3f}  {p.title}")
                logger.info("dry_run is on; no email sent and nothing marked as recommended.")
                return

            logger.info("Sending email...")
            send_email(self.config, render_email(papers))
            logger.info("Email sent successfully")
            # 寄出成功才計入推薦次數，否則寄信失敗會讓這些論文白白排到隊伍後方。
            pool.mark_recommended(selected)
        finally:
            pool.close()
