from loguru import logger
from omegaconf import DictConfig, ListConfig
from heptabase_reading_history import load_reading_history
from .utils import glob_match
from .retriever import get_retriever_cls
from .protocol import CorpusPaper, Paper
import random
from time import sleep
from .reranker import get_reranker_cls
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
        self.reranker = get_reranker_cls(config.executor.reranker)(config)
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


    def fetch_full_texts(self, papers:list[Paper]) -> None:
        """Download full text for the papers that survived reranking.

        Only TLDR and affiliation generation read full_text, and only these top papers reach
        that stage, so fetching earlier would download a whole day of papers to throw most away.
        """
        logger.info(f"Downloading full text of the top {len(papers)} papers...")
        for paper in tqdm(papers, desc="Fetching full text"):
            try:
                paper.full_text = self.retrievers[paper.source].fetch_full_text(paper)
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
        all_papers = []
        for source, retriever in self.retrievers.items():
            logger.info(f"Retrieving {source} papers...")
            papers = retriever.retrieve_papers()
            if len(papers) == 0:
                logger.info(f"No {source} papers found")
                continue
            logger.info(f"Retrieved {len(papers)} {source} papers")
            all_papers.extend(papers)
        logger.info(f"Total {len(all_papers)} papers retrieved from all sources")
        reranked_papers = []
        if len(all_papers) > 0:
            logger.info("Reranking papers...")
            reranked_papers = self.reranker.rerank(all_papers, history)
            reranked_papers = reranked_papers[:self.config.executor.max_paper_num]
            self.fetch_full_texts(reranked_papers)
            logger.info("Generating TLDR and affiliations...")
            for p in tqdm(reranked_papers):
                p.generate_tldr(self.openai_client, self.config.llm)
                p.generate_affiliations(self.openai_client, self.config.llm)
        elif not self.config.executor.send_empty:
            logger.info("No new papers found. No email will be sent.")
            return
        logger.info("Sending email...")
        email_content = render_email(reranked_papers)
        send_email(self.config, email_content)
        logger.info("Email sent successfully")
