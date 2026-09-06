from .base import BaseRetriever, register_retriever
import arxiv
from arxiv import Result as ArxivResult
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf, extract_tex_code_from_tar
from datetime import datetime, timedelta, timezone
from tempfile import TemporaryDirectory
import feedparser
from tqdm import tqdm
import multiprocessing
import os
from queue import Empty
from time import sleep
from typing import Any, Callable, TypeVar
from loguru import logger
import requests

T = TypeVar("T")

DOWNLOAD_TIMEOUT = (10, 60)
PDF_EXTRACT_TIMEOUT = 180
TAR_EXTRACT_TIMEOUT = 180


def _download_file(url: str, path: str) -> None:
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def _run_in_subprocess(
    result_queue: Any,
    func: Callable[..., T | None],
    args: tuple[Any, ...],
) -> None:
    try:
        result_queue.put(("ok", func(*args)))
    except Exception as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _run_with_hard_timeout(
    func: Callable[..., T | None],
    args: tuple[Any, ...],
    *,
    timeout: float,
    operation: str,
    paper_title: str,
) -> T | None:
    start_methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("fork" if "fork" in start_methods else start_methods[0])
    result_queue = context.Queue()
    process = context.Process(target=_run_in_subprocess, args=(result_queue, func, args))
    process.start()

    try:
        status, payload = result_queue.get(timeout=timeout)
    except Empty:
        if process.is_alive():
            process.kill()
        process.join(5)
        result_queue.close()
        result_queue.join_thread()
        logger.warning(f"{operation} timed out for {paper_title} after {timeout} seconds")
        return None

    process.join(5)
    result_queue.close()
    result_queue.join_thread()

    if status == "ok":
        return payload

    logger.warning(f"{operation} failed for {paper_title}: {payload}")
    return None


def _extract_text_from_pdf_worker(pdf_url: str) -> str:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.pdf")
        _download_file(pdf_url, path)
        return extract_markdown_from_pdf(path)


def _extract_text_from_html_worker(html_url: str) -> str | None:
    import trafilatura

    downloaded = trafilatura.fetch_url(html_url)
    if downloaded is None:
        raise ValueError(f"Failed to download HTML from {html_url}")
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not text:
        raise ValueError(f"No text extracted from {html_url}")
    return text


def _extract_text_from_tar_worker(source_url: str, paper_id: str, paper_title: str | None = None) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.tar.gz")
        _download_file(source_url, path)
        file_contents = extract_tex_code_from_tar(path, paper_id, paper_title=paper_title)
        if not file_contents or "all" not in file_contents:
            raise ValueError("Main tex file not found.")
        return file_contents["all"]


def build_date_range_query(categories: list[str], lookback_days: int, now: datetime | None = None) -> str:
    """Build an arXiv API query for *categories* submitted within the last *lookback_days* days.

    submittedDate is the original submission time, so this selects genuinely new papers --
    the API equivalent of the RSS feed's announce_type "new".
    """
    if lookback_days < 1:
        raise ValueError("source.arxiv.lookback_days must be at least 1.")
    end = now or datetime.now(timezone.utc)
    return build_window_query(categories, end - timedelta(days=lookback_days), end)


def build_window_query(categories: list[str], start, end) -> str:
    """Build an arXiv API query for *categories* submitted between *start* and *end*.

    Backfilling several years has to be chunked, because a single arXiv query caps the
    number of results it will page through. Taking an explicit window rather than a
    lookback lets the caller walk quarter by quarter.
    """
    if not categories:
        raise ValueError("category must be specified for arxiv.")
    if start > end:
        raise ValueError("start must not be after end.")
    cat_query = " OR ".join(f"cat:{c}" for c in categories)
    return f"({cat_query}) AND submittedDate:[{start:%Y%m%d%H%M} TO {end:%Y%m%d%H%M}]"


class _ArxivRef:
    """Stand-in for an arxiv.Result, built from a stored URL.

    Papers selected from the persistent pool were retrieved on an earlier run, so the
    API results cached in _raw_by_url are gone. The extraction helpers only need the
    source URL, entry id and title, and all three are derivable from the abs URL.
    """

    def __init__(self, entry_id: str, title: str, pdf_url: str | None):
        self.entry_id = entry_id
        self.title = title
        self.pdf_url = pdf_url or f"https://arxiv.org/pdf/{entry_id.rsplit('/', 1)[-1]}"

    def source_url(self) -> str:
        return f"https://arxiv.org/e-print/{self.entry_id.rsplit('/', 1)[-1]}"


def _arxiv_ref(paper: Paper) -> "_ArxivRef | None":
    url = paper.url or ""
    if "arxiv.org/abs/" not in url:
        return None
    return _ArxivRef(url, paper.title, paper.pdf_url)


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")
        # Full text is fetched lazily after reranking; keep the API results to fetch it from.
        self._raw_by_url: dict[str, ArxivResult] = {}

    def _retrieve_raw_papers(self) -> list[ArxivResult]:
        lookback_days = self.config.source.arxiv.get("lookback_days", None)
        if lookback_days and lookback_days > 1:
            return self._retrieve_by_date_range(lookback_days)
        return self._retrieve_from_rss()

    def _retrieve_by_date_range(self, lookback_days: int) -> list[ArxivResult]:
        """Retrieve papers submitted in the last *lookback_days* days via the arXiv API.

        The RSS feed only carries one announcement day, so a weekly run needs the API's
        submittedDate range instead. RSS marks cross-lists with announce_type; the API has
        no such field, so cross-lists are excluded by checking primary_category instead.
        """
        categories = list(self.config.source.arxiv.category)
        query = build_date_range_query(categories, lookback_days)
        logger.info(f"Querying arxiv API for the last {lookback_days} days: {query}")
        return self._run_query(query, categories)

    def _retrieve_by_date_range_between(self, start, end) -> list[ArxivResult]:
        """Retrieve papers submitted between two explicit datetimes. Used by backfill."""
        categories = list(self.config.source.arxiv.category)
        query = build_window_query(categories, start, end)
        logger.info(f"Querying arxiv API for {start} to {end}")
        return self._run_query(query, categories)

    def _run_query(self, query: str, categories: list[str]) -> list[ArxivResult]:
        client = arxiv.Client(num_retries=10, delay_seconds=10, page_size=100)
        include_cross_list = self.config.source.arxiv.get("include_cross_list", False)
        max_results = self.config.source.arxiv.get("max_results", 2000)

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )
        raw_papers = []
        seen = set()
        for result in client.results(search):
            if not include_cross_list and result.primary_category not in categories:
                continue
            if result.entry_id in seen:
                continue
            seen.add(result.entry_id)
            raw_papers.append(result)
        if len(raw_papers) >= max_results:
            logger.warning(
                f"Hit the max_results cap ({max_results}); older papers in the window were "
                "dropped. Narrow source.arxiv.category or lower source.arxiv.lookback_days."
            )
        logger.info(f"arxiv API returned {len(raw_papers)} papers")
        if self.config.executor.debug:
            raw_papers = raw_papers[:10]
        return raw_papers

    def _retrieve_from_rss(self) -> list[ArxivResult]:
        client = arxiv.Client(num_retries=10, delay_seconds=10)
        query = '+'.join(self.config.source.arxiv.category)
        include_cross_list = self.config.source.arxiv.get("include_cross_list", False)
        # Get the latest paper from arxiv rss feed
        feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
        if 'Feed error for query' in feed.feed.title:
            raise Exception(f"Invalid ARXIV_QUERY: {query}.")
        raw_papers = []
        allowed_announce_types = {"new", "cross"} if include_cross_list else {"new"}
        all_paper_ids = [
            i.id.removeprefix("oai:arXiv.org:")
            for i in feed.entries
            if i.get("arxiv_announce_type", "new") in allowed_announce_types
        ]
        if self.config.executor.debug:
            all_paper_ids = all_paper_ids[:10]

        # Get full information of each paper from arxiv api
        bar = tqdm(total=len(all_paper_ids))
        max_batch_retries = 5
        batch_retry_delay = 30
        for i in range(0, len(all_paper_ids), 20):
            search = arxiv.Search(id_list=all_paper_ids[i:i + 20])
            for attempt in range(max_batch_retries):
                try:
                    batch = list(client.results(search))
                    bar.update(len(batch))
                    raw_papers.extend(batch)
                    break
                except arxiv.HTTPError as exc:
                    if exc.status == 429 and attempt < max_batch_retries - 1:
                        wait = batch_retry_delay * (attempt + 1)
                        logger.warning(f"arXiv API 429 on batch {i // 20}, retry {attempt + 1}/{max_batch_retries} in {wait}s")
                        sleep(wait)
                    else:
                        raise
            if i + 20 < len(all_paper_ids):
                sleep(3)
        bar.close()

        return raw_papers

    def convert_to_paper(self, raw_paper: ArxivResult) -> Paper:
        # Full text is NOT downloaded here. Reranking only reads the abstract, so downloading
        # every candidate's LaTeX source or PDF up front costs ~14 s per paper for papers that
        # are usually discarded. fetch_full_text() below does it after reranking instead.
        self._raw_by_url[raw_paper.entry_id] = raw_paper
        return Paper(
            source=self.name,
            title=raw_paper.title,
            authors=[a.name for a in raw_paper.authors],
            abstract=raw_paper.summary,
            url=raw_paper.entry_id,
            pdf_url=raw_paper.pdf_url,
            full_text=None,
        )

    def published_date(self, paper: Paper) -> str | None:
        raw_paper = self._raw_by_url.get(paper.url)
        published = getattr(raw_paper, "published", None)
        return published.date().isoformat() if published else None

    def fetch_full_text(self, paper: Paper) -> str | None:
        raw_paper = self._raw_by_url.get(paper.url) or _arxiv_ref(paper)
        if raw_paper is None:
            logger.warning(f"Cannot derive an arxiv source URL for {paper.url}; no full text.")
            return None
        full_text = extract_text_from_tar(raw_paper)
        if full_text is None:
            full_text = extract_text_from_html(raw_paper)
        if full_text is None:
            full_text = extract_text_from_pdf(raw_paper)
        return full_text


def extract_text_from_html(paper: ArxivResult) -> str | None:
    html_url = paper.entry_id.replace("/abs/", "/html/")
    try:
        return _extract_text_from_html_worker(html_url)
    except Exception as exc:
        logger.warning(f"HTML extraction failed for {paper.title}: {exc}")
        return None


def extract_text_from_pdf(paper: ArxivResult) -> str | None:
    if paper.pdf_url is None:
        logger.warning(f"No PDF URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_pdf_worker,
        (paper.pdf_url,),
        timeout=PDF_EXTRACT_TIMEOUT,
        operation="PDF extraction",
        paper_title=paper.title,
    )


def extract_text_from_tar(paper: ArxivResult) -> str | None:
    source_url = paper.source_url()
    if source_url is None:
        logger.warning(f"No source URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_tar_worker,
        (source_url, paper.entry_id, paper.title),
        timeout=TAR_EXTRACT_TIMEOUT,
        operation="Tar extraction",
        paper_title=paper.title,
    )
