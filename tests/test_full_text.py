"""全文只有排序後的前 N 篇才需要，因此要在排序之後才下載。"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from zotero_arxiv_daily.executor import Executor  # noqa: E402
from zotero_arxiv_daily.protocol import Paper  # noqa: E402
from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever  # noqa: E402


def arxiv_result(entry_id="https://arxiv.org/abs/1", title="T"):
    return SimpleNamespace(
        entry_id=entry_id,
        title=title,
        authors=[SimpleNamespace(name="A. Author")],
        summary="An abstract.",
        pdf_url="https://arxiv.org/pdf/1",
        primary_category="eess.SP",
    )


def arxiv_retriever():
    config = OmegaConf.create(
        {
            "source": {"arxiv": {"category": ["eess.SP"], "include_cross_list": False}},
            "executor": {"debug": False},
        }
    )
    return ArxivRetriever(config)


class ConvertDoesNotDownloadTests(unittest.TestCase):
    def setUp(self):
        self.retriever = arxiv_retriever()

    def test_convert_leaves_full_text_empty(self):
        with (
            patch("zotero_arxiv_daily.retriever.arxiv_retriever.extract_text_from_tar") as tar,
            patch("zotero_arxiv_daily.retriever.arxiv_retriever.extract_text_from_html") as html,
            patch("zotero_arxiv_daily.retriever.arxiv_retriever.extract_text_from_pdf") as pdf,
        ):
            paper = self.retriever.convert_to_paper(arxiv_result())
        self.assertIsNone(paper.full_text)
        # 這三個下載函式在排序前一次都不該被呼叫。
        tar.assert_not_called()
        html.assert_not_called()
        pdf.assert_not_called()

    def test_fetch_full_text_uses_the_cached_result(self):
        self.retriever.convert_to_paper(arxiv_result())
        paper = Paper(
            source="arxiv", title="T", authors=[], abstract="a", url="https://arxiv.org/abs/1"
        )
        with patch(
            "zotero_arxiv_daily.retriever.arxiv_retriever.extract_text_from_tar",
            return_value="tex body",
        ):
            self.assertEqual(self.retriever.fetch_full_text(paper), "tex body")

    def test_fetch_falls_back_through_tar_html_pdf(self):
        self.retriever.convert_to_paper(arxiv_result())
        paper = Paper(
            source="arxiv", title="T", authors=[], abstract="a", url="https://arxiv.org/abs/1"
        )
        with (
            patch(
                "zotero_arxiv_daily.retriever.arxiv_retriever.extract_text_from_tar",
                return_value=None,
            ),
            patch(
                "zotero_arxiv_daily.retriever.arxiv_retriever.extract_text_from_html",
                return_value=None,
            ),
            patch(
                "zotero_arxiv_daily.retriever.arxiv_retriever.extract_text_from_pdf",
                return_value="pdf body",
            ),
        ):
            self.assertEqual(self.retriever.fetch_full_text(paper), "pdf body")

    def test_unknown_url_returns_none_instead_of_raising(self):
        paper = Paper(source="arxiv", title="T", authors=[], abstract="a", url="unknown")
        self.assertIsNone(self.retriever.fetch_full_text(paper))


class ExecutorFetchTests(unittest.TestCase):
    def executor(self, retriever):
        executor = Executor.__new__(Executor)
        executor.retrievers = {"arxiv": retriever}
        return executor

    def papers(self, n):
        return [
            Paper(source="arxiv", title=f"T{i}", authors=[], abstract="a", url=f"u{i}")
            for i in range(n)
        ]

    def test_only_the_given_papers_are_fetched(self):
        calls = []

        class Fake:
            def fetch_full_text(self, paper):
                calls.append(paper.url)
                return f"text-{paper.url}"

        papers = self.papers(3)
        with patch("zotero_arxiv_daily.executor.sleep"):
            self.executor(Fake()).fetch_full_texts(papers)
        self.assertEqual(calls, ["u0", "u1", "u2"])
        self.assertEqual(papers[0].full_text, "text-u0")

    def test_one_failure_does_not_abort_the_rest(self):
        class Flaky:
            def fetch_full_text(self, paper):
                if paper.url == "u0":
                    raise RuntimeError("network down")
                return "ok"

        papers = self.papers(2)
        with patch("zotero_arxiv_daily.executor.sleep"):
            self.executor(Flaky()).fetch_full_texts(papers)
        self.assertIsNone(papers[0].full_text)
        self.assertEqual(papers[1].full_text, "ok")


if __name__ == "__main__":
    unittest.main()
