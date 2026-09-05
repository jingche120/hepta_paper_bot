import glob
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from zotero_arxiv_daily.utils import _translate_glob, glob_match  # noqa: E402

CASES = [
    ("LLM", "LLM", True),
    ("LLM", "llm", False),
    ("LLM/RAG", "LLM/*", True),
    ("LLM/RAG/Eval", "LLM/*", False),  # `*` 不跨路徑分隔
    ("LLM/RAG/Eval", "LLM/**", True),  # `**` 跨層
    ("LLM/RAG/Eval", "**/Eval", True),
    ("2026/survey/a", "2026/**/a", True),
    ("2026/a", "2026/**/a", True),  # `**/` 也可匹配零層
    ("archive/old", "archive/**", True),
    ("archived", "archive/**", False),
    ("LLM/RAG", "LLM/RA?", True),
    ("LLM/RAGX", "LLM/RA?", False),
    ("a.b", "a.b", True),
    ("axb", "a.b", False),  # `.` 是字面字元，不是萬用字元
]


class TranslateGlobTests(unittest.TestCase):
    def test_backport_matches_expected_cases(self):
        for path, pattern, expected in CASES:
            with self.subTest(path=path, pattern=pattern):
                with patch.object(glob, "translate", None, create=True):
                    self.assertEqual(glob_match(path, pattern), expected)

    def test_backport_agrees_with_stdlib_when_available(self):
        if not hasattr(glob, "translate"):
            self.skipTest("stdlib glob.translate 需要 Python 3.13 以上")
        for path, pattern, _ in CASES:
            with self.subTest(path=path, pattern=pattern):
                self.assertEqual(
                    glob_match(path, pattern),
                    __import__("re").match(_translate_glob(pattern), path) is not None,
                )

    def test_pattern_must_match_whole_path(self):
        with patch.object(glob, "translate", None, create=True):
            self.assertFalse(glob_match("LLM/RAG", "LLM"))


if __name__ == "__main__":
    unittest.main()
