# Repository Guidelines

## Project Status

A daily paper recommendation bot. The reading history (this project's term for the interest corpus) comes from the Heptabase whiteboard 「讀過的論文總覽們」, where each card is one paper overview; the bot retrieves new papers from arXiv and similar sources, ranks them by embedding similarity against that reading history, and emails the list. Git resolves to a parent repository; scope changes to this project directory and avoid staging sibling projects.

## Project Structure & Module Organization

Two layers, deliberately kept apart:

- **Project root** — the Heptabase side. `get_hepthabase_bot.py` (entry point; `--export` writes the reading-history file), `heptabase_oauth.py` (browser authorization), `heptabase_keychain.py`, `heptabase_cards.py` (paginated MCP reads plus whiteboard card listing), `heptabase_reading_history.py` (overview-card section parsing and the JSON reading-history file; pure functions, no network).
- **`src/zotero_arxiv_daily/`** — a verbatim copy of [TideDra/zotero-arxiv-daily](https://github.com/TideDra/zotero-arxiv-daily). **Keep it byte-for-byte close to upstream** so it stays diffable and mergeable: do not reformat, rename, or restyle it, and confine edits to what the Heptabase swap actually requires. Ruff excludes this directory for the same reason.

`main.py` at the root is the pipeline entry point; it only fixes `sys.path` and delegates. `config/` holds the Hydra config (`base.yaml` documents every option; `custom.yaml` is local and untracked). `data/` holds the exported reading history and is untracked. `tests/` mirrors both layers.

Keep entry points thin.

## Build, Test, and Development Commands

Use Python 3.11+ (the current `.venv` is 3.12; note that `glob.translate` is 3.13-only, so `utils.glob_match` carries a backport — keep both paths tested). Create a virtual environment with `python3 -m venv .venv`, then install with `.venv/bin/python -m pip install -r requirements.txt` (add `-r requirements-reranker.txt` for the local embedding model). Run `.venv/bin/python get_hepthabase_bot.py` to complete browser OAuth and verify MCP initialization, `.venv/bin/python get_hepthabase_bot.py --export` to refresh the reading history, and `.venv/bin/python main.py` for the recommendation run. Run `.venv/bin/python -m unittest discover -s tests -v` for offline tests.

## Coding Style & Naming Conventions

Follow the selected language’s standard formatter and indentation rules; avoid mixing styles. Use descriptive module and function names, keep functions focused, and isolate external-service integrations. Run `.venv/bin/ruff check .` and `.venv/bin/ruff format .`; rules live in `pyproject.toml`. The upstream copy under `src/zotero_arxiv_daily/` is exempt — do not reformat it.

## Testing Guidelines

Use standard-library `unittest`, with `test_*.py` files and `test_*` methods. No coverage threshold exists. Mock network calls; do not initiate real OAuth during automated verification, and do not download embedding models in tests. Cover ranking and duplicate handling as those features are added.

Card parsing (`test_reading_history.py`) is the highest-risk area: only `abstract` is embedded, so a parsing slip silently degrades every recommendation without raising. When changing `parse_overview_card`, add a case for the format it newly accepts, and keep the tests asserting that non-abstract sections (方法概覽, 主要結果, …) stay *out* of the abstract and that `read_object` artifacts (the `card "…" [id]` header line, `N\t` line-number prefixes, `<hepta-*>` tags) are stripped.

Our code says **reading history**, not corpus; the only exception is upstream's `CorpusPaper` dataclass in `protocol.py`, kept under its original name so `src/` stays diffable against upstream.

MCP tool responses are human-readable text, not structured JSON — `list_cards` returns lines like `card "TITLE" [uuid] created: ISO; updated: ISO`, parsed by `parse_card_lines`. Treat that shape as a real dependency: if Heptabase changes the wording, listing silently returns zero cards.

## Commit & Pull Request Guidelines

There is no existing commit convention. Use concise, imperative subjects such as `Add paper retrieval adapter`. Keep commits focused. Pull requests should describe the change, include validation results, link relevant issues, and identify configuration changes. Include screenshots only for visual changes.

## Security & Agent Instructions

Respond to the user in Traditional Chinese. Never read `.env` or other private `.env.*` files, directly or through scripts, subprocesses, logs, or environment dumps. Only `.env.example` may contain public configuration placeholders. Do not load private environment files during agent-run verification.

Keep credentials out of source control; document required variables using placeholders in `.env.example`. Exclude local secrets and generated data from Git.

Use `$...$` for inline mathematics and `$$...$$` for display mathematics, with single-backslash LaTeX commands. Place Heptabase-ready notes in a Markdown code block so raw formulas remain copyable.
