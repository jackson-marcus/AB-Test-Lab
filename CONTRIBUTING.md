# Contributing to ABTestLab

## Setup

Python 3.12 is required (see `.python-version`).

```bash
uv sync --group dev
```

On Windows you can use `.\tasks.ps1 install` instead of the Makefile.

Install git hooks (optional but recommended):

```bash
uv run pre-commit install
```

## Checks

Run the same jobs CI runs:

```bash
uv run ruff check .
uv run ruff format .
uv run pytest --cov
```

Or `make lint` / `make format` / `make test` (PowerShell: `.\tasks.ps1 lint`, etc.).

Do not add Black; formatting is **ruff format**.

## Pull requests

- Target `main`.
- Keep changes focused. Do not invent statistical methods or replace the batch DAG skeleton without discussion.
- If you touch `abtestlab.functional` vs `abtestlab.stats.core`, remember they are **two implementations**. The API uses `stats.core`; the library example uses `functional`. Their mSPRT formulas are not the same.
- Add tests for bug fixes (including a regression assertion when you found a specific defect).

## License

Contributions are accepted under the Apache License 2.0. See [LICENSE](LICENSE).
