# Repository Guidelines

## Project Structure & Module Organization

Core Python code lives in `ultralytics/`. Model implementations are grouped under `ultralytics/models/`, reusable neural-network components under `ultralytics/nn/`, and training, validation, prediction, and export workflows under `ultralytics/engine/`. Configuration files and dataset/model YAMLs belong in `ultralytics/cfg/`; packaged sample media belongs in `ultralytics/assets/`. Put unit and integration coverage in `tests/`, documentation in `docs/`, and standalone demonstrations in `examples/`. Treat generated directories such as `runs/`, `build/`, and `dist/` as disposable outputs, not source.

## Build, Test, and Development Commands

- `python -m pip install -e ".[dev]"` installs the package in editable mode with test and documentation dependencies.
- `pytest tests/` runs the standard suite; slow tests are excluded by default.
- `pytest --slow tests/` includes tests marked `slow`. Target a file while iterating, for example `pytest tests/test_engine.py -v`.
- `pytest --cov=ultralytics --cov-report=term-missing tests/` reports coverage for the package.
- `ruff check .` performs lint checks; `ruff format .` applies the repository formatter.
- `python docs/build_docs.py` builds documentation and checks it for warnings.
- `python -m build` creates source and wheel distributions after installing `build`.

## Coding Style & Naming Conventions

Use four-space indentation and a 120-character line limit. Follow standard Python naming: `snake_case` for functions, variables, and modules; `PascalCase` for classes; and `UPPER_CASE` for constants. Add type hints where they clarify interfaces. Write concise Google-style docstrings with complete-sentence summaries. Ruff controls formatting and linting; configuration also defines isort-compatible import ordering. Prefer small, reusable changes and preserve backward compatibility.

## Testing Guidelines

Pytest configuration is in `pyproject.toml`, with shared fixtures and the `--slow` option in `tests/conftest.py`. Name tests `test_<behavior>` and place new cases in the closest existing `test_*.py` file rather than creating a new file unnecessarily. Cover fixes with a regression test and avoid relying on network downloads unless the integration specifically requires them. CI must pass, but no fixed percentage threshold is declared.

## Commit & Pull Request Guidelines

Recent commits use concise, imperative subjects such as `Fix ...`, `Improve ...`, and `Add ...`, often followed by a PR number. Keep each commit and PR focused. PRs should target `main`, explain purpose and scope, link relevant issues, include tests and documentation updates, and provide screenshots or logs for visible behavior changes. Feature work requires a discussed, approved issue. All CI checks and automated-review comments must be addressed, and contributors must sign the CLA before merge.
