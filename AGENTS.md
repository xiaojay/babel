# Repository Guidelines

## Project Structure & Module Organization
`babel.py` is the CLI entrypoint that orchestrates the full pipeline. Keep stage-specific logic in `tools/` modules (`transcribe.py`, `reference_audio.py`, `translate.py`, `synthesize.py`, `concatenate.py`, `youtube_download.py`) instead of growing the CLI.  
`tests/` contains pytest unit tests, generally one file per tool module (for example, `tests/test_translate.py`).  
Use `scripts/` for one-off utilities and comparisons.  
Treat `data/`, `log/`, and `checkpoints/` as runtime/model artifacts, not core source.  
`doc/` stores operational notes.

## Build, Test, and Development Commands
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python babel.py input.mp3 -o data/input_zh.mp3
python babel.py "https://youtu.be/VIDEO_ID" --download-only -o data/source.mp3
pytest -q
pytest tests/test_translate.py -q
```
Use the first three commands to set up a local environment. Run `babel.py` for end-to-end pipeline checks, and use `pytest` for fast regression validation.

## Coding Style & Naming Conventions
Target Python 3.10+ style with 4-space indentation and PEP 8 spacing.  
Use `snake_case` for functions, variables, and module names; use `PascalCase` for test classes.  
Prefer small, composable functions in `tools/` and keep file/network side effects explicit.  
Add type hints for new public functions and keep imports grouped (stdlib, third-party, local).  
There is no committed lint/format config; follow existing style consistently.

## Testing Guidelines
Use `pytest` with `unittest.mock` and `monkeypatch`.  
Name tests `test_<behavior>` and place them in `tests/test_<module>.py`.  
Mock external APIs and heavy dependencies (OpenAI, WhisperX, yt-dlp, TTS backends) so unit tests remain offline and fast.  
For each behavior change, add coverage for both the main path and at least one fallback/error path.  
Run `pytest -q` before opening a PR.

## Commit & Pull Request Guidelines
Recent history uses short, imperative commit subjects (for example, `add summary`, `default output dir data`).  
Keep commit messages concise and action-oriented, with one logical change per commit.  
PRs should include the problem statement, key pipeline-stage changes, test commands executed, and any new flags/env vars.  
Link related issues when available and include an example CLI invocation for behavior changes.

## Security & Configuration Tips
Store secrets in `.env` and never commit API keys.  
Avoid committing generated audio/model outputs from `data/`, `log/`, or `checkpoints/` unless intentionally adding fixtures.
