# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Babel is a Python pipeline that converts English podcasts into Chinese podcasts with multi-speaker voice cloning. The pipeline: transcribe (WhisperX) → extract reference audio → translate (DeepSeek/OpenAI) → synthesize (Qwen3-TTS/IndexTTS2) → concatenate into MP3.

## Environment Setup

- Python venv at `.venv/` (Python 3.12): use `.venv/bin/python` to run scripts
- No system `pip` — use `python -m pip install` instead
- `python` is not on PATH; use `python3` or `.venv/bin/python`
- System dependency: `ffmpeg` (required by pydub)

## Commands

```bash
# Run the full pipeline
.venv/bin/python babel.py /path/to/input.mp3

# Run tests
.venv/bin/pytest -q

# Run a single test file
.venv/bin/pytest tests/test_translate.py -q

# Run a single test
.venv/bin/pytest tests/test_translate.py::test_function_name -q

# Install dependencies
.venv/bin/python -m pip install -r requirements.txt
```

## Architecture

**Entry point:** `babel.py` — CLI orchestrator that runs the pipeline sequentially via argparse.

**Pipeline modules in `tools/`:**

| Module | Function | External Service |
|--------|----------|-----------------|
| `transcribe.py` | `transcribe()` — WhisperX speech-to-text + speaker diarization | WhisperX (local), HF_TOKEN for diarization |
| `reference_audio.py` | `extract_reference_audio()` — picks best 3-10s clip per speaker using quality scoring (SNR, speech ratio, loudness, clipping) | pydub/soundfile |
| `translate.py` | `translate_segments()` + `summarize_translated_segments()` — batch LLM translation (20 segs/call) with numbered-line parsing | DeepSeek or OpenAI API |
| `synthesize.py` | `synthesize_segments()` — voice-cloned TTS | IndexTTS2 (default) or Qwen3-TTS |
| `concatenate.py` | `concatenate_audio()` — assembles clips with gap calculation (100ms–3000ms bounds from original timing) | pydub |
| `youtube_download.py` | `download_youtube_mp3()` — validates YouTube URLs and downloads via yt-dlp | yt-dlp |

**Device selection** (`tools/__init__.py`): CUDA → MPS → CPU. WhisperX only supports CUDA/CPU, so MPS falls back to CPU for transcription.

**Work directory:** `data/<input_name>_babel/` stores intermediate files (transcription.json, translation.json, ref_audio/, tts_clips/).

## Environment Variables

- `HF_TOKEN` — Hugging Face token for speaker diarization (optional; diarization skipped if absent)
- `DEEPSEEK_API_KEY` — required when using `--translation-provider deepseek` (default)
- `OPENAI_API_KEY` — required when using `--translation-provider openai`
- `HF_ENDPOINT` — optional HuggingFace mirror URL

## Key Implementation Details

- `babel.py` patches `torch.load` to force `weights_only=False` (PyTorch 2.6+ broke pyannote checkpoint loading)
- Translation uses numbered-line format for batch parsing; falls back to original text on parse failure
- Reference audio scoring: speech_ratio (35%), SNR (25%), loudness (15%), duration preference (15%), clipping penalty (-25%); composes multiple short clips if no 3-10s segment exists

## Testing Patterns

- Framework: pytest with `unittest.mock`
- External modules (whisperx, openai, yt_dlp) are mocked via `sys.modules` injection + `importlib.reload()` on the target module
- OpenAI client: patch `OpenAI` at module level with `monkeypatch.setattr()` rather than injecting a fake module (due to `cached_property` on `.chat`)
- Audio test data: generated synthetically with `pydub.generators.Sine`
