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

## 静态播客网站生成器

`site.py` 是独立于翻译管道的 CLI 工具，用于将 `babel.py` 的产出（中文 MP3、摘要文件）发布为可通过播客客户端订阅的静态网站。

### 命令

```bash
# 初始化站点（创建 site/ 目录、config.json、episodes.json）
.venv/bin/python site.py init --title "我的播客" --base-url "https://example.com/podcast"

# 添加一集（从 babel.py 产出文件）
.venv/bin/python site.py add \
  --title "Episode Title" \
  --zh-audio data/input_zh.mp3 \
  --en-audio input.mp3 \
  --summary data/input_zh.summary.txt \
  --detailed-summary data/input_zh.summary.detailed.md

# 生成静态站点（输出到 site/build/）
.venv/bin/python site.py build

# 本地预览（默认 http://localhost:8000）
.venv/bin/python site.py serve [--port 8000]
```

### 站点目录结构

```
site/
  config.json           # 站点配置（标题、base_url、作者等）
  episodes.json         # 剧集元数据列表
  audio/<slug>/         # 每集音频文件（zh.mp3、en.mp3）
  build/                # 构建产出（HTML、RSS、CSS）
    index.html          # 首页（剧集列表 + 播放器）
    episodes/<slug>.html  # 单集详情页
    feed.xml            # iTunes 兼容 RSS 订阅源
    style.css           # 样式表
    audio -> ../audio   # 符号链接
```

### 模块

| 模块 | 功能 |
|------|------|
| `site_tools/config.py` | 站点初始化、config.json / episodes.json 读写 |
| `site_tools/episodes.py` | 添加剧集：复制音频、用 pydub 计算时长/大小、读取摘要文件、更新 episodes.json |
| `site_tools/build.py` | Jinja2 渲染 HTML + RSS，复制 CSS，创建音频符号链接 |
| `site_tools/serve.py` | 基于 `http.server` 的本地预览服务器 |

### 数据模型

**config.json：**
- `title` — 站点标题
- `description` — 站点描述
- `author` — 作者
- `base_url` — 部署后的完整 URL（用于 RSS `<enclosure>` 和页面链接）
- `language` — 语言代码（默认 `zh-cn`）
- `cover_url` — 播客封面图 URL（可选）

**episodes.json** 每条记录：
- `slug` — URL 标识符（从标题自动生成，或 `--slug` 手动指定）
- `title` — 剧集标题
- `pub_date` — 发布日期（YYYY-MM-DD）
- `zh_audio` / `en_audio` — 音频相对路径
- `zh_audio_size_bytes` / `zh_audio_duration_seconds` — RSS `<enclosure>` 所需元数据
- `summary` — 简短摘要（内联文本）
- `detailed_summary_md` — 详细摘要（内联 Markdown，构建时渲染为 HTML）

### 模板与样式

- 模板位于 `site_tools/templates/`，使用 Jinja2，基础布局为 `base.html`
- RSS 模板（`rss.xml`）包含 iTunes 命名空间（`itunes:author`、`itunes:duration`、`itunes:image`）和 Atom self link
- CSS 位于 `site_tools/static/style.css`：中文字体栈（PingFang SC → Hiragino Sans GB → Microsoft YaHei → system-ui）、800px 居中、卡片式列表、响应式布局
- Jinja2 自定义 filter：`format_duration`（秒→时:分:秒）、`rfc2822`（日期→RFC 2822）、`markdown`（MD→HTML）

### 依赖

`jinja2`、`markdown`（已加入 requirements.txt）。`pydub` 仅在 `add` 命令中延迟导入。

## Testing Patterns

- Framework: pytest with `unittest.mock`
- External modules (whisperx, openai, yt_dlp) are mocked via `sys.modules` injection + `importlib.reload()` on the target module
- OpenAI client: patch `OpenAI` at module level with `monkeypatch.setattr()` rather than injecting a fake module (due to `cached_property` on `.chat`)
- Audio test data: generated synthetically with `pydub.generators.Sine`
