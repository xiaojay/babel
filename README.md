# Babel

把英语播客转换为中文播客的端到端流水线：

1. WhisperX 转录 + 说话人分离
2. LLM 翻译（DeepSeek / OpenAI GPT-5 mini）
3. 声音克隆合成（Qwen3-TTS / IndexTTS2）
4. 拼接为单一 MP3

本仓库包含完整 CLI、分步工具模块以及单元测试，用于将英文播客音频翻译成中文并保留多说话人声线。

## 目录结构

- `babel.py`：主入口 CLI，串联完整流水线
- `tools/`：流水线各步骤模块
- `tests/`：单元测试
- `requirements.txt`：Python 依赖
- `clawdbot*.mp3`：示例音频
- `clawdbot_babel/`、`clawdbot_5min_babel/`：示例中间产物目录

## 依赖与环境

Python 版本建议 3.10+，并需要以下系统能力：

- `ffmpeg`：`pydub` 处理 MP3/WAV 需要，未安装会导致音频读写失败
- GPU 可选：CUDA 可显著加速 WhisperX、Qwen3-TTS、IndexTTS2

Python 依赖（见 `requirements.txt`）：

- `whisperx`
- `qwen-tts`（来自 `https://github.com/QwenLM/Qwen3-TTS`）
- `indextts`（默认后端 `indextts2` 需要，来自 `https://github.com/index-tts/index-tts`）
- `openai`
- `pydub`
- `python-dotenv`
- `torch`
- `soundfile`
- `yt-dlp`
- `pytest`

## 环境变量

- `HF_TOKEN`：Hugging Face Token，用于 WhisperX 说话人分离。
  - 未设置时会跳过说话人分离，所有片段标记为 `SPEAKER_00`。
- `DEEPSEEK_API_KEY`：DeepSeek API Key，用于翻译。
- `OPENAI_API_KEY`：OpenAI API Key（当 `--translation-provider openai` 时用于翻译）。

可在项目根目录放置 `.env` 文件，`babel.py` 会自动读取。

## 快速开始

1. 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple -r requirements.txt || pip install -r requirements.txt
```

默认后端是 `IndexTTS2`（`--tts-backend indextts2`），需按官方仓库安装 `indextts` 并下载模型到 `checkpoints/`（或自定义目录）：

```bash
pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple -U uv
git clone https://github.com/index-tts/index-tts.git
cd index-tts
uv sync --all-extras --default-index "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
uv tool install "huggingface-hub[cli,hf_xet]"
hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints
```

若暂时不安装 `indextts`，运行时请显式指定 `--tts-backend qwen3`。

2. 设置环境变量

```bash
export HF_TOKEN=your_hf_token
export DEEPSEEK_API_KEY=your_deepseek_key
export OPENAI_API_KEY=your_openai_key
```

如果你的环境使用了 SOCKS 代理（例如设置了 `ALL_PROXY=socks5://...`），还需要安装：

```bash
pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple socksio
```

3. 运行

```bash
python babel.py /path/to/input.mp3
```

输入也可以是 YouTube 链接：

```bash
python babel.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

仅下载 YouTube 音频为 MP3（不走翻译流水线）：

```bash
python babel.py "https://www.youtube.com/watch?v=VIDEO_ID" --download-only -o podcast.mp3
```

默认输出在项目 `data/` 目录下，文件名为输入同名加 `_zh.mp3` 后缀。

已验证示例（`IndexTTS2`，默认保留中间文件）：

```bash
python babel.py clawdbot_5min.mp3 --tts-backend indextts2 -o clawdbot_5min_zh.mp3
```

## CLI 用法

`babel.py` 的参数：

- `input`（必填）：输入英文播客 MP3，或 YouTube 链接
- `-o, --output`：输出文件路径（默认在 `data/` 下生成 `input_zh.mp3`；`--download-only` 时为下载的 MP3）
- `--whisper-model`：Whisper 模型大小（默认 `large-v3`）
- `--translation-provider`：翻译提供方（`deepseek` 或 `openai`，默认 `deepseek`）
- `--translation-model`：翻译模型名（默认随提供方自动选择：`deepseek-chat` 或 `gpt-5-mini`）
- `--tts-backend`：语音合成后端（`qwen3` 或 `indextts2`，默认 `indextts2`）
- `--index-tts-model-dir`：IndexTTS2 模型目录（默认 `checkpoints`）
- `--index-tts-cfg-path`：IndexTTS2 配置路径（默认 `<index-tts-model-dir>/config.yaml`）
- `--concatenate-without-timestamps`：第 5 步拼接时忽略时间戳，不额外插入停顿
- `--concatenate-fixed-gap-ms MS`：第 5 步拼接时忽略时间戳，并在片段间插入固定停顿（毫秒）
- `--keep-intermediate`：保留中间文件（默认）
- `--no-keep-intermediate`：不保留中间文件（流程结束后自动清理）
- `--download-only`：仅下载 YouTube 音频为 MP3 后退出

示例：

```bash
python babel.py input.mp3 --whisper-model medium -o output_zh.mp3
python babel.py input.mp3 --translation-provider openai --translation-model gpt-5-mini -o output_zh.mp3
python babel.py input.mp3 --tts-backend indextts2 --index-tts-model-dir /path/to/checkpoints
python babel.py input.mp3 --concatenate-without-timestamps -o output_zh.mp3
python babel.py input.mp3 --concatenate-fixed-gap-ms 180 -o output_zh.mp3
python babel.py clawdbot_5min.mp3 --tts-backend indextts2 -o clawdbot_5min_zh.mp3
python babel.py "https://youtu.be/VIDEO_ID" --download-only -o source.mp3
```

## 流水线细节

### 1. 转录与说话人分离（`tools/transcribe.py`）

- 使用 WhisperX 转录，并进行字级时间对齐。
- 如果设置了 `HF_TOKEN`，启用说话人分离并为片段标注 `speaker`。
- 设备选择逻辑：CUDA 优先，其次 MPS，最后 CPU。
- 在 `babel.py` 中对 `torch.load` 做兼容性补丁，以适配 pyannote 检查点。

输出片段格式示例：

```json
{"start": 0.0, "end": 1.5, "text": "Hello", "speaker": "SPEAKER_00"}
```

### 2. 参考音频提取（`tools/reference_audio.py`）

- 按说话人分组，从原音频中选取 3-10 秒的片段作为参考音频。
- 优先选择 3-10 秒区间内最长片段；若无合格片段则回退到最长片段。
- 片段超过 10 秒会截断。

输出：`{speaker_id: ref_wav_path}`

### 3. 翻译（`tools/translate.py`）

- 支持两种翻译提供方：
  - `deepseek`（默认）：`openai` SDK + `base_url=https://api.deepseek.com`，默认模型 `deepseek-chat`
  - `openai`：默认模型 `gpt-5-mini`
- 分批翻译，每批默认 20 段。
- 解析返回文本中的编号，写入 `text_zh` 字段。
- 若解析失败则保留原文。

### 4. 声音克隆合成（`tools/synthesize.py`）

- 后端一：`indextts2`（默认）
  - 使用 `indextts.infer_v2.IndexTTS2` 逐段推理。
  - 每个片段按 `speaker` 选择对应参考音频（缺失时回退到首个参考音频）。
  - 默认读取 `<index-tts-model-dir>/config.yaml`，可用 `--index-tts-cfg-path` 覆盖。
  - 首次运行会自动下载额外模型到 `<index-tts-model-dir>/hf_cache`（例如 `facebook/w2v-bert-2.0`、`amphion/MaskGCT`、`funasr/campplus`、`nvidia/bigvgan_v2_22khz_80band_256x`）。
  - 若访问 HuggingFace 较慢，可先设置：`export HF_ENDPOINT=https://hf-mirror.com`。
- 后端二：`qwen3`
  - 使用 `Qwen/Qwen3-TTS-12Hz-1.7B-Base` 进行中文语音合成。
  - 每个说话人提取一次 voice clone prompt，复用到所有同说话人片段。
  - 设备逻辑：
  - CPU：`float32` + `sdpa`
  - MPS：`bfloat16` + `sdpa`
  - CUDA：`bfloat16` + 优先 `flash_attention_2`，不可用则 `sdpa`

### 5. 拼接输出（`tools/concatenate.py`）

- 按原始片段时间间隔插入静音。
- gap 最小 100ms、最大 3000ms。
- 输出 MP3，码率 192k。

## 中间产物

默认会生成一个与输入同目录、同名加 `_babel` 后缀的目录（可用 `--no-keep-intermediate` 关闭），例如：

- `transcription.json`：转录结果
- `translation.json`：翻译结果
- `ref_audio/`：每个说话人的参考音频 WAV
- `tts_clips/`：每段合成后的 WAV

JSON 格式：

```json
{
  "segments": [
    {"start": 0.0, "end": 1.5, "text": "Hello", "speaker": "SPEAKER_00", "text_zh": "你好"}
  ]
}
```

## 测试

```bash
pytest -q
```

测试覆盖范围：

- WhisperX 调用流程与说话人分离逻辑
- 参考音频选取与裁剪规则
- 翻译批处理与编号解析
- Qwen3-TTS 与 IndexTTS2 后端选择、参数与调用流程
- 拼接间隔计算与 MP3 输出
- YouTube 链接识别与 MP3 下载工具

## 常见问题

- 没有 `HF_TOKEN`：说话人分离会跳过，只有 `SPEAKER_00`。
- 翻译失败：确认已按 `--translation-provider` 设置对应密钥（`DEEPSEEK_API_KEY` 或 `OPENAI_API_KEY`）。
- 代理环境报错 `Using SOCKS proxy, but the 'socksio' package is not installed`：安装 `socksio`（见“快速开始”）。
- `pydub` 报错：确认 `ffmpeg` 已安装并可在 PATH 中找到。
- CUDA/MPS 性能不佳：可切换 `--whisper-model` 为更小模型以减少显存占用。
- 使用 `indextts2` 报 `ModuleNotFoundError`：先按上文安装 `indextts`，并确认模型目录存在 `config.yaml` 与权重文件。
- `indextts2` 报 `PytorchStreamReader ... archive is corrupted`：通常是模型缓存下载中断，删除损坏缓存后重下：

```bash
mv checkpoints/hf_cache/models--nvidia--bigvgan_v2_22khz_80band_256x checkpoints/hf_cache/models--nvidia--bigvgan_v2_22khz_80band_256x.bad
mv checkpoints/hf_cache/.locks/models--nvidia--bigvgan_v2_22khz_80band_256x checkpoints/hf_cache/.locks/models--nvidia--bigvgan_v2_22khz_80band_256x.bad
HF_ENDPOINT=https://hf-mirror.com hf download nvidia/bigvgan_v2_22khz_80band_256x --cache-dir ./checkpoints/hf_cache
```

## 设计与注意事项

- `torch.load` 被强制 `weights_only=False`，避免 pyannote checkpoint 反序列化失败。
- WhisperX 在 MPS 上不支持，需要自动回退到 CPU。
- 翻译结果按行解析，建议模型输出严格对应编号。
- `qwen3` 后端使用每个说话人首个片段文本作为 reference text；`indextts2` 后端直接使用参考音频进行零样本克隆。

## 许可

仓库未包含明确许可证信息。如需公开使用或分发，请先确认授权与依赖许可要求。
