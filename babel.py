#!/usr/bin/env python3
"""Babel - 英语播客转中文播客

Pipeline: WhisperX STT + Diarization → LLM Translation (DeepSeek / OpenAI) → Voice Clone (Qwen3 / IndexTTS2) → MP3
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from dotenv import load_dotenv

# PyTorch >=2.6 defaults weights_only=True in torch.load, but pyannote
# checkpoints contain omegaconf objects that fail strict unpickling.
# Force weights_only=False for compatibility.
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

load_dotenv()

from tools import (
    transcribe,
    extract_reference_audio,
    translate_segments,
    synthesize_segments,
    concatenate_audio,
    download_youtube_mp3,
    is_youtube_url,
)


def save_intermediate(data: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Babel - 英语播客转中文播客",
    )
    parser.add_argument("input", help="输入英语播客 MP3 文件路径，或 YouTube 链接")
    parser.add_argument(
        "-o", "--output",
        help="输出文件路径（默认 input_zh.mp3；--download-only 下为下载的 MP3 路径）",
    )
    parser.add_argument(
        "--download-only", action="store_true",
        help="仅下载 YouTube 音频为 MP3 并退出，不执行后续翻译流水线",
    )
    parser.add_argument(
        "--whisper-model", default="large-v3",
        help="Whisper 模型大小（默认 large-v3）",
    )
    parser.add_argument(
        "--translation-provider",
        default="deepseek",
        choices=["deepseek", "openai"],
        help="翻译提供方：deepseek 或 openai（默认 deepseek）",
    )
    parser.add_argument(
        "--translation-model",
        default=None,
        help="翻译模型名（默认随 --translation-provider 自动选择）",
    )
    parser.add_argument(
        "--keep-intermediate",
        dest="keep_intermediate",
        action="store_true",
        default=True,
        help="保留中间文件（转录文本、翻译文本、音频片段等，默认开启）",
    )
    parser.add_argument(
        "--no-keep-intermediate",
        dest="keep_intermediate",
        action="store_false",
        help="不保留中间文件（流程结束后自动清理）",
    )
    parser.add_argument(
        "--tts-backend", default="indextts2",
        help="语音合成后端：qwen3 或 indextts2（默认 indextts2）",
    )
    parser.add_argument(
        "--index-tts-model-dir", default="checkpoints",
        help="IndexTTS2 模型目录（默认 checkpoints）",
    )
    parser.add_argument(
        "--index-tts-cfg-path", default=None,
        help="IndexTTS2 配置文件路径（默认 <index-tts-model-dir>/config.yaml）",
    )
    args = parser.parse_args()

    raw_input = args.input.strip()
    source_is_youtube = is_youtube_url(raw_input)
    download_tmp_dir: str | None = None
    work_dir: str | None = None

    if args.download_only and not source_is_youtube:
        print("错误: --download-only 仅支持 YouTube 链接输入", file=sys.stderr)
        sys.exit(1)

    try:
        if source_is_youtube:
            print("[Step 0] 下载 YouTube 音频...")
            if args.download_only:
                downloaded_path = download_youtube_mp3(
                    raw_input,
                    output_path=args.output,
                )
                print(f"[Step 0] 下载完成: {downloaded_path}")
                print()
                print("完成！")
                return

            if args.keep_intermediate:
                download_dir = str(Path.cwd())
            else:
                download_tmp_dir = tempfile.mkdtemp(prefix="babel_youtube_")
                download_dir = download_tmp_dir

            input_path = download_youtube_mp3(raw_input, output_dir=download_dir)
            print(f"[Step 0] 下载完成: {input_path}")
        else:
            input_path = raw_input
            if not os.path.isfile(input_path):
                print(f"错误: 文件不存在 - {input_path}", file=sys.stderr)
                sys.exit(1)

        if args.output:
            output_path = args.output
        else:
            output_base_dir = Path.cwd() if source_is_youtube else Path(input_path).parent
            output_path = str(output_base_dir / f"{Path(input_path).stem}_zh.mp3")

        # Working directory for intermediate files
        if args.keep_intermediate:
            work_base_dir = Path.cwd() if source_is_youtube else Path(input_path).parent
            work_dir = str(work_base_dir / f"{Path(input_path).stem}_babel")
            os.makedirs(work_dir, exist_ok=True)
        else:
            work_dir = tempfile.mkdtemp(prefix="babel_")

        print(f"输入: {input_path}")
        print(f"输出: {output_path}")
        print()

        # Step 1: Transcribe + diarize
        segments = transcribe(input_path, model_size=args.whisper_model)
        if args.keep_intermediate:
            save_intermediate(
                {"segments": segments},
                os.path.join(work_dir, "transcription.json"),
            )
        print()

        # Step 2: Extract reference audio per speaker
        ref_paths = extract_reference_audio(input_path, segments, work_dir)
        print()

        # Step 3: Translate
        segments = translate_segments(
            segments,
            provider=args.translation_provider,
            model=args.translation_model,
        )
        if args.keep_intermediate:
            save_intermediate(
                {"segments": segments},
                os.path.join(work_dir, "translation.json"),
            )
        print()

        # Step 4: Synthesize with voice cloning
        wav_paths = synthesize_segments(
            segments,
            ref_paths,
            work_dir,
            tts_backend=args.tts_backend,
            index_tts_model_dir=args.index_tts_model_dir,
            index_tts_cfg_path=args.index_tts_cfg_path,
        )
        print()

        # Step 5: Concatenate
        concatenate_audio(wav_paths, segments, output_path)

        print()
        print("完成！")
    finally:
        if not args.keep_intermediate and work_dir is not None:
            shutil.rmtree(work_dir, ignore_errors=True)
        if download_tmp_dir is not None:
            shutil.rmtree(download_tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
