#!/usr/bin/env python3
"""Babel - 英语播客转中文播客

Pipeline: WhisperX STT + Diarization → LLM Translation → Translation Summary
→ Detailed Summary → Voice Clone (Qwen3 / IndexTTS2) → MP3
"""

import argparse
import json
import os
import shutil
import subprocess
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
    summarize_translated_segments,
    summarize_translated_segments_detailed,
    synthesize_segments,
    concatenate_audio,
    download_youtube_mp3,
    is_youtube_url,
)


def save_intermediate(data: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_text(text: str, path: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text.strip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Babel - 英语播客转中文播客",
    )
    parser.add_argument("input", help="输入英语播客 MP3 文件路径，或 YouTube 链接")
    parser.add_argument(
        "-o", "--output",
        help="输出文件路径（默认在 data/ 下；--download-only 下为下载的 MP3 路径）",
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
        default="openai",
        choices=["deepseek", "openai"],
        help="翻译提供方：deepseek 或 openai（默认 openai）",
    )
    parser.add_argument(
        "--translation-model",
        default=None,
        help=(
            "翻译模型名（默认随 --translation-provider 自动选择："
            "openai 为 gpt-5-mini，deepseek 为 deepseek-chat）"
        ),
    )
    parser.add_argument(
        "--summary-mode",
        default="both",
        choices=["short", "detailed", "both"],
        help="总结模式：short（简短）/ detailed（详细）/ both（两者，默认）",
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
        "--auto-publish",
        action="store_true",
        help="转录完成后自动发布到网站",
    )
    parser.add_argument(
        "--publish-title",
        default=None,
        help="发布时使用的标题（默认从文件名提取）",
    )
    parser.add_argument(
        "--publish-slug",
        default=None,
        help="发布时使用的 URL slug（默认从标题生成）",
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
    concat_group = parser.add_mutually_exclusive_group()
    concat_group.add_argument(
        "--concatenate-without-timestamps",
        action="store_true",
        help="第5步拼接时忽略时间戳，直接顺序拼接音频片段",
    )
    concat_group.add_argument(
        "--concatenate-fixed-gap-ms",
        type=int,
        metavar="MS",
        help="第5步拼接时忽略时间戳，并在片段间插入固定停顿（毫秒）",
    )
    args = parser.parse_args()

    data_dir = (Path(__file__).resolve().parent / "data").resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    raw_input = args.input.strip()
    source_is_youtube = is_youtube_url(raw_input)
    download_tmp_dir: str | None = None
    work_dir: str | None = None

    if args.download_only and not source_is_youtube:
        print("错误: --download-only 仅支持 YouTube 链接输入", file=sys.stderr)
        sys.exit(1)
    if args.concatenate_fixed_gap_ms is not None and args.concatenate_fixed_gap_ms < 0:
        print("错误: --concatenate-fixed-gap-ms 必须 >= 0", file=sys.stderr)
        sys.exit(1)

    try:
        if source_is_youtube:
            print("[Step 0] 下载 YouTube 音频...")
            if args.download_only:
                download_kwargs = (
                    {"output_path": args.output}
                    if args.output
                    else {"output_dir": str(data_dir)}
                )
                downloaded_path = download_youtube_mp3(raw_input, **download_kwargs)
                print(f"[Step 0] 下载完成: {downloaded_path}")
                print()
                print("完成！")
                return

            if args.keep_intermediate:
                download_dir = str(data_dir)
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
            output_path = str(data_dir / f"{Path(input_path).stem}_zh.mp3")
        summary_output_path = str(Path(output_path).with_suffix(".summary.txt"))
        detailed_summary_output_path = str(
            Path(output_path).with_suffix(".summary.detailed.md")
        )

        # Working directory for intermediate files
        if args.keep_intermediate:
            work_dir = str(data_dir / f"{Path(input_path).stem}_babel")
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

        if args.summary_mode in {"short", "both"}:
            try:
                summary_text = summarize_translated_segments(
                    segments,
                    provider=args.translation_provider,
                    model=args.translation_model,
                )
                save_text(summary_text, summary_output_path)
                print(f"[Step 3.5] 简短总结已写入: {summary_output_path}")
            except Exception as exc:
                print(
                    f"[Step 3.5] 警告: 简短总结生成失败，将继续后续流程: {exc}",
                    file=sys.stderr,
                )

        if args.summary_mode in {"detailed", "both"}:
            try:
                detailed_summary_text = summarize_translated_segments_detailed(
                    segments,
                    provider=args.translation_provider,
                    model=args.translation_model,
                )
                save_text(detailed_summary_text, detailed_summary_output_path)
                print(f"[Step 3.6] 详细总结已写入: {detailed_summary_output_path}")
            except Exception as exc:
                print(
                    f"[Step 3.6] 警告: 详细总结生成失败，将继续后续流程: {exc}",
                    file=sys.stderr,
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
        use_timestamps_for_concat = True
        fixed_gap_ms = None
        if args.concatenate_fixed_gap_ms is not None:
            use_timestamps_for_concat = False
            fixed_gap_ms = args.concatenate_fixed_gap_ms
        elif args.concatenate_without_timestamps:
            use_timestamps_for_concat = False

        concatenate_audio(
            wav_paths,
            segments,
            output_path,
            use_timestamps=use_timestamps_for_concat,
            fixed_gap_ms=fixed_gap_ms,
        )

        print()
        print("完成！")
        # Step 6: Auto-publish (optional)
        if args.auto_publish:
            print()
            print("[Step 6] 自动发布到网站...")
            publish_cmd = [
                sys.executable,
                str(Path(__file__).parent / "publish.py"),
                str(output_path),
            ]
            if args.publish_title:
                publish_cmd.extend(["--title", args.publish_title])
            if args.publish_slug:
                publish_cmd.extend(["--slug", args.publish_slug])
            # 查找英文原版
            en_path = Path(str(output_path).replace("_zh.mp3", ".mp3"))
            if en_path.exists() and en_path != Path(output_path):
                publish_cmd.extend(["--en-audio", str(en_path)])
            subprocess.run(publish_cmd)

    finally:
        if not args.keep_intermediate and work_dir is not None:
            shutil.rmtree(work_dir, ignore_errors=True)
        if download_tmp_dir is not None:
            shutil.rmtree(download_tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
