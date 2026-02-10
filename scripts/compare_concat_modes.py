#!/usr/bin/env python3
"""Generate concatenation outputs for three Step-5 modes."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.concatenate import concatenate_audio


def _default_work_dir() -> Path:
    return ROOT_DIR / "data" / "Sarah Paine — How Mao conquered China (lecture & interview)_babel"


def load_segments(work_dir: Path) -> tuple[list[dict], Path]:
    for name in ("translation.json", "transcription.json"):
        meta_path = work_dir / name
        if not meta_path.is_file():
            continue
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        segments = payload.get("segments")
        if isinstance(segments, list):
            return segments, meta_path
    raise FileNotFoundError(
        f"未找到可用 segments 元数据。请确认 {work_dir} 下存在 translation.json 或 transcription.json"
    )


def list_wav_paths(tts_dir: Path) -> list[str]:
    wav_files = sorted(tts_dir.glob("seg_*.wav"), key=lambda p: int(p.stem.split("_")[1]))
    if not wav_files:
        raise FileNotFoundError(f"未找到 wav 片段: {tts_dir}")
    return [str(path) for path in wav_files]


def clamp_same_length(segments: list[dict], wav_paths: list[str]) -> tuple[list[dict], list[str]]:
    seg_n = len(segments)
    wav_n = len(wav_paths)
    if seg_n == wav_n:
        return segments, wav_paths

    min_n = min(seg_n, wav_n)
    print(
        f"警告: segments={seg_n}, wav={wav_n}，将按最小长度 {min_n} 进行拼接",
        file=sys.stderr,
    )
    return segments[:min_n], wav_paths[:min_n]


def maybe_render_video(mp3_path: Path) -> Path | None:
    if shutil.which("ffmpeg") is None:
        print("警告: 未找到 ffmpeg，跳过 MP4 生成", file=sys.stderr)
        return None

    mp4_path = mp3_path.with_suffix(".mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=size=1280x720:rate=30:color=black",
        "-i",
        str(mp3_path),
        "-shortest",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(mp4_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return mp4_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="比较 Step 5 的三种拼接方式，生成可试听文件"
    )
    parser.add_argument(
        "--work-dir",
        default=str(_default_work_dir()),
        help="Babel 工作目录（包含 translation/transcription.json 和 tts_clips）",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录（默认 <work-dir>/concat_compare）",
    )
    parser.add_argument(
        "--fixed-gap-ms",
        type=int,
        default=180,
        help="固定短停顿毫秒数（默认 180）",
    )
    parser.add_argument(
        "--with-video",
        action="store_true",
        help="额外生成 mp4（黑底画面 + 音频轨）",
    )
    args = parser.parse_args()

    if args.fixed_gap_ms < 0:
        raise ValueError("--fixed-gap-ms 必须 >= 0")

    work_dir = Path(args.work_dir).expanduser().resolve()
    tts_dir = work_dir / "tts_clips"
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (work_dir / "concat_compare").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    segments, used_meta = load_segments(work_dir)
    wav_paths = list_wav_paths(tts_dir)
    segments, wav_paths = clamp_same_length(segments, wav_paths)

    print(f"工作目录: {work_dir}")
    print(f"片段目录: {tts_dir}")
    print(f"元数据: {used_meta}")
    print(f"片段数: {len(wav_paths)}")
    print(f"输出目录: {output_dir}")
    print()

    tasks = [
        ("01_with_timestamps", True, None),
        ("02_no_timestamps_no_gap", False, None),
        (f"03_no_timestamps_fixed_gap_{args.fixed_gap_ms}ms", False, args.fixed_gap_ms),
    ]

    generated: list[tuple[Path, Path | None]] = []
    for name, use_timestamps, fixed_gap_ms in tasks:
        mp3_path = output_dir / f"{name}.mp3"
        print(f"[Generate] {mp3_path.name}")
        concatenate_audio(
            wav_paths=wav_paths,
            segments=segments,
            output_path=str(mp3_path),
            use_timestamps=use_timestamps,
            fixed_gap_ms=fixed_gap_ms,
        )

        mp4_path = None
        if args.with_video:
            print(f"[Render ] {mp3_path.with_suffix('.mp4').name}")
            mp4_path = maybe_render_video(mp3_path)

        generated.append((mp3_path, mp4_path))
        print()

    print("生成完成：")
    for mp3_path, mp4_path in generated:
        print(f"- MP3: {mp3_path}")
        if mp4_path is not None:
            print(f"  MP4: {mp4_path}")


if __name__ == "__main__":
    main()
