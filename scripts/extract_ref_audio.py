#!/usr/bin/env python3
"""Extract per-speaker reference audio for a podcast file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.reference_audio import extract_reference_audio


DEFAULT_INPUT = (
    ROOT_DIR / "data" / "Sarah Paine — How Mao conquered China (lecture & interview).mp3"
)


def _default_work_dir(input_path: Path) -> Path:
    return input_path.parent / f"{input_path.stem}_babel"


def _read_segments(meta_path: Path) -> list[dict]:
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    segments = payload.get("segments")
    if not isinstance(segments, list):
        raise ValueError(f"{meta_path} 中未找到有效的 segments 列表")
    cleaned = [seg for seg in segments if isinstance(seg, dict)]
    if not cleaned:
        raise ValueError(f"{meta_path} 中 segments 为空")
    return cleaned


def _find_existing_segments_file(work_dir: Path) -> Path | None:
    for name in ("transcription.json", "translation.json"):
        candidate = work_dir / name
        if candidate.is_file():
            return candidate
    return None


def _save_transcription(meta_path: Path, segments: list[dict]) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps({"segments": segments}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="提取每个说话人的 ref audio")
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="输入音频路径（默认 Sarah Paine — How Mao conquered China）",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="工作目录（默认 <input_stem>_babel）",
    )
    parser.add_argument(
        "--segments-json",
        default=None,
        help="已有 segments 的 JSON 路径（含 {\"segments\": [...]}）",
    )
    parser.add_argument(
        "--force-transcribe",
        action="store_true",
        help="忽略已有 segments，强制重新转录",
    )
    parser.add_argument(
        "--whisper-model",
        default="large-v3",
        help="重新转录时使用的 Whisper 模型（默认 large-v3）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"输入音频不存在: {input_path}")

    work_dir = (
        Path(args.work_dir).expanduser().resolve()
        if args.work_dir
        else _default_work_dir(input_path).resolve()
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    explicit_segments_path = (
        Path(args.segments_json).expanduser().resolve() if args.segments_json else None
    )
    segments_path = (
        explicit_segments_path
        if explicit_segments_path is not None
        else _find_existing_segments_file(work_dir)
    )

    segments: list[dict] | None = None
    if not args.force_transcribe and segments_path is not None and segments_path.is_file():
        segments = _read_segments(segments_path)
        print(f"使用已有 segments: {segments_path} ({len(segments)} 条)")

    if segments is None:
        print("未找到可用 segments，开始转录...")
        from tools.transcribe import transcribe

        segments = transcribe(str(input_path), model_size=args.whisper_model)
        transcription_path = work_dir / "transcription.json"
        _save_transcription(transcription_path, segments)
        print(f"转录已写入: {transcription_path} ({len(segments)} 条)")

    print("开始提取 ref audio...")
    ref_paths = extract_reference_audio(str(input_path), segments, str(work_dir))

    ref_dir = work_dir / "ref_audio"
    metadata_path = ref_dir / "ref_metadata.json"
    print()
    print("完成。输出如下：")
    print(f"- ref 目录: {ref_dir}")
    print(f"- speaker 数: {len(ref_paths)}")
    for speaker, path in sorted(ref_paths.items()):
        print(f"  - {speaker}: {path}")
    if metadata_path.is_file():
        print(f"- metadata: {metadata_path}")


if __name__ == "__main__":
    main()
