"""Step 5: 拼接音频."""

from pydub import AudioSegment


def concatenate_audio(
    wav_paths: list[str],
    segments: list[dict],
    output_path: str,
    use_timestamps: bool = True,
    fixed_gap_ms: int | None = None,
) -> None:
    """Concatenate synthesized clips into a single MP3.

    When use_timestamps=True, preserve inter-segment gaps based on segment timings.
    When use_timestamps=False and fixed_gap_ms is provided, insert a fixed silence gap.
    """
    if fixed_gap_ms is not None and fixed_gap_ms < 0:
        raise ValueError("fixed_gap_ms 必须 >= 0")

    print("[Step 5] 拼接音频...")
    combined = AudioSegment.empty()

    for i, wav_path in enumerate(wav_paths):
        clip = AudioSegment.from_wav(wav_path)

        if i > 0:
            if use_timestamps:
                # Add silence gap between segments based on original timing.
                gap_ms = int((segments[i]["start"] - segments[i - 1]["end"]) * 1000)
                gap_ms = max(gap_ms, 100)   # minimum 100ms pause
                gap_ms = min(gap_ms, 3000)  # maximum 3s pause
                combined += AudioSegment.silent(duration=gap_ms)
            elif fixed_gap_ms is not None and fixed_gap_ms > 0:
                combined += AudioSegment.silent(duration=fixed_gap_ms)

        combined += clip

    combined.export(output_path, format="mp3", bitrate="192k")
    duration_s = len(combined) / 1000
    print(f"[Step 5] 输出完成: {output_path} ({duration_s:.1f}s)")
