"""Step 5: 拼接音频."""

from pydub import AudioSegment


def concatenate_audio(
    wav_paths: list[str],
    segments: list[dict],
    output_path: str,
) -> None:
    """Concatenate synthesized clips into a single MP3, preserving inter-segment gaps."""
    print("[Step 5] 拼接音频...")
    combined = AudioSegment.empty()

    for i, wav_path in enumerate(wav_paths):
        clip = AudioSegment.from_wav(wav_path)

        # Add silence gap between segments based on original timing
        if i > 0:
            gap_ms = int((segments[i]["start"] - segments[i - 1]["end"]) * 1000)
            gap_ms = max(gap_ms, 100)   # minimum 100ms pause
            gap_ms = min(gap_ms, 3000)  # maximum 3s pause
            combined += AudioSegment.silent(duration=gap_ms)

        combined += clip

    combined.export(output_path, format="mp3", bitrate="192k")
    duration_s = len(combined) / 1000
    print(f"[Step 5] 输出完成: {output_path} ({duration_s:.1f}s)")
