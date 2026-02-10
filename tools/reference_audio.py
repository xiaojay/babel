"""Step 2: 提取每个说话人的参考音频."""

import os

from pydub import AudioSegment


def extract_reference_audio(
    audio_path: str, segments: list[dict], work_dir: str
) -> dict[str, str]:
    """Extract a reference audio clip (3-10s) per speaker from the original audio.

    Returns {speaker_id: ref_wav_path}.
    """
    print("[Step 2] 提取参考音频...")
    audio = AudioSegment.from_file(audio_path)
    ref_dir = os.path.join(work_dir, "ref_audio")
    os.makedirs(ref_dir, exist_ok=True)

    # Group segments by speaker, pick the longest one (clamped to 3-10s)
    speaker_segments: dict[str, list[dict]] = {}
    for seg in segments:
        speaker_segments.setdefault(seg["speaker"], []).append(seg)

    ref_paths: dict[str, str] = {}
    for speaker, segs in speaker_segments.items():
        # Sort by duration descending
        segs_sorted = sorted(segs, key=lambda s: s["end"] - s["start"], reverse=True)
        best = None
        for s in segs_sorted:
            dur = s["end"] - s["start"]
            if 3.0 <= dur <= 10.0:
                best = s
                break
        if best is None:
            # Fall back to the longest segment available
            best = segs_sorted[0]

        start_ms = int(best["start"] * 1000)
        end_ms = int(best["end"] * 1000)
        # Clamp to 10 seconds
        if end_ms - start_ms > 10000:
            end_ms = start_ms + 10000

        clip = audio[start_ms:end_ms]
        ref_path = os.path.join(ref_dir, f"{speaker}.wav")
        clip.export(ref_path, format="wav")
        ref_paths[speaker] = ref_path
        print(f"  {speaker}: {(end_ms - start_ms) / 1000:.1f}s → {ref_path}")

    return ref_paths
