"""Step 2: 提取每个说话人的参考音频."""

import math
import os

from pydub import AudioSegment


def _segment_duration(seg: dict) -> float:
    return max(float(seg["end"]) - float(seg["start"]), 0.0)


def _clamp_segment_bounds_ms(seg: dict, audio_length_ms: int) -> tuple[int, int]:
    start_ms = max(0, int(float(seg["start"]) * 1000))
    end_ms = max(start_ms + 1, int(float(seg["end"]) * 1000))
    end_ms = min(end_ms, audio_length_ms)

    if end_ms - start_ms > 10000:
        end_ms = start_ms + 10000

    if end_ms > audio_length_ms:
        end_ms = audio_length_ms
        start_ms = max(0, end_ms - 10000)

    if end_ms <= start_ms:
        start_ms = max(0, min(start_ms, max(audio_length_ms - 1, 0)))
        end_ms = min(audio_length_ms, start_ms + 1)

    return start_ms, end_ms


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    idx = (len(vals) - 1) * (p / 100.0)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return vals[lo]
    frac = idx - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def _normalize(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    scaled = (value - low) / (high - low)
    return max(0.0, min(1.0, scaled))


def _duration_score(duration_s: float) -> float:
    if duration_s <= 0.0:
        return 0.0
    if 4.0 <= duration_s <= 8.0:
        return 1.0
    if duration_s < 4.0:
        return max(0.0, duration_s / 4.0)
    return max(0.0, 1.0 - ((duration_s - 8.0) / 4.0))


def _estimate_clip_ratio(clip: AudioSegment) -> float:
    samples = clip.get_array_of_samples()
    if not samples:
        return 1.0

    sample_width = clip.sample_width
    if sample_width <= 0:
        return 0.0

    max_int = float((1 << (8 * sample_width - 1)) - 1)
    if max_int <= 0:
        return 0.0

    threshold = max_int * 0.995
    clipped = sum(1 for sample in samples if abs(int(sample)) >= threshold)
    return clipped / len(samples)


def _frame_dbfs(clip: AudioSegment, frame_ms: int = 50, hop_ms: int = 25) -> list[float]:
    if len(clip) <= 0:
        return []
    if len(clip) <= frame_ms:
        return [clip.dBFS]

    vals: list[float] = []
    last_start = len(clip) - frame_ms
    pos = 0
    while pos <= last_start:
        vals.append(clip[pos:pos + frame_ms].dBFS)
        pos += hop_ms

    if vals and last_start > 0 and (last_start % hop_ms != 0):
        vals.append(clip[last_start:last_start + frame_ms].dBFS)

    return vals


def _score_reference_clip(clip: AudioSegment, duration_s: float) -> tuple[float, dict[str, float]]:
    frame_levels = [
        (v if math.isfinite(v) else -90.0)
        for v in _frame_dbfs(clip)
    ]
    if not frame_levels:
        metrics = {
            "speech_ratio": 0.0,
            "snr_db": 0.0,
            "loudness_dbfs": -90.0,
            "duration_score": _duration_score(duration_s),
            "clip_ratio": 1.0,
        }
        return -1.0, metrics

    noise_floor = _percentile(frame_levels, 20.0)
    max_level = max(frame_levels)
    speech_threshold = max(noise_floor + 6.0, -45.0)
    speech_threshold = min(speech_threshold, max_level - 2.0)
    speech_frames = [v for v in frame_levels if v >= speech_threshold]

    speech_ratio = len(speech_frames) / len(frame_levels)
    speech_level = (
        (sum(speech_frames) / len(speech_frames))
        if speech_frames
        else max(frame_levels)
    )
    noise_level = _percentile(frame_levels, 10.0)
    snr_db = max(0.0, speech_level - noise_level)
    # Suppress false high-SNR scores when only a small portion is voiced.
    snr_score = _normalize(snr_db, 6.0, 24.0) * speech_ratio

    loudness_dbfs = clip.dBFS if math.isfinite(clip.dBFS) else -90.0
    loudness_score = 1.0 - min(abs(loudness_dbfs + 19.0) / 12.0, 1.0)
    duration_pref = _duration_score(duration_s)

    clip_ratio = _estimate_clip_ratio(clip)
    clip_penalty = min(1.0, clip_ratio / 0.01)

    score = (
        0.35 * speech_ratio
        + 0.25 * snr_score
        + 0.15 * loudness_score
        + 0.15 * duration_pref
        - 0.25 * clip_penalty
    )

    if duration_s < 1.2:
        score -= 0.25
    if speech_ratio < 0.25:
        score -= 0.25

    metrics = {
        "speech_ratio": speech_ratio,
        "snr_db": snr_db,
        "loudness_dbfs": loudness_dbfs,
        "duration_score": duration_pref,
        "clip_ratio": clip_ratio,
    }
    return score, metrics


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

    # Group segments by speaker and select the best quality reference clip.
    speaker_segments: dict[str, list[dict]] = {}
    for seg in segments:
        speaker_segments.setdefault(seg["speaker"], []).append(seg)

    ref_paths: dict[str, str] = {}
    audio_length_ms = len(audio)

    for speaker, segs in speaker_segments.items():
        segs_sorted = sorted(segs, key=_segment_duration, reverse=True)
        in_range = [s for s in segs_sorted if 3.0 <= _segment_duration(s) <= 10.0]
        candidates = in_range if in_range else segs_sorted

        best_seg: dict | None = None
        best_bounds: tuple[int, int] | None = None
        best_clip: AudioSegment | None = None
        best_score: float = float("-inf")
        best_tie_breaker: float = float("-inf")
        best_metrics: dict[str, float] | None = None

        for seg in candidates:
            start_ms, end_ms = _clamp_segment_bounds_ms(seg, audio_length_ms)
            if end_ms <= start_ms:
                continue

            clip = audio[start_ms:end_ms]
            duration_s = (end_ms - start_ms) / 1000.0
            score, metrics = _score_reference_clip(clip, duration_s=duration_s)
            tie_breaker = _segment_duration(seg)

            if best_clip is None or (score, tie_breaker) > (best_score, best_tie_breaker):
                best_seg = seg
                best_bounds = (start_ms, end_ms)
                best_clip = clip
                best_score = score
                best_tie_breaker = tie_breaker
                best_metrics = metrics

        if best_clip is None or best_bounds is None:
            # Defensive fallback: this should rarely happen.
            best_seg = segs_sorted[0]
            start_ms, end_ms = _clamp_segment_bounds_ms(best_seg, audio_length_ms)
            best_bounds = (start_ms, end_ms)
            best_clip = audio[start_ms:end_ms]
            best_score, best_metrics = _score_reference_clip(
                best_clip,
                duration_s=(end_ms - start_ms) / 1000.0,
            )
            best_tie_breaker = _segment_duration(best_seg)

        start_ms, end_ms = best_bounds
        if best_metrics is None:
            best_metrics = {
                "speech_ratio": 0.0,
                "snr_db": 0.0,
                "loudness_dbfs": -90.0,
                "duration_score": 0.0,
                "clip_ratio": 1.0,
            }
        ref_path = os.path.join(ref_dir, f"{speaker}.wav")
        best_clip.export(ref_path, format="wav")
        ref_paths[speaker] = ref_path
        print(
            f"  {speaker}: {(end_ms - start_ms) / 1000:.1f}s "
            f"(score={best_score:.3f}, speech={best_metrics['speech_ratio']:.2f}, "
            f"snr={best_metrics['snr_db']:.1f}dB) → {ref_path}"
        )

    return ref_paths
