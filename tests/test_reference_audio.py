"""Tests for tools.reference_audio."""

import os
import tempfile

from pydub import AudioSegment
from pydub.generators import Sine

from tools.reference_audio import extract_reference_audio


def _make_audio(duration_ms: int = 15000) -> AudioSegment:
    """Generate a simple sine wave audio segment."""
    return Sine(440).to_audio_segment(duration=duration_ms)


class TestSpeakerGrouping:
    """Test that segments are grouped correctly by speaker."""

    def test_single_speaker(self, tmp_path):
        audio = _make_audio(10000)
        audio_path = str(tmp_path / "input.wav")
        audio.export(audio_path, format="wav")

        segments = [
            {"start": 0.0, "end": 5.0, "text": "Hello", "speaker": "SPEAKER_00"},
            {"start": 5.0, "end": 8.0, "text": "World", "speaker": "SPEAKER_00"},
        ]

        result = extract_reference_audio(audio_path, segments, str(tmp_path))

        assert len(result) == 1
        assert "SPEAKER_00" in result
        assert os.path.isfile(result["SPEAKER_00"])

    def test_multiple_speakers(self, tmp_path):
        audio = _make_audio(15000)
        audio_path = str(tmp_path / "input.wav")
        audio.export(audio_path, format="wav")

        segments = [
            {"start": 0.0, "end": 5.0, "text": "Hi", "speaker": "SPEAKER_00"},
            {"start": 5.0, "end": 10.0, "text": "Hey", "speaker": "SPEAKER_01"},
            {"start": 10.0, "end": 14.0, "text": "Bye", "speaker": "SPEAKER_00"},
        ]

        result = extract_reference_audio(audio_path, segments, str(tmp_path))

        assert len(result) == 2
        assert "SPEAKER_00" in result
        assert "SPEAKER_01" in result


class TestBestSegmentSelection:
    """Test the 3-10s selection logic."""

    def test_selects_segment_in_3_to_10s_range(self, tmp_path):
        audio = _make_audio(20000)
        audio_path = str(tmp_path / "input.wav")
        audio.export(audio_path, format="wav")

        segments = [
            {"start": 0.0, "end": 2.0, "text": "Short", "speaker": "SPEAKER_00"},  # 2s - too short
            {"start": 2.0, "end": 7.0, "text": "Good", "speaker": "SPEAKER_00"},   # 5s - good
            {"start": 7.0, "end": 19.0, "text": "Long", "speaker": "SPEAKER_00"},  # 12s - too long
        ]

        result = extract_reference_audio(audio_path, segments, str(tmp_path))
        ref_audio = AudioSegment.from_wav(result["SPEAKER_00"])
        # The 5s segment should be selected
        assert abs(len(ref_audio) - 5000) < 100  # ~5s with some tolerance

    def test_clamps_long_segment_to_10s(self, tmp_path):
        audio = _make_audio(20000)
        audio_path = str(tmp_path / "input.wav")
        audio.export(audio_path, format="wav")

        # Only one segment, longer than 10s - should be clamped
        segments = [
            {"start": 0.0, "end": 15.0, "text": "Very long", "speaker": "SPEAKER_00"},
        ]

        result = extract_reference_audio(audio_path, segments, str(tmp_path))
        ref_audio = AudioSegment.from_wav(result["SPEAKER_00"])
        assert len(ref_audio) <= 10100  # 10s + small tolerance

    def test_falls_back_to_longest_segment(self, tmp_path):
        audio = _make_audio(5000)
        audio_path = str(tmp_path / "input.wav")
        audio.export(audio_path, format="wav")

        # All segments are shorter than 3s
        segments = [
            {"start": 0.0, "end": 1.0, "text": "A", "speaker": "SPEAKER_00"},
            {"start": 1.0, "end": 2.5, "text": "B", "speaker": "SPEAKER_00"},
            {"start": 2.5, "end": 3.5, "text": "C", "speaker": "SPEAKER_00"},
        ]

        result = extract_reference_audio(audio_path, segments, str(tmp_path))
        ref_audio = AudioSegment.from_wav(result["SPEAKER_00"])
        # Should fall back to the 3-3.5s segment (longest sorted first, then
        # the 3.0-3.5 = 0.5? No: 1.0s segment). Actually the sorted order is:
        # C(1.0s), B(1.5s) ... wait let me recalculate.
        # A: 1.0s, B: 1.5s, C: 1.0s
        # Sorted descending: B(1.5s), A(1.0s), C(1.0s)
        # None in 3-10s range, so fallback to B (longest = 1.5s)
        assert abs(len(ref_audio) - 1500) < 100


class TestQualitySelection:
    """Test quality-based scoring for reference clip selection."""

    def test_prefers_higher_quality_clip_when_duration_equal(self, tmp_path):
        # Poor clip first (mostly silence), good clip second (continuous voiced tone).
        poor = AudioSegment.silent(duration=4500) + Sine(440).to_audio_segment(duration=500)
        good = Sine(440).to_audio_segment(duration=5000)
        audio = poor + good

        audio_path = str(tmp_path / "input.wav")
        audio.export(audio_path, format="wav")

        segments = [
            {"start": 0.0, "end": 5.0, "text": "Poor", "speaker": "SPEAKER_00"},
            {"start": 5.0, "end": 10.0, "text": "Good", "speaker": "SPEAKER_00"},
        ]

        result = extract_reference_audio(audio_path, segments, str(tmp_path))
        ref_audio = AudioSegment.from_wav(result["SPEAKER_00"])

        # If scoring works, it should pick the second segment (much louder on average).
        assert abs(len(ref_audio) - 5000) < 100
        assert ref_audio.dBFS > -8.0
