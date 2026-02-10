"""Tests for tools.concatenate."""

import os

from pydub import AudioSegment
from pydub.generators import Sine
import pytest

from tools.concatenate import concatenate_audio


def _make_wav(tmp_path, name: str, duration_ms: int = 1000) -> str:
    """Create a short WAV file and return its path."""
    audio = Sine(440).to_audio_segment(duration=duration_ms)
    path = str(tmp_path / name)
    audio.export(path, format="wav")
    return path


class TestConcatenation:
    """Test basic audio concatenation."""

    def test_concatenates_multiple_clips(self, tmp_path):
        wav1 = _make_wav(tmp_path, "a.wav", 1000)
        wav2 = _make_wav(tmp_path, "b.wav", 1000)
        wav3 = _make_wav(tmp_path, "c.wav", 1000)

        segments = [
            {"start": 0.0, "end": 1.0},
            {"start": 1.5, "end": 2.5},
            {"start": 3.0, "end": 4.0},
        ]
        output = str(tmp_path / "output.mp3")

        concatenate_audio([wav1, wav2, wav3], segments, output)

        assert os.path.isfile(output)
        result = AudioSegment.from_mp3(output)
        # 3 clips of 1s each + 2 gaps
        assert len(result) > 3000

    def test_single_clip_no_gap(self, tmp_path):
        wav = _make_wav(tmp_path, "single.wav", 2000)

        segments = [{"start": 0.0, "end": 2.0}]
        output = str(tmp_path / "output.mp3")

        concatenate_audio([wav], segments, output)

        assert os.path.isfile(output)
        result = AudioSegment.from_mp3(output)
        assert abs(len(result) - 2000) < 100


class TestGapCalculation:
    """Test inter-segment silence gap logic."""

    def test_minimum_gap_100ms(self, tmp_path):
        """Even overlapping segments get at least 100ms gap."""
        wav1 = _make_wav(tmp_path, "a.wav", 500)
        wav2 = _make_wav(tmp_path, "b.wav", 500)

        # Segments with no gap (or overlap)
        segments = [
            {"start": 0.0, "end": 1.0},
            {"start": 1.0, "end": 2.0},  # 0ms gap -> should become 100ms
        ]
        output = str(tmp_path / "output.mp3")

        concatenate_audio([wav1, wav2], segments, output)

        result = AudioSegment.from_mp3(output)
        # 500 + 500 + 100ms gap = 1100ms
        assert len(result) >= 1050  # some codec tolerance

    def test_maximum_gap_3000ms(self, tmp_path):
        """Gaps larger than 3s are clamped to 3s."""
        wav1 = _make_wav(tmp_path, "a.wav", 500)
        wav2 = _make_wav(tmp_path, "b.wav", 500)

        # 10 second gap -> should be clamped to 3s
        segments = [
            {"start": 0.0, "end": 1.0},
            {"start": 11.0, "end": 12.0},
        ]
        output = str(tmp_path / "output.mp3")

        concatenate_audio([wav1, wav2], segments, output)

        result = AudioSegment.from_mp3(output)
        # 500 + 500 + 3000ms max gap = 4000ms
        assert len(result) <= 4200  # some codec tolerance

    def test_negative_gap_becomes_minimum(self, tmp_path):
        """Overlapping segments (negative gap) still get 100ms minimum."""
        wav1 = _make_wav(tmp_path, "a.wav", 500)
        wav2 = _make_wav(tmp_path, "b.wav", 500)

        # Overlapping segments
        segments = [
            {"start": 0.0, "end": 2.0},
            {"start": 1.0, "end": 3.0},  # negative gap
        ]
        output = str(tmp_path / "output.mp3")

        concatenate_audio([wav1, wav2], segments, output)

        result = AudioSegment.from_mp3(output)
        # Should have used 100ms minimum gap
        assert len(result) >= 1050

    def test_without_timestamps_has_no_inserted_gap(self, tmp_path):
        """When timestamps are disabled, clips are joined directly."""
        wav1 = _make_wav(tmp_path, "a.wav", 500)
        wav2 = _make_wav(tmp_path, "b.wav", 500)

        segments = [
            {"start": 0.0, "end": 1.0},
            {"start": 11.0, "end": 12.0},
        ]
        output = str(tmp_path / "output.mp3")

        concatenate_audio([wav1, wav2], segments, output, use_timestamps=False)

        result = AudioSegment.from_mp3(output)
        # No timestamp-based gap, should be close to 1000ms total.
        assert len(result) < 1200

    def test_without_timestamps_uses_fixed_gap(self, tmp_path):
        """When fixed gap is provided, it should be inserted between clips."""
        wav1 = _make_wav(tmp_path, "a.wav", 500)
        wav2 = _make_wav(tmp_path, "b.wav", 500)

        segments = [
            {"start": 0.0, "end": 1.0},
            {"start": 11.0, "end": 12.0},
        ]
        output = str(tmp_path / "output_fixed_gap.mp3")

        concatenate_audio(
            [wav1, wav2],
            segments,
            output,
            use_timestamps=False,
            fixed_gap_ms=200,
        )

        result = AudioSegment.from_mp3(output)
        # 500 + 500 + 200ms fixed gap = 1200ms
        assert 1150 <= len(result) <= 1300

    def test_negative_fixed_gap_raises(self, tmp_path):
        """Negative fixed gap should be rejected."""
        wav1 = _make_wav(tmp_path, "a.wav", 300)
        wav2 = _make_wav(tmp_path, "b.wav", 300)
        output = str(tmp_path / "output_invalid_gap.mp3")
        segments = [
            {"start": 0.0, "end": 0.3},
            {"start": 0.4, "end": 0.7},
        ]

        with pytest.raises(ValueError):
            concatenate_audio(
                [wav1, wav2],
                segments,
                output,
                use_timestamps=False,
                fixed_gap_ms=-1,
            )


class TestOutputFormat:
    """Test output file format."""

    def test_outputs_mp3(self, tmp_path):
        wav = _make_wav(tmp_path, "clip.wav", 1000)
        output = str(tmp_path / "result.mp3")

        concatenate_audio([wav], [{"start": 0.0, "end": 1.0}], output)

        # Verify it's a valid MP3 by loading it
        result = AudioSegment.from_mp3(output)
        assert len(result) > 0
