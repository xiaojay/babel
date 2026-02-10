"""Tests for tools.transcribe."""

import importlib
import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest


def _make_fake_whisperx():
    fake = ModuleType("whisperx")
    fake.load_model = MagicMock()
    fake.load_audio = MagicMock(return_value="audio_array")
    fake.load_align_model = MagicMock(return_value=("align_model", "metadata"))
    fake.align = MagicMock()
    fake.assign_word_speakers = MagicMock()
    return fake


@pytest.fixture(autouse=True)
def _setup(monkeypatch):
    """Inject fake whisperx and force CPU device into tools.transcribe."""
    fake = _make_fake_whisperx()
    fake_diarize = ModuleType("whisperx.diarize")
    fake_diarize.DiarizationPipeline = MagicMock()

    monkeypatch.setitem(sys.modules, "whisperx", fake)
    monkeypatch.setitem(sys.modules, "whisperx.diarize", fake_diarize)

    # Reload tools.transcribe so it picks up the fake whisperx
    mod = importlib.import_module("tools.transcribe")
    importlib.reload(mod)
    # Patch get_device to return cpu by default
    monkeypatch.setattr(mod, "get_device", lambda: "cpu")

    yield fake, fake_diarize, mod


def _get(setup_result):
    """Unpack fixture into (whisperx, diarize, module)."""
    return setup_result


class TestTranscribeCallFlow:
    """Test the WhisperX call sequence."""

    def test_loads_model_and_transcribes(self, _setup, monkeypatch):
        whisperx, _, mod = _setup
        monkeypatch.delenv("HF_TOKEN", raising=False)

        raw_segments = [
            {"start": 0.0, "end": 1.5, "text": "Hello world", "speaker": "SPEAKER_00"},
        ]
        whisperx.align.return_value = {"segments": raw_segments}
        model = MagicMock()
        model.transcribe.return_value = {"segments": raw_segments, "language": "en"}
        whisperx.load_model.return_value = model

        result = mod.transcribe("test.mp3", model_size="base")

        whisperx.load_model.assert_called_once_with("base", "cpu", compute_type="float32")
        whisperx.load_audio.assert_called_once_with("test.mp3")
        model.transcribe.assert_called_once()
        assert len(result) == 1
        assert result[0]["speaker"] == "SPEAKER_00"

    def test_cuda_device_uses_float16(self, _setup, monkeypatch):
        whisperx, _, mod = _setup
        monkeypatch.setattr(mod, "get_device", lambda: "cuda")
        monkeypatch.delenv("HF_TOKEN", raising=False)

        raw_segments = [{"start": 0.0, "end": 1.0, "text": "Hi", "speaker": "SPEAKER_00"}]
        whisperx.align.return_value = {"segments": raw_segments}
        model = MagicMock()
        model.transcribe.return_value = {"segments": raw_segments, "language": "en"}
        whisperx.load_model.return_value = model

        mod.transcribe("test.mp3")

        whisperx.load_model.assert_called_once_with("large-v3", "cuda", compute_type="float16")


class TestNoDiarization:
    """When HF_TOKEN is not set, diarization is skipped."""

    def test_skips_diarization_without_hf_token(self, _setup, monkeypatch):
        whisperx, _, mod = _setup
        monkeypatch.delenv("HF_TOKEN", raising=False)

        raw_segments = [
            {"start": 0.0, "end": 2.0, "text": "Hello"},
            {"start": 2.5, "end": 4.0, "text": "World"},
        ]
        whisperx.align.return_value = {"segments": raw_segments}
        model = MagicMock()
        model.transcribe.return_value = {"segments": raw_segments, "language": "en"}
        whisperx.load_model.return_value = model

        result = mod.transcribe("test.mp3")

        for seg in result:
            assert seg["speaker"] == "SPEAKER_00"


class TestWithDiarization:
    """When HF_TOKEN is set, diarization runs."""

    def test_runs_diarization_with_hf_token(self, _setup, monkeypatch):
        whisperx, fake_diarize, mod = _setup
        monkeypatch.setenv("HF_TOKEN", "fake_token")

        aligned_segments = [
            {"start": 0.0, "end": 2.0, "text": " Hello ", "speaker": "SPEAKER_01"},
            {"start": 2.5, "end": 4.0, "text": " World ", "speaker": "SPEAKER_02"},
        ]
        whisperx.align.return_value = {"segments": aligned_segments}
        whisperx.assign_word_speakers.return_value = {"segments": aligned_segments}

        model = MagicMock()
        model.transcribe.return_value = {"segments": [], "language": "en"}
        whisperx.load_model.return_value = model

        diarize_mock = MagicMock(return_value="diarize_result")
        fake_diarize.DiarizationPipeline.return_value = diarize_mock

        result = mod.transcribe("test.mp3")

        assert len(result) == 2
        assert result[0]["speaker"] == "SPEAKER_01"
        assert result[1]["speaker"] == "SPEAKER_02"
        assert result[0]["text"] == "Hello"
        assert result[1]["text"] == "World"
