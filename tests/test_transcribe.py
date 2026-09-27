"""Tests for tools.transcribe."""

import importlib
import json
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


def _prepare(whisperx, segments, language="en"):
    """Make the fake WhisperX return the given aligned segments."""
    whisperx.align.return_value = {"segments": segments}
    whisperx.assign_word_speakers.return_value = {"segments": segments}
    model = MagicMock()
    model.transcribe.return_value = {"segments": segments, "language": language}
    whisperx.load_model.return_value = model
    return model


class TestTranscribeCallFlow:
    """Test the WhisperX call sequence."""

    def test_loads_model_and_transcribes(self, _setup, monkeypatch):
        whisperx, _, mod = _setup
        monkeypatch.delenv("HF_TOKEN", raising=False)
        model = _prepare(whisperx, [
            {"start": 0.0, "end": 1.5, "text": "Hello world", "speaker": "SPEAKER_00"},
        ])

        result = mod.transcribe("test.mp3", model_size="base")

        whisperx.load_model.assert_called_once_with(
            "base", "cpu", compute_type="float32", language="en"
        )
        whisperx.load_audio.assert_called_once_with("test.mp3")
        model.transcribe.assert_called_once_with("audio_array", batch_size=16, language="en")
        assert len(result) == 1
        assert result[0]["speaker"] == "SPEAKER_00"

    def test_cuda_device_uses_float16(self, _setup, monkeypatch):
        whisperx, _, mod = _setup
        monkeypatch.setattr(mod, "get_device", lambda: "cuda")
        monkeypatch.delenv("HF_TOKEN", raising=False)
        _prepare(whisperx, [{"start": 0.0, "end": 1.0, "text": "Hi"}])

        mod.transcribe("test.mp3")

        whisperx.load_model.assert_called_once_with(
            "large-v3", "cuda", compute_type="float16", language="en"
        )

    def test_language_none_auto_detects(self, _setup, monkeypatch):
        whisperx, _, mod = _setup
        monkeypatch.delenv("HF_TOKEN", raising=False)
        _prepare(whisperx, [{"start": 0.0, "end": 1.0, "text": "Hallo"}], language="de")

        mod.transcribe("test.mp3", language=None)

        assert whisperx.load_model.call_args.kwargs["language"] is None
        assert whisperx.load_align_model.call_args.kwargs["language_code"] == "de"

    def test_passes_hotwords(self, _setup, monkeypatch):
        whisperx, _, mod = _setup
        monkeypatch.delenv("HF_TOKEN", raising=False)
        _prepare(whisperx, [{"start": 0.0, "end": 1.0, "text": "Hi"}])

        mod.transcribe("test.mp3", hotwords="Sam Altman, OpenAI")

        assert whisperx.load_model.call_args.kwargs["asr_options"] == {
            "hotwords": "Sam Altman, OpenAI"
        }

    def test_omits_asr_options_without_hotwords(self, _setup, monkeypatch):
        whisperx, _, mod = _setup
        monkeypatch.delenv("HF_TOKEN", raising=False)
        _prepare(whisperx, [{"start": 0.0, "end": 1.0, "text": "Hi"}])

        mod.transcribe("test.mp3")

        assert "asr_options" not in whisperx.load_model.call_args.kwargs


class TestNoDiarization:
    """When HF_TOKEN is not set, diarization is skipped."""

    def test_skips_diarization_without_hf_token(self, _setup, monkeypatch):
        whisperx, fake_diarize, mod = _setup
        monkeypatch.delenv("HF_TOKEN", raising=False)
        _prepare(whisperx, [
            {"start": 0.0, "end": 2.0, "text": "Hello"},
            {"start": 5.0, "end": 7.0, "text": "World"},
        ])

        result = mod.transcribe("test.mp3")

        fake_diarize.DiarizationPipeline.assert_not_called()
        assert len(result) == 2
        for seg in result:
            assert seg["speaker"] == "SPEAKER_00"


class TestWithDiarization:
    """When HF_TOKEN is set, diarization runs."""

    SEGMENTS = [
        {"start": 0.0, "end": 2.0, "text": " Hello ", "speaker": "SPEAKER_01"},
        {"start": 2.5, "end": 4.0, "text": " World ", "speaker": "SPEAKER_02"},
    ]

    def test_runs_diarization_with_hf_token(self, _setup, monkeypatch):
        whisperx, fake_diarize, mod = _setup
        monkeypatch.setenv("HF_TOKEN", "fake_token")
        _prepare(whisperx, self.SEGMENTS)
        diarize_mock = MagicMock(return_value="diarize_result")
        fake_diarize.DiarizationPipeline.return_value = diarize_mock

        result = mod.transcribe("test.mp3")

        diarize_mock.assert_called_once_with("audio_array")
        assert len(result) == 2
        assert result[0]["speaker"] == "SPEAKER_01"
        assert result[1]["speaker"] == "SPEAKER_02"
        assert result[0]["text"] == "Hello"
        assert result[1]["text"] == "World"

    def test_passes_speaker_hints(self, _setup, monkeypatch):
        whisperx, fake_diarize, mod = _setup
        monkeypatch.setenv("HF_TOKEN", "fake_token")
        _prepare(whisperx, self.SEGMENTS)
        diarize_mock = MagicMock(return_value="diarize_result")
        fake_diarize.DiarizationPipeline.return_value = diarize_mock

        mod.transcribe("test.mp3", min_speakers=2, max_speakers=3)

        diarize_mock.assert_called_once_with(
            "audio_array", min_speakers=2, max_speakers=3
        )

    def test_uses_token_argument_on_new_whisperx(self, _setup, monkeypatch):
        whisperx, fake_diarize, mod = _setup
        monkeypatch.setenv("HF_TOKEN", "fake_token")
        _prepare(whisperx, self.SEGMENTS)
        created = {}

        class NewPipeline:
            def __init__(self, model_name=None, token=None, device="cpu", cache_dir=None):
                created.update(model_name=model_name, token=token, device=device)

            def __call__(self, audio, **kwargs):
                return "diarize_result"

        fake_diarize.DiarizationPipeline = NewPipeline

        mod.transcribe("test.mp3", diarization_model="pyannote/custom")

        assert created == {
            "model_name": "pyannote/custom", "token": "fake_token", "device": "cpu",
        }

    def test_uses_use_auth_token_argument_on_old_whisperx(self, _setup, monkeypatch):
        whisperx, fake_diarize, mod = _setup
        monkeypatch.setenv("HF_TOKEN", "fake_token")
        _prepare(whisperx, self.SEGMENTS)
        created = {}

        class OldPipeline:
            def __init__(self, model_name=None, use_auth_token=None, device="cpu"):
                created.update(use_auth_token=use_auth_token, device=device)

            def __call__(self, audio, **kwargs):
                return "diarize_result"

        fake_diarize.DiarizationPipeline = OldPipeline

        mod.transcribe("test.mp3")

        assert created == {"use_auth_token": "fake_token", "device": "cpu"}


class TestPostProcessing:
    """Segments are rebuilt from word-level output."""

    WORD_SEGMENTS = [
        {
            "start": 0.0, "end": 0.7, "text": " So I",
            "words": [
                {"word": "So", "start": 0.0, "end": 0.3},
                {"word": "I", "start": 0.4, "end": 0.7},
            ],
        },
        {
            "start": 0.8, "end": 1.5, "text": " agree.",
            "words": [{"word": "agree.", "start": 0.8, "end": 1.5}],
        },
    ]

    def test_merges_fragments_by_default(self, _setup, monkeypatch):
        whisperx, _, mod = _setup
        monkeypatch.delenv("HF_TOKEN", raising=False)
        _prepare(whisperx, self.WORD_SEGMENTS)

        result = mod.transcribe("test.mp3")

        assert result == [
            {"start": 0.0, "end": 1.5, "text": "So I agree.", "speaker": "SPEAKER_00"}
        ]

    def test_keeps_whisperx_segments_when_disabled(self, _setup, monkeypatch):
        whisperx, _, mod = _setup
        monkeypatch.delenv("HF_TOKEN", raising=False)
        _prepare(whisperx, self.WORD_SEGMENTS)

        result = mod.transcribe("test.mp3", resegment_words=False)

        assert [s["text"] for s in result] == ["So I", "agree."]
        assert all(set(s) == {"start", "end", "text", "speaker"} for s in result)

    def test_writes_raw_output(self, _setup, monkeypatch, tmp_path):
        whisperx, _, mod = _setup
        monkeypatch.delenv("HF_TOKEN", raising=False)
        _prepare(whisperx, self.WORD_SEGMENTS)
        raw_path = tmp_path / "transcription_raw.json"

        mod.transcribe("test.mp3", raw_output_path=str(raw_path))

        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        assert raw["language"] == "en"
        assert raw["segments"] == self.WORD_SEGMENTS


class TestLocalDiarizationModel:
    """A model stored in a local directory works without a token."""

    SEGMENTS = [
        {"start": 0.0, "end": 2.0, "text": " Hello ", "speaker": "SPEAKER_01"},
        {"start": 2.5, "end": 4.0, "text": " World ", "speaker": "SPEAKER_02"},
    ]

    def test_runs_diarization_without_hf_token(self, _setup, monkeypatch, tmp_path):
        whisperx, fake_diarize, mod = _setup
        monkeypatch.delenv("HF_TOKEN", raising=False)
        _prepare(whisperx, self.SEGMENTS)
        created = {}

        class NewPipeline:
            def __init__(self, model_name=None, token=None, device="cpu", cache_dir=None):
                created.update(model_name=model_name, token=token)

            def __call__(self, audio, **kwargs):
                return "diarize_result"

        fake_diarize.DiarizationPipeline = NewPipeline

        result = mod.transcribe("test.mp3", diarization_model=str(tmp_path))

        assert created == {"model_name": str(tmp_path), "token": None}
        assert [s["speaker"] for s in result] == ["SPEAKER_01", "SPEAKER_02"]

    def test_model_name_without_token_skips_diarization(self, _setup, monkeypatch):
        whisperx, fake_diarize, mod = _setup
        monkeypatch.delenv("HF_TOKEN", raising=False)
        _prepare(whisperx, [{"start": 0.0, "end": 1.0, "text": "Hi"}])

        mod.transcribe("test.mp3", diarization_model="pyannote/speaker-diarization-3.1")

        fake_diarize.DiarizationPipeline.assert_not_called()
