"""Tests for tools.synthesize."""

import json
import sys
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _fake_qwen_tts(monkeypatch):
    """Provide a fake qwen_tts module."""
    fake = ModuleType("qwen_tts")
    fake.Qwen3TTSModel = MagicMock()
    monkeypatch.setitem(sys.modules, "qwen_tts", fake)
    yield fake


@pytest.fixture(autouse=True)
def _fake_soundfile(monkeypatch):
    """Provide a fake soundfile module."""
    fake = ModuleType("soundfile")
    fake.write = MagicMock()
    monkeypatch.setitem(sys.modules, "soundfile", fake)
    yield fake


@pytest.fixture
def _fake_indextts(monkeypatch):
    """Provide fake indextts.infer_v2 module."""
    fake_pkg = ModuleType("indextts")
    fake_infer_v2 = ModuleType("indextts.infer_v2")
    fake_infer_v2.IndexTTS2 = MagicMock()
    monkeypatch.setitem(sys.modules, "indextts", fake_pkg)
    monkeypatch.setitem(sys.modules, "indextts.infer_v2", fake_infer_v2)
    yield fake_infer_v2


class TestDeviceDtypeSelection:
    """Test device/dtype/attention logic."""

    def test_cpu_uses_float32_sdpa(self, monkeypatch, _fake_qwen_tts):
        import tools.synthesize as mod
        monkeypatch.setattr(mod, "get_device", lambda: "cpu")
        import torch

        tts_mock = MagicMock()
        tts_mock.generate_voice_clone.return_value = ([np.zeros(1000)], 24000)
        _fake_qwen_tts.Qwen3TTSModel.from_pretrained.return_value = tts_mock

        segments = [{"start": 0.0, "end": 1.0, "text": "Hi", "text_zh": "你好", "speaker": "S0"}]
        ref_paths = {"S0": "/tmp/ref.wav"}

        mod.synthesize_segments(segments, ref_paths, "/tmp/work", tts_backend="qwen3")

        call_kwargs = _fake_qwen_tts.Qwen3TTSModel.from_pretrained.call_args[1]
        assert call_kwargs["dtype"] == torch.float32
        assert call_kwargs["attn_implementation"] == "sdpa"

    def test_mps_uses_bfloat16_sdpa(self, monkeypatch, _fake_qwen_tts):
        import tools.synthesize as mod
        monkeypatch.setattr(mod, "get_device", lambda: "mps")
        import torch

        tts_mock = MagicMock()
        tts_mock.generate_voice_clone.return_value = ([np.zeros(1000)], 24000)
        _fake_qwen_tts.Qwen3TTSModel.from_pretrained.return_value = tts_mock

        segments = [{"start": 0.0, "end": 1.0, "text": "Hi", "text_zh": "你好", "speaker": "S0"}]
        ref_paths = {"S0": "/tmp/ref.wav"}

        mod.synthesize_segments(segments, ref_paths, "/tmp/work", tts_backend="qwen3")

        call_kwargs = _fake_qwen_tts.Qwen3TTSModel.from_pretrained.call_args[1]
        assert call_kwargs["dtype"] == torch.bfloat16
        assert call_kwargs["attn_implementation"] == "sdpa"

    def test_cuda_without_flash_attn_uses_sdpa(self, monkeypatch, _fake_qwen_tts):
        import tools.synthesize as mod
        monkeypatch.setattr(mod, "get_device", lambda: "cuda")
        # Ensure flash_attn is NOT importable
        monkeypatch.delitem(sys.modules, "flash_attn", raising=False)
        import builtins
        original_import = builtins.__import__
        def _no_flash(name, *args, **kwargs):
            if name == "flash_attn":
                raise ImportError("no flash_attn")
            return original_import(name, *args, **kwargs)
        monkeypatch.setattr(builtins, "__import__", _no_flash)

        import torch

        tts_mock = MagicMock()
        tts_mock.generate_voice_clone.return_value = ([np.zeros(1000)], 24000)
        _fake_qwen_tts.Qwen3TTSModel.from_pretrained.return_value = tts_mock

        segments = [{"start": 0.0, "end": 1.0, "text": "Hi", "text_zh": "你好", "speaker": "S0"}]
        ref_paths = {"S0": "/tmp/ref.wav"}

        mod.synthesize_segments(segments, ref_paths, "/tmp/work", tts_backend="qwen3")

        call_kwargs = _fake_qwen_tts.Qwen3TTSModel.from_pretrained.call_args[1]
        assert call_kwargs["dtype"] == torch.bfloat16
        assert call_kwargs["attn_implementation"] == "sdpa"


class TestSynthesizeCallFlow:
    """Test the TTS call sequence."""

    def test_generates_wav_for_each_segment(self, monkeypatch, _fake_qwen_tts, _fake_soundfile):
        import tools.synthesize as mod
        monkeypatch.setattr(mod, "get_device", lambda: "cpu")

        tts_mock = MagicMock()
        tts_mock.generate_voice_clone.return_value = ([np.zeros(1000)], 24000)
        tts_mock.create_voice_clone_prompt.return_value = "prompt"
        _fake_qwen_tts.Qwen3TTSModel.from_pretrained.return_value = tts_mock

        segments = [
            {"start": 0.0, "end": 1.0, "text": "Hello", "text_zh": "你好", "speaker": "S0"},
            {"start": 1.0, "end": 2.0, "text": "World", "text_zh": "世界", "speaker": "S1"},
            {"start": 2.0, "end": 3.0, "text": "Bye", "text_zh": "再见", "speaker": "S0"},
        ]
        ref_paths = {"S0": "/tmp/ref_s0.wav", "S1": "/tmp/ref_s1.wav"}

        result = mod.synthesize_segments(segments, ref_paths, "/tmp/work", tts_backend="qwen3")

        assert len(result) == 3
        assert tts_mock.generate_voice_clone.call_count == 3
        assert tts_mock.create_voice_clone_prompt.call_count == 2  # one per speaker

    def test_voice_clone_prompt_uses_first_segment_text(self, monkeypatch, _fake_qwen_tts, _fake_soundfile):
        import tools.synthesize as mod
        monkeypatch.setattr(mod, "get_device", lambda: "cpu")

        tts_mock = MagicMock()
        tts_mock.generate_voice_clone.return_value = ([np.zeros(1000)], 24000)
        tts_mock.create_voice_clone_prompt.return_value = "prompt"
        _fake_qwen_tts.Qwen3TTSModel.from_pretrained.return_value = tts_mock

        segments = [
            {"start": 0.0, "end": 1.0, "text": "First", "text_zh": "第一", "speaker": "S0"},
            {"start": 1.0, "end": 2.0, "text": "Second", "text_zh": "第二", "speaker": "S0"},
        ]
        ref_paths = {"S0": "/tmp/ref.wav"}

        mod.synthesize_segments(segments, ref_paths, "/tmp/work", tts_backend="qwen3")

        # Should use the first segment's text as ref_text
        tts_mock.create_voice_clone_prompt.assert_called_once_with(
            ref_audio="/tmp/ref.wav",
            ref_text="First",
        )

    def test_voice_clone_prompt_prefers_ref_metadata_text(
        self, tmp_path, monkeypatch, _fake_qwen_tts, _fake_soundfile
    ):
        import tools.synthesize as mod
        monkeypatch.setattr(mod, "get_device", lambda: "cpu")

        tts_mock = MagicMock()
        tts_mock.generate_voice_clone.return_value = ([np.zeros(1000)], 24000)
        tts_mock.create_voice_clone_prompt.return_value = "prompt"
        _fake_qwen_tts.Qwen3TTSModel.from_pretrained.return_value = tts_mock

        work_dir = tmp_path / "work"
        ref_dir = work_dir / "ref_audio"
        ref_dir.mkdir(parents=True)
        metadata_path = ref_dir / "ref_metadata.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "speakers": {
                        "S0": {
                            "ref_text": "metadata ref text",
                        }
                    }
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        segments = [
            {"start": 0.0, "end": 1.0, "text": "First", "text_zh": "第一", "speaker": "S0"},
            {"start": 1.0, "end": 2.0, "text": "Second", "text_zh": "第二", "speaker": "S0"},
        ]
        ref_paths = {"S0": "/tmp/ref.wav"}

        mod.synthesize_segments(segments, ref_paths, str(work_dir), tts_backend="qwen3")

        tts_mock.create_voice_clone_prompt.assert_called_once_with(
            ref_audio="/tmp/ref.wav",
            ref_text="metadata ref text",
        )

    def test_invalid_backend_raises_value_error(self, monkeypatch):
        import tools.synthesize as mod
        monkeypatch.setattr(mod, "get_device", lambda: "cpu")

        segments = [{"start": 0.0, "end": 1.0, "text_zh": "你好", "speaker": "S0"}]
        ref_paths = {"S0": "/tmp/ref.wav"}

        with pytest.raises(ValueError):
            mod.synthesize_segments(segments, ref_paths, "/tmp/work", tts_backend="unknown")

    def test_default_backend_indextts2_calls_infer(self, monkeypatch, _fake_indextts):
        import tools.synthesize as mod
        monkeypatch.setattr(mod, "get_device", lambda: "cpu")

        tts_mock = MagicMock()
        _fake_indextts.IndexTTS2.return_value = tts_mock

        segments = [
            {"start": 0.0, "end": 1.0, "text_zh": "你好", "speaker": "S0"},
            {"start": 1.0, "end": 2.0, "text_zh": "世界", "speaker": "S1"},
        ]
        ref_paths = {"S0": "/tmp/ref_s0.wav", "S1": "/tmp/ref_s1.wav"}

        result = mod.synthesize_segments(
            segments,
            ref_paths,
            "/tmp/work",
        )

        assert len(result) == 2
        _fake_indextts.IndexTTS2.assert_called_once_with(
            cfg_path="checkpoints/config.yaml",
            model_dir="checkpoints",
            use_fp16=False,
            device="cpu",
            use_cuda_kernel=False,
            use_deepspeed=False,
        )
        assert tts_mock.infer.call_count == 2
        first_call = tts_mock.infer.call_args_list[0].kwargs
        assert first_call["spk_audio_prompt"] == "/tmp/ref_s0.wav"
        assert first_call["text"] == "你好"

    def test_indextts2_cuda_uses_cuda0(self, monkeypatch, _fake_indextts):
        import tools.synthesize as mod
        monkeypatch.setattr(mod, "get_device", lambda: "cuda")

        tts_mock = MagicMock()
        _fake_indextts.IndexTTS2.return_value = tts_mock

        segments = [{"start": 0.0, "end": 1.0, "text_zh": "你好", "speaker": "S0"}]
        ref_paths = {"S0": "/tmp/ref_s0.wav"}

        mod.synthesize_segments(
            segments,
            ref_paths,
            "/tmp/work",
            tts_backend="index-tts2",
            index_tts_model_dir="/models/index-tts2",
            index_tts_cfg_path="/models/index-tts2/my-config.yaml",
        )

        _fake_indextts.IndexTTS2.assert_called_once_with(
            cfg_path="/models/index-tts2/my-config.yaml",
            model_dir="/models/index-tts2",
            use_fp16=True,
            device="cuda:0",
            use_cuda_kernel=True,
            use_deepspeed=False,
        )
