"""Tests for tools.synthesize."""

import json
import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")


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


@pytest.fixture
def _fake_indextts25(monkeypatch, _fake_indextts):
    """Provide fake indextts.infer_v2_5 module next to infer_v2."""
    fake_infer_v2_5 = ModuleType("indextts.infer_v2_5")
    fake_infer_v2_5.IndexTTS2 = MagicMock()
    monkeypatch.setitem(sys.modules, "indextts.infer_v2_5", fake_infer_v2_5)
    yield fake_infer_v2_5


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

    def test_indextts2_calls_infer(self, tmp_path, monkeypatch, _fake_indextts):
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
            str(tmp_path),
            tts_backend="indextts2",
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
        assert "lang" not in first_call

    def test_indextts2_cuda_uses_cuda0(self, tmp_path, monkeypatch, _fake_indextts):
        import tools.synthesize as mod
        monkeypatch.setattr(mod, "get_device", lambda: "cuda")

        tts_mock = MagicMock()
        _fake_indextts.IndexTTS2.return_value = tts_mock

        segments = [{"start": 0.0, "end": 1.0, "text_zh": "你好", "speaker": "S0"}]
        ref_paths = {"S0": "/tmp/ref_s0.wav"}

        mod.synthesize_segments(
            segments,
            ref_paths,
            str(tmp_path),
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


class TestIndexTTS25:
    """Test the default backend, IndexTTS-2.5."""

    SEGMENTS = [
        {"start": 0.0, "end": 1.0, "text_zh": "你好", "speaker": "S0"},
        {"start": 1.0, "end": 2.0, "text_zh": "世界", "speaker": "S1"},
    ]
    REF_PATHS = {"S0": "/tmp/ref_s0.wav", "S1": "/tmp/ref_s1.wav"}

    def test_is_the_default_backend(self, tmp_path, monkeypatch, _fake_indextts25, _fake_indextts):
        import tools.synthesize as mod
        monkeypatch.setattr(mod, "get_device", lambda: "cpu")
        tts_mock = MagicMock()
        _fake_indextts25.IndexTTS2.return_value = tts_mock

        result = mod.synthesize_segments(self.SEGMENTS, self.REF_PATHS, str(tmp_path))

        assert len(result) == 2
        _fake_indextts.IndexTTS2.assert_not_called()
        _fake_indextts25.IndexTTS2.assert_called_once_with(
            cfg_path="checkpoints_2.5/config.yaml",
            model_dir="checkpoints_2.5",
            use_bf16=False,
            device="cpu",
            use_cuda_kernel=False,
            use_deepspeed=False,
        )
        assert tts_mock.infer.call_count == 2
        first_call = tts_mock.infer.call_args_list[0].kwargs
        assert first_call["spk_audio_prompt"] == "/tmp/ref_s0.wav"
        assert first_call["text"] == "你好"
        assert first_call["lang"] == "ZH"
        assert first_call["output_path"] == str(tmp_path / "tts_clips" / "seg_0000.wav")

    def test_cuda_uses_bf16(self, tmp_path, monkeypatch, _fake_indextts25):
        import tools.synthesize as mod
        monkeypatch.setattr(mod, "get_device", lambda: "cuda")
        _fake_indextts25.IndexTTS2.return_value = MagicMock()

        mod.synthesize_segments(
            self.SEGMENTS,
            self.REF_PATHS,
            str(tmp_path),
            tts_backend="indextts2.5",
            index_tts_model_dir="/models/index-tts-2.5",
        )

        _fake_indextts25.IndexTTS2.assert_called_once_with(
            cfg_path="/models/index-tts-2.5/config.yaml",
            model_dir="/models/index-tts-2.5",
            use_bf16=True,
            device="cuda:0",
            use_cuda_kernel=True,
            use_deepspeed=False,
        )

    def test_old_indextts_install_gives_clear_error(self, tmp_path, monkeypatch, _fake_indextts):
        import tools.synthesize as mod
        monkeypatch.setattr(mod, "get_device", lambda: "cpu")
        # Only infer_v2 exists, as in an index-tts checkout from before 2.5.
        monkeypatch.setitem(sys.modules, "indextts.infer_v2_5", None)

        with pytest.raises(RuntimeError, match="indextts2"):
            mod.synthesize_segments(self.SEGMENTS, self.REF_PATHS, str(tmp_path))

    def test_skips_clips_that_already_exist(self, tmp_path, monkeypatch, _fake_indextts25):
        import tools.synthesize as mod
        monkeypatch.setattr(mod, "get_device", lambda: "cpu")
        tts_mock = MagicMock()
        _fake_indextts25.IndexTTS2.return_value = tts_mock
        clips = tmp_path / "tts_clips"
        clips.mkdir()
        (clips / "seg_0000.wav").write_bytes(b"x" * 2000)

        result = mod.synthesize_segments(self.SEGMENTS, self.REF_PATHS, str(tmp_path))

        assert result == [str(clips / "seg_0000.wav"), str(clips / "seg_0001.wav")]
        assert tts_mock.infer.call_count == 1
        assert tts_mock.infer.call_args.kwargs["text"] == "世界"

    def test_requires_reference_audio(self, tmp_path, monkeypatch, _fake_indextts25):
        import tools.synthesize as mod
        monkeypatch.setattr(mod, "get_device", lambda: "cpu")
        _fake_indextts25.IndexTTS2.return_value = MagicMock()

        with pytest.raises(ValueError):
            mod.synthesize_segments(self.SEGMENTS, {}, str(tmp_path))
