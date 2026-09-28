"""Step 4: 声音克隆合成（IndexTTS-2.5 / IndexTTS2 / Qwen3-TTS）."""

import json
import os

import soundfile as sf
import torch

from tools import get_device

DEFAULT_TTS_BACKEND = "indextts2.5"
# The two IndexTTS models cannot share a directory: their checkpoints differ.
INDEX_TTS_MODEL_DIRS = {
    "indextts2.5": "checkpoints_2.5",
    "indextts2": "checkpoints",
}
_BACKEND_ALIASES = {
    "indextts2.5": {"indextts2.5", "index-tts2.5", "index_tts2.5", "indextts-2.5", "indextts25"},
    "indextts2": {"indextts2", "index-tts2", "index_tts2", "indextts-2"},
    "qwen3": {"qwen", "qwen3", "qwen3-tts", "qwen_tts"},
}


def synthesize_segments(
    segments: list[dict],
    ref_audio_paths: dict[str, str],
    work_dir: str,
    tts_backend: str = DEFAULT_TTS_BACKEND,
    index_tts_model_dir: str | None = None,
    index_tts_cfg_path: str | None = None,
    progress_every: int = 10,
) -> list[str]:
    """Synthesize Chinese speech for each segment using the selected TTS backend.

    Returns a list of WAV file paths in segment order.
    """
    name = (tts_backend or "").strip().lower()
    backend = next(
        (key for key, aliases in _BACKEND_ALIASES.items() if name in aliases), None
    )
    if backend == "qwen3":
        return _synthesize_with_qwen(
            segments=segments,
            ref_audio_paths=ref_audio_paths,
            work_dir=work_dir,
            progress_every=progress_every,
        )
    if backend in INDEX_TTS_MODEL_DIRS:
        return _synthesize_with_indextts(
            backend=backend,
            segments=segments,
            ref_audio_paths=ref_audio_paths,
            work_dir=work_dir,
            index_tts_model_dir=index_tts_model_dir or INDEX_TTS_MODEL_DIRS[backend],
            index_tts_cfg_path=index_tts_cfg_path,
            progress_every=progress_every,
        )

    raise ValueError(
        f"不支持的 tts_backend: {tts_backend}. 可选: indextts2.5, indextts2, qwen3"
    )


def _segment_text(seg: dict) -> str:
    text_zh = (seg.get("text_zh") or "").strip()
    if text_zh:
        return text_zh
    text = (seg.get("text") or "").strip()
    if text:
        return text
    return "你好"


def _load_ref_text_overrides(work_dir: str) -> dict[str, str]:
    metadata_path = os.path.join(work_dir, "ref_audio", "ref_metadata.json")
    if not os.path.isfile(metadata_path):
        return {}

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"[Step 4] 警告: 读取参考元数据失败，回退默认参考文本: {exc}")
        return {}

    speakers = data.get("speakers")
    if not isinstance(speakers, dict):
        return {}

    overrides: dict[str, str] = {}
    for speaker, info in speakers.items():
        if not isinstance(info, dict):
            continue
        ref_text = (info.get("ref_text") or "").strip()
        if ref_text:
            overrides[speaker] = ref_text
    return overrides


def _synthesize_with_qwen(
    segments: list[dict],
    ref_audio_paths: dict[str, str],
    work_dir: str,
    progress_every: int,
) -> list[str]:
    """Synthesize with Qwen3-TTS voice cloning."""
    from qwen_tts import Qwen3TTSModel

    device = get_device()
    print(f"[Step 4] 加载 Qwen3-TTS 模型 ({device})...")

    # Mac MPS: use bfloat16 + sdpa (no flash_attention_2)
    # CUDA: use bfloat16 + flash_attention_2 (if available)
    # CPU: use float32
    if device == "cpu":
        dtype = torch.float32
        attn_impl = "sdpa"
    elif device == "mps":
        dtype = torch.bfloat16
        attn_impl = "sdpa"
    else:
        dtype = torch.bfloat16
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"

    tts = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map=device,
        dtype=dtype,
        attn_implementation=attn_impl,
    )

    out_dir = os.path.join(work_dir, "tts_clips")
    os.makedirs(out_dir, exist_ok=True)

    # Pre-compute voice clone prompts per speaker for efficiency
    print("[Step 4] 为每个说话人生成声音特征...")
    speaker_prompts: dict = {}
    ref_text_overrides = _load_ref_text_overrides(work_dir)
    if ref_text_overrides:
        print("[Step 4] 使用 ref_metadata.json 中的参考文本")

    def _pick_ref_text(target_speaker: str) -> str:
        # Prefer English text from the same speaker; fall back to Chinese or any text.
        for seg in segments:
            if seg.get("speaker") == target_speaker:
                text = (seg.get("text") or "").strip()
                if text:
                    return text
                text_zh = (seg.get("text_zh") or "").strip()
                if text_zh:
                    return text_zh
        for seg in segments:
            text = (seg.get("text") or "").strip()
            if text:
                return text
            text_zh = (seg.get("text_zh") or "").strip()
            if text_zh:
                return text_zh
        return "你好"

    for speaker, ref_path in ref_audio_paths.items():
        ref_text = ref_text_overrides.get(speaker) or _pick_ref_text(speaker)
        if not ref_text.strip():
            ref_text = "你好"
            print(f"  警告: 未找到 {speaker} 的参考文本，使用占位文本")
        speaker_prompts[speaker] = tts.create_voice_clone_prompt(
            ref_audio=ref_path,
            ref_text=ref_text,
        )
        print(f"  {speaker}: 声音特征已提取")

    wav_paths: list[str] = []
    total = len(segments)

    if progress_every < 1:
        progress_every = 1

    for i, seg in enumerate(segments):
        speaker = seg["speaker"]
        out_path = os.path.join(out_dir, f"seg_{i:04d}.wav")
        prompt = speaker_prompts.get(speaker)

        wavs, sample_rate = tts.generate_voice_clone(
            text=_segment_text(seg),
            language="Chinese",
            voice_clone_prompt=prompt,
        )
        sf.write(out_path, wavs[0], sample_rate)
        wav_paths.append(out_path)

        if (i + 1) % progress_every == 0 or i == total - 1:
            print(f"  已合成 {i + 1}/{total}")

    return wav_paths


def _load_indextts(backend: str, model_dir: str, cfg_path: str, device: str):
    index_device = "cuda:0" if device == "cuda" else device
    on_cuda = device == "cuda"

    if backend == "indextts2":
        from indextts.infer_v2 import IndexTTS2

        print(f"[Step 4] 加载 IndexTTS2 模型 ({index_device})...")
        tts = IndexTTS2(
            cfg_path=cfg_path,
            model_dir=model_dir,
            use_fp16=on_cuda,
            device=index_device,
            use_cuda_kernel=on_cuda,
            use_deepspeed=False,
        )
        return tts, {}

    try:
        from indextts.infer_v2_5 import IndexTTS2
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "已安装的 indextts 不包含 IndexTTS-2.5。请从 "
            "https://github.com/index-tts/index-tts 安装新版，"
            "或使用 --tts-backend indextts2。"
        ) from exc

    print(f"[Step 4] 加载 IndexTTS-2.5 模型 ({index_device})...")
    tts = IndexTTS2(
        cfg_path=cfg_path,
        model_dir=model_dir,
        use_bf16=on_cuda,
        device=index_device,
        use_cuda_kernel=on_cuda,
        use_deepspeed=False,
    )
    return tts, {"lang": "ZH"}


def _synthesize_with_indextts(
    backend: str,
    segments: list[dict],
    ref_audio_paths: dict[str, str],
    work_dir: str,
    index_tts_model_dir: str,
    index_tts_cfg_path: str | None,
    progress_every: int,
) -> list[str]:
    """Synthesize with IndexTTS voice cloning (IndexTTS-2.5 or IndexTTS2)."""
    cfg_path = index_tts_cfg_path or os.path.join(index_tts_model_dir, "config.yaml")
    tts, infer_options = _load_indextts(
        backend, index_tts_model_dir, cfg_path, get_device()
    )

    out_dir = os.path.join(work_dir, "tts_clips")
    os.makedirs(out_dir, exist_ok=True)

    if progress_every < 1:
        progress_every = 1

    default_ref = next(iter(ref_audio_paths.values()), None)
    if default_ref is None:
        raise ValueError("未找到参考音频，无法执行声音克隆")

    wav_paths: list[str] = []
    total = len(segments)

    for i, seg in enumerate(segments):
        speaker = seg.get("speaker", "")
        ref_path = ref_audio_paths.get(speaker, default_ref)
        out_path = os.path.join(out_dir, f"seg_{i:04d}.wav")
        # Skip existing clips
        if os.path.isfile(out_path) and os.path.getsize(out_path) > 1000:
            wav_paths.append(out_path)
            if (i + 1) % progress_every == 0 or i == total - 1:
                print(f"  已合成 {i + 1}/{total} (跳过已有)")
            continue
        tts.infer(
            spk_audio_prompt=ref_path,
            text=_segment_text(seg),
            output_path=out_path,
            verbose=False,
            **infer_options,
        )
        wav_paths.append(out_path)

        if (i + 1) % progress_every == 0 or i == total - 1:
            print(f"  已合成 {i + 1}/{total}")

    return wav_paths
