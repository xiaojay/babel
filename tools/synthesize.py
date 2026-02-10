"""Step 4: 声音克隆合成（Qwen3-TTS / IndexTTS2）."""

import os

import soundfile as sf
import torch

from tools import get_device


def synthesize_segments(
    segments: list[dict],
    ref_audio_paths: dict[str, str],
    work_dir: str,
    tts_backend: str = "indextts2",
    index_tts_model_dir: str = "checkpoints",
    index_tts_cfg_path: str | None = None,
    progress_every: int = 10,
) -> list[str]:
    """Synthesize Chinese speech for each segment using the selected TTS backend.

    Returns a list of WAV file paths in segment order.
    """
    backend = (tts_backend or "").strip().lower()
    if backend in {"qwen", "qwen3", "qwen3-tts", "qwen_tts"}:
        return _synthesize_with_qwen(
            segments=segments,
            ref_audio_paths=ref_audio_paths,
            work_dir=work_dir,
            progress_every=progress_every,
        )
    if backend in {"indextts2", "index-tts2", "index_tts2"}:
        return _synthesize_with_indextts2(
            segments=segments,
            ref_audio_paths=ref_audio_paths,
            work_dir=work_dir,
            index_tts_model_dir=index_tts_model_dir,
            index_tts_cfg_path=index_tts_cfg_path,
            progress_every=progress_every,
        )

    raise ValueError(
        f"不支持的 tts_backend: {tts_backend}. 可选: qwen3, indextts2"
    )


def _segment_text(seg: dict) -> str:
    text_zh = (seg.get("text_zh") or "").strip()
    if text_zh:
        return text_zh
    text = (seg.get("text") or "").strip()
    if text:
        return text
    return "你好"


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
        ref_text = _pick_ref_text(speaker)
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


def _synthesize_with_indextts2(
    segments: list[dict],
    ref_audio_paths: dict[str, str],
    work_dir: str,
    index_tts_model_dir: str,
    index_tts_cfg_path: str | None,
    progress_every: int,
) -> list[str]:
    """Synthesize with IndexTTS2 voice cloning."""
    from indextts.infer_v2 import IndexTTS2

    device = get_device()
    index_device = "cuda:0" if device == "cuda" else device
    cfg_path = index_tts_cfg_path or os.path.join(index_tts_model_dir, "config.yaml")
    use_fp16 = device == "cuda"
    use_cuda_kernel = device == "cuda"

    print(f"[Step 4] 加载 IndexTTS2 模型 ({index_device})...")
    tts = IndexTTS2(
        cfg_path=cfg_path,
        model_dir=index_tts_model_dir,
        use_fp16=use_fp16,
        device=index_device,
        use_cuda_kernel=use_cuda_kernel,
        use_deepspeed=False,
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
        tts.infer(
            spk_audio_prompt=ref_path,
            text=_segment_text(seg),
            output_path=out_path,
            verbose=False,
        )
        wav_paths.append(out_path)

        if (i + 1) % progress_every == 0 or i == total - 1:
            print(f"  已合成 {i + 1}/{total}")

    return wav_paths
