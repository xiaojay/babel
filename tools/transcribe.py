"""Step 1: WhisperX 转录 + 说话人分离."""

import gc
import inspect
import json
import os

import whisperx

from tools import get_device
from tools.segmentation import DEFAULT_SPEAKER, resegment


def _release_memory() -> None:
    """Free GPU memory held by models that are no longer referenced."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_diarization_pipeline(hf_token: str | None, device: str, model_name: str | None):
    from whisperx.diarize import DiarizationPipeline

    # whisperx 3.8 renamed use_auth_token to token (pyannote.audio 4).
    try:
        params = inspect.signature(DiarizationPipeline.__init__).parameters
    except (TypeError, ValueError):
        params = {}
    token_arg = "use_auth_token" if "use_auth_token" in params else "token"

    kwargs = {token_arg: hf_token, "device": device}
    if model_name:
        kwargs["model_name"] = model_name
    return DiarizationPipeline(**kwargs)


def _count_speakers(raw_segments: list[dict]) -> int:
    speakers = set()
    for seg in raw_segments:
        speakers.add(seg.get("speaker"))
        for word in seg.get("words") or []:
            speakers.add(word.get("speaker"))
    speakers.discard(None)
    return len(speakers)


def transcribe(
    audio_path: str,
    model_size: str = "large-v3",
    language: str | None = "en",
    hotwords: str | None = None,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    diarization_model: str | None = None,
    resegment_words: bool = True,
    min_speaker_seconds: float = 15.0,
    raw_output_path: str | None = None,
) -> list[dict]:
    """Transcribe audio with WhisperX and assign speaker labels.

    Pass language=None to auto-detect. When raw_output_path is set, the
    word-level WhisperX result is written there before segments are rebuilt.

    Returns a list of segments: [{start, end, text, speaker}, ...]
    """
    device = get_device()
    # WhisperX (faster-whisper/ctranslate2) only supports cuda and cpu
    whisper_device = "cuda" if device == "cuda" else "cpu"
    compute_type = "float16" if whisper_device == "cuda" else "float32"

    print(f"[Step 1] 加载 WhisperX 模型 ({model_size}, {whisper_device})...")
    load_kwargs = {"compute_type": compute_type, "language": language}
    if hotwords:
        print(f"[Step 1] 热词: {hotwords}")
        load_kwargs["asr_options"] = {"hotwords": hotwords}
    model = whisperx.load_model(model_size, whisper_device, **load_kwargs)

    print("[Step 1] 转录中...")
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=16, language=language)
    detected_language = result["language"]
    del model
    _release_memory()

    # Align whisper output for word-level timestamps
    print("[Step 1] 对齐时间戳...")
    align_model, metadata = whisperx.load_align_model(
        language_code=detected_language, device=whisper_device
    )
    result = whisperx.align(
        result["segments"], align_model, metadata, audio, whisper_device,
        return_char_alignments=False,
    )
    del align_model
    _release_memory()

    # Speaker diarization. A model stored in a local directory needs no token.
    hf_token = os.getenv("HF_TOKEN")
    if hf_token or (diarization_model and os.path.isdir(diarization_model)):
        print("[Step 1] 说话人分离中...")
        diarize_pipeline = _load_diarization_pipeline(
            hf_token, whisper_device, diarization_model
        )
        speaker_hints = {
            "num_speakers": num_speakers,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
        }
        diarize_segments = diarize_pipeline(
            audio, **{k: v for k, v in speaker_hints.items() if v is not None}
        )
        result = whisperx.assign_word_speakers(diarize_segments, result)
        del diarize_pipeline
        _release_memory()
    else:
        print(f"警告: 未设置 HF_TOKEN，跳过说话人分离，所有片段标记为 {DEFAULT_SPEAKER}")

    raw_segments = result["segments"]
    if raw_output_path:
        with open(raw_output_path, "w", encoding="utf-8") as f:
            json.dump(
                {"language": detected_language, "segments": raw_segments},
                f, ensure_ascii=False, default=float,
            )

    if resegment_words:
        segments = resegment(raw_segments, min_speaker_seconds=min_speaker_seconds)
    else:
        segments = [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip(),
                "speaker": seg.get("speaker") or DEFAULT_SPEAKER,
            }
            for seg in raw_segments
        ]

    speakers = {s["speaker"] for s in segments}
    print(f"[Step 1] 转录完成，共 {len(segments)} 个片段，"
          f"识别到 {len(speakers)} 个说话人")
    if resegment_words:
        merged_speakers = _count_speakers(raw_segments) - len(speakers)
        print(f"[Step 1] 重新分段前为 {len(raw_segments)} 个片段"
              + (f"，合并了 {merged_speakers} 个时长过短的说话人"
                 if merged_speakers > 0 else ""))
    return segments
