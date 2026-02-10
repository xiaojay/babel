"""Step 1: WhisperX 转录 + 说话人分离."""

import os

import whisperx

from tools import get_device


def transcribe(audio_path: str, model_size: str = "large-v3") -> list[dict]:
    """Transcribe audio with WhisperX and assign speaker labels.

    Returns a list of segments: [{start, end, text, speaker}, ...]
    """
    device = get_device()
    # WhisperX (faster-whisper/ctranslate2) only supports cuda and cpu
    whisper_device = "cuda" if device == "cuda" else "cpu"
    compute_type = "float16" if whisper_device == "cuda" else "float32"

    print(f"[Step 1] 加载 WhisperX 模型 ({model_size}, {whisper_device})...")
    model = whisperx.load_model(model_size, whisper_device, compute_type=compute_type)

    print("[Step 1] 转录中...")
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=16)

    # Align whisper output for word-level timestamps
    print("[Step 1] 对齐时间戳...")
    align_model, metadata = whisperx.load_align_model(
        language_code=result["language"], device=whisper_device
    )
    result = whisperx.align(
        result["segments"], align_model, metadata, audio, whisper_device,
        return_char_alignments=False,
    )

    # Speaker diarization
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("警告: 未设置 HF_TOKEN，跳过说话人分离，所有片段标记为 SPEAKER_00")
        for seg in result["segments"]:
            seg["speaker"] = "SPEAKER_00"
        return result["segments"]

    print("[Step 1] 说话人分离中...")
    from whisperx.diarize import DiarizationPipeline
    diarize_pipeline = DiarizationPipeline(
        use_auth_token=hf_token, device=whisper_device
    )
    diarize_segments = diarize_pipeline(audio_path)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    segments = []
    for seg in result["segments"]:
        segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip(),
            "speaker": seg.get("speaker", "SPEAKER_00"),
        })

    print(f"[Step 1] 转录完成，共 {len(segments)} 个片段，"
          f"识别到 {len({s['speaker'] for s in segments})} 个说话人")
    return segments
