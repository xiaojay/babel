"""Babel tools - 英语播客转中文播客 pipeline 各步骤."""

import torch


def get_device() -> str:
    """Detect best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


from tools.transcribe import transcribe
from tools.reference_audio import extract_reference_audio
from tools.translate import (
    translate_segments,
    summarize_translated_segments,
    summarize_translated_segments_detailed,
)
from tools.synthesize import synthesize_segments
from tools.concatenate import concatenate_audio
from tools.youtube_download import download_youtube_mp3, is_youtube_url

__all__ = [
    "get_device",
    "transcribe",
    "extract_reference_audio",
    "translate_segments",
    "summarize_translated_segments",
    "summarize_translated_segments_detailed",
    "synthesize_segments",
    "concatenate_audio",
    "download_youtube_mp3",
    "is_youtube_url",
]
