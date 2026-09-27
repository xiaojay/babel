"""Babel tools - 英语播客转中文播客 pipeline 各步骤."""

import importlib


def get_device() -> str:
    """Detect best available device: CUDA > MPS > CPU."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Exports are resolved on first access so that importing one step does not
# pull in the heavy dependencies (torch, whisperx, ...) of the others.
_EXPORTS = {
    "transcribe": "tools.transcribe",
    "extract_reference_audio": "tools.reference_audio",
    "translate_segments": "tools.translate",
    "summarize_translated_segments": "tools.translate",
    "summarize_translated_segments_detailed": "tools.translate",
    "synthesize_segments": "tools.synthesize",
    "concatenate_audio": "tools.concatenate",
    "download_youtube_mp3": "tools.youtube_download",
    "is_youtube_url": "tools.youtube_download",
}


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    # Importing tools.transcribe binds the submodule to this name; rebind it
    # to the function so `from tools import transcribe` keeps working.
    globals()[name] = value
    return value


__all__ = ["get_device", *_EXPORTS]
