"""YouTube 下载工具：输入链接，输出 MP3 文件路径。"""

import importlib
import shutil
from pathlib import Path
from urllib.parse import urlparse


def is_youtube_url(url: str) -> bool:
    """Check whether the given string is a YouTube URL."""
    if not url:
        return False

    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"}:
        return False

    host = (parsed.hostname or "").lower()
    return host == "youtu.be" or host == "youtube.com" or host.endswith(".youtube.com")


def download_youtube_mp3(
    youtube_url: str,
    output_dir: str | None = None,
    output_path: str | None = None,
) -> str:
    """Download audio from a YouTube URL and convert it to MP3."""
    if not is_youtube_url(youtube_url):
        raise ValueError(f"无效的 YouTube 链接: {youtube_url}")

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("未找到 ffmpeg，无法转换为 MP3。请先安装 ffmpeg 并加入 PATH。")

    try:
        yt_dlp = importlib.import_module("yt_dlp")
    except ModuleNotFoundError as exc:
        raise RuntimeError("未安装 yt-dlp。请先执行: pip install yt-dlp") from exc

    if output_path:
        target_path = Path(output_path).expanduser().resolve()
        if target_path.suffix.lower() != ".mp3":
            target_path = target_path.with_suffix(".mp3")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        base_dir = target_path.parent
    else:
        base_dir = Path(output_dir or ".").expanduser().resolve()
        base_dir.mkdir(parents=True, exist_ok=True)
        target_path = None

    ydl_opts = {
        "format": "bestaudio/best",
        "noplaylist": True,
        "outtmpl": str(base_dir / "%(title)s.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        downloaded_path = Path(ydl.prepare_filename(info)).with_suffix(".mp3")

    if not downloaded_path.is_file():
        raise RuntimeError(f"下载失败，未生成 MP3 文件: {downloaded_path}")

    if target_path is not None and downloaded_path.resolve() != target_path:
        shutil.move(str(downloaded_path), str(target_path))
        downloaded_path = target_path

    return str(downloaded_path)
