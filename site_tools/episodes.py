"""Add episodes to the site."""

import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .config import load_episodes, save_episodes


def _get_duration_seconds(audio_path: Path) -> int:
    """Get audio duration in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            str(audio_path),
        ],
        capture_output=True, text=True,
    )
    info = json.loads(result.stdout)
    return round(float(info["format"]["duration"]))


def _slugify(title: str) -> str:
    """Generate a URL-safe slug from a title."""
    slug = title.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug.strip("-")


def add_episode(args):
    """Add an episode: copy audio, read summaries, update episodes.json."""
    site_dir = Path(args.site_dir)
    episodes = load_episodes(site_dir)

    slug = args.slug if args.slug else _slugify(args.title)
    if not slug:
        raise ValueError("无法从标题生成 slug，请用 --slug 指定")

    # Check for duplicate slug
    if any(ep["slug"] == slug for ep in episodes):
        raise ValueError(f"slug 已存在: {slug}")

    # Create audio directory for this episode
    audio_dir = site_dir / "audio" / slug
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Copy Chinese audio (required)
    zh_src = Path(args.zh_audio)
    zh_dest = audio_dir / "zh.mp3"
    shutil.copy2(zh_src, zh_dest)
    print(f"已复制中文音频: {zh_dest}")

    # Copy English audio (optional)
    en_audio_path = None
    if args.en_audio:
        en_src = Path(args.en_audio)
        en_dest = audio_dir / "en.mp3"
        shutil.copy2(en_src, en_dest)
        en_audio_path = f"audio/{slug}/en.mp3"
        print(f"已复制英文音频: {en_dest}")

    # Get audio metadata via ffprobe
    duration_seconds = _get_duration_seconds(zh_dest)
    file_size_bytes = zh_dest.stat().st_size

    # Read summaries
    summary = ""
    if args.summary:
        summary = Path(args.summary).read_text(encoding="utf-8").strip()

    detailed_summary_md = ""
    if args.detailed_summary:
        detailed_summary_md = Path(args.detailed_summary).read_text(encoding="utf-8").strip()

    episode = {
        "slug": slug,
        "title": args.title,
        "pub_date": args.pub_date or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "added_at": datetime.now(timezone.utc).isoformat(),
        "zh_audio": f"audio/{slug}/zh.mp3",
        "en_audio": en_audio_path,
        "zh_audio_size_bytes": file_size_bytes,
        "zh_audio_duration_seconds": duration_seconds,
        "summary": summary,
        "detailed_summary_md": detailed_summary_md,
    }

    episodes.append(episode)
    save_episodes(site_dir, episodes)
    print(f"已添加剧集: {args.title} (slug: {slug})")
