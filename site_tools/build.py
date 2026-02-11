"""Build static site from templates and episode data."""

import os
import shutil
from datetime import datetime, timezone
from email.utils import format_datetime
from pathlib import Path

import markdown as md
from jinja2 import Environment, FileSystemLoader

from .config import load_config, load_episodes

TEMPLATE_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"


def _format_duration(seconds: int) -> str:
    """Convert seconds to HH:MM:SS."""
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _rfc2822(date_str: str) -> str:
    """Convert YYYY-MM-DD to RFC 2822 format for RSS."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return format_datetime(dt)


def _render_markdown(text: str) -> str:
    """Render Markdown text to HTML."""
    return md.markdown(text, extensions=["tables", "fenced_code"])


def build_site(args):
    """Render all HTML pages, RSS feed, and copy static assets."""
    site_dir = Path(args.site_dir)
    build_dir = site_dir / "build"

    config = load_config(site_dir)
    episodes = load_episodes(site_dir)

    # Sort episodes by pub_date descending
    episodes.sort(key=lambda ep: ep["pub_date"], reverse=True)

    # Set up Jinja2
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=True,
    )
    env.filters["format_duration"] = _format_duration
    env.filters["rfc2822"] = _rfc2822
    env.filters["markdown"] = _render_markdown

    base_url = config.get("base_url", "").rstrip("/")

    # Clean and recreate build dir (except audio symlink target)
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)

    # Render index.html
    tpl = env.get_template("index.html")
    html = tpl.render(config=config, episodes=episodes, base_url=base_url)
    (build_dir / "index.html").write_text(html, encoding="utf-8")
    print(f"已生成: {build_dir / 'index.html'}")

    # Render episode pages
    episodes_dir = build_dir / "episodes"
    episodes_dir.mkdir(exist_ok=True)
    tpl = env.get_template("episode.html")
    for ep in episodes:
        html = tpl.render(config=config, episode=ep, base_url=base_url)
        ep_path = episodes_dir / f"{ep['slug']}.html"
        ep_path.write_text(html, encoding="utf-8")
    print(f"已生成: {len(episodes)} 个剧集页面")

    # Render RSS feed
    tpl = env.get_template("rss.xml")
    rss = tpl.render(
        config=config,
        episodes=episodes,
        base_url=base_url,
        build_date=format_datetime(datetime.now(timezone.utc)),
    )
    (build_dir / "feed.xml").write_text(rss, encoding="utf-8")
    print(f"已生成: {build_dir / 'feed.xml'}")

    # Copy static assets
    css_dest = build_dir / "style.css"
    shutil.copy2(STATIC_DIR / "style.css", css_dest)
    print(f"已复制: {css_dest}")

    # Create audio symlink
    audio_link = build_dir / "audio"
    audio_target = site_dir / "audio"
    if audio_target.exists():
        os.symlink(os.path.relpath(audio_target, build_dir), audio_link)
        print(f"已创建符号链接: {audio_link} → {audio_target}")

    print("站点构建完成！")
