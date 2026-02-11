"""Site configuration and data management."""

import json
from pathlib import Path

DEFAULT_CONFIG = {
    "title": "Babel 播客",
    "description": "英语播客的中文翻译版",
    "author": "Babel",
    "base_url": "https://example.com/podcast",
    "language": "zh-cn",
    "cover_url": "",
}


def init_site(args):
    """Create site/ directory structure, config.json, and empty episodes.json."""
    site_dir = Path(args.site_dir)
    site_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / "audio").mkdir(exist_ok=True)
    (site_dir / "build").mkdir(exist_ok=True)

    config = dict(DEFAULT_CONFIG)
    if args.title:
        config["title"] = args.title
    if args.base_url:
        config["base_url"] = args.base_url.rstrip("/")
    if args.description:
        config["description"] = args.description
    if args.author:
        config["author"] = args.author

    config_path = site_dir / "config.json"
    if config_path.exists():
        print(f"config.json 已存在，跳过: {config_path}")
    else:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(f"已创建: {config_path}")

    episodes_path = site_dir / "episodes.json"
    if episodes_path.exists():
        print(f"episodes.json 已存在，跳过: {episodes_path}")
    else:
        with open(episodes_path, "w", encoding="utf-8") as f:
            json.dump([], f)
        print(f"已创建: {episodes_path}")

    print("站点初始化完成！")


def load_config(site_dir: str | Path) -> dict:
    config_path = Path(site_dir) / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_episodes(site_dir: str | Path) -> list[dict]:
    episodes_path = Path(site_dir) / "episodes.json"
    with open(episodes_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_episodes(site_dir: str | Path, episodes: list[dict]) -> None:
    episodes_path = Path(site_dir) / "episodes.json"
    with open(episodes_path, "w", encoding="utf-8") as f:
        json.dump(episodes, f, ensure_ascii=False, indent=2)
