"""Tests for site_tools.build."""

import json
from types import SimpleNamespace

from site_tools.build import build_site

AUDIO_PATH = "audio/ep-one/zh.mp3"


def _make_site(tmp_path, episode_overrides=None, **config_overrides):
    site_dir = tmp_path / "site"
    (site_dir / "audio").mkdir(parents=True)

    config = {
        "title": "测试播客",
        "description": "描述",
        "author": "Babel",
        "base_url": "https://example.com/podcast",
        "language": "zh-cn",
        "cover_url": "",
    }
    config.update(config_overrides)
    episodes = [
        {
            "slug": "ep-one",
            "title": "Episode One",
            "pub_date": "2026-02-11",
            "zh_audio": AUDIO_PATH,
            "en_audio": None,
            "zh_audio_size_bytes": 1234,
            "zh_audio_duration_seconds": 61,
            "summary": "摘要",
            "detailed_summary_md": "",
        }
    ]
    episodes[0].update(episode_overrides or {})

    (site_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False), encoding="utf-8"
    )
    (site_dir / "episodes.json").write_text(
        json.dumps(episodes, ensure_ascii=False), encoding="utf-8"
    )
    return site_dir


def _build(site_dir):
    build_site(SimpleNamespace(site_dir=str(site_dir)))
    build_dir = site_dir / "build"
    return (
        (build_dir / "index.html").read_text(encoding="utf-8"),
        (build_dir / "episodes" / "ep-one.html").read_text(encoding="utf-8"),
        (build_dir / "feed.xml").read_text(encoding="utf-8"),
    )


class TestAudioUrls:
    """Test how audio URLs are resolved."""

    def test_defaults_to_same_origin_audio(self, tmp_path):
        index, episode, feed = _build(_make_site(tmp_path))

        assert f'src="./{AUDIO_PATH}"' in index
        assert f'src="../{AUDIO_PATH}"' in episode
        assert f'url="https://example.com/podcast/{AUDIO_PATH}"' in feed

    def test_empty_audio_base_url_falls_back(self, tmp_path):
        _, _, feed = _build(_make_site(tmp_path, audio_base_url=""))

        assert f'url="https://example.com/podcast/{AUDIO_PATH}"' in feed

    def test_uses_configured_audio_base_url(self, tmp_path):
        site_dir = _make_site(tmp_path, audio_base_url="https://cdn.example.com/")

        index, episode, feed = _build(site_dir)

        expected = f"https://cdn.example.com/{AUDIO_PATH}"
        assert f'src="{expected}"' in index
        assert f'src="{expected}"' in episode
        assert f'url="{expected}"' in feed


class TestEpisodeCover:
    """Test the optional per-episode cover image in the feed."""

    COVER = "https://cdn.example.com/covers/ep-one.jpg"

    def test_episode_cover_is_added_to_its_item(self, tmp_path):
        site_dir = _make_site(tmp_path, episode_overrides={"cover_url": self.COVER})

        _, _, feed = _build(site_dir)

        item = feed[feed.index("<item>"):feed.index("</item>")]
        assert f'<itunes:image href="{self.COVER}"/>' in item

    def test_episode_without_cover_has_no_image(self, tmp_path):
        _, _, feed = _build(_make_site(tmp_path))

        assert "<itunes:image" not in feed
