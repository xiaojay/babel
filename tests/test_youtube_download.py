"""Tests for tools.youtube_download."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.youtube_download import download_youtube_mp3, is_youtube_url


def _build_fake_yt_dlp():
    class _FakeYoutubeDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_info(self, url, download=True):
            assert download is True
            info = {"title": "unit_test_video"}
            raw_path = self._raw_path(info["title"])
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_bytes(b"raw")
            raw_path.with_suffix(".mp3").write_bytes(b"mp3")
            return info

        def prepare_filename(self, info):
            return str(self._raw_path(info["title"]))

        def _raw_path(self, title: str) -> Path:
            template = self.opts["outtmpl"]
            return Path(template.replace("%(title)s", title).replace("%(ext)s", "webm"))

    return SimpleNamespace(YoutubeDL=_FakeYoutubeDL)


class TestYoutubeUrlValidation:
    """Test URL detection."""

    def test_accepts_youtube_domains(self):
        assert is_youtube_url("https://www.youtube.com/watch?v=abc")
        assert is_youtube_url("https://youtu.be/abc")
        assert is_youtube_url("https://music.youtube.com/watch?v=abc")

    def test_rejects_non_youtube_domains(self):
        assert not is_youtube_url("https://example.com/video")
        assert not is_youtube_url("/tmp/a.mp3")
        assert not is_youtube_url("youtube.com/watch?v=abc")


class TestDownloadYoutubeMp3:
    """Test download orchestration behavior."""

    def test_rejects_invalid_url(self):
        with pytest.raises(ValueError):
            download_youtube_mp3("https://example.com/not-youtube")

    def test_requires_ffmpeg(self, monkeypatch):
        import tools.youtube_download as mod

        monkeypatch.setattr(mod.shutil, "which", lambda _: None)

        with pytest.raises(RuntimeError, match="ffmpeg"):
            mod.download_youtube_mp3("https://youtu.be/abc")

    def test_requires_yt_dlp(self, monkeypatch):
        import tools.youtube_download as mod

        monkeypatch.setattr(mod.shutil, "which", lambda _: "/usr/bin/ffmpeg")

        real_import_module = importlib.import_module

        def _fake_import(name: str):
            if name == "yt_dlp":
                raise ModuleNotFoundError("yt_dlp")
            return real_import_module(name)

        monkeypatch.setattr(mod.importlib, "import_module", _fake_import)

        with pytest.raises(RuntimeError, match="yt-dlp"):
            mod.download_youtube_mp3("https://youtu.be/abc")

    def test_downloads_to_output_dir(self, tmp_path, monkeypatch):
        import tools.youtube_download as mod

        monkeypatch.setattr(mod.shutil, "which", lambda _: "/usr/bin/ffmpeg")
        monkeypatch.setattr(
            mod.importlib,
            "import_module",
            lambda name: _build_fake_yt_dlp(),
        )

        output = mod.download_youtube_mp3(
            "https://youtu.be/abc",
            output_dir=str(tmp_path),
        )

        assert output == str(tmp_path / "unit_test_video.mp3")
        assert Path(output).is_file()

    def test_downloads_and_moves_to_output_path(self, tmp_path, monkeypatch):
        import tools.youtube_download as mod

        monkeypatch.setattr(mod.shutil, "which", lambda _: "/usr/bin/ffmpeg")
        monkeypatch.setattr(
            mod.importlib,
            "import_module",
            lambda name: _build_fake_yt_dlp(),
        )

        target = tmp_path / "custom_name.mp3"
        output = mod.download_youtube_mp3(
            "https://youtu.be/abc",
            output_dir=str(tmp_path),
            output_path=str(target),
        )

        assert output == str(target.resolve())
        assert target.is_file()
        assert not (tmp_path / "unit_test_video.mp3").exists()

    def test_output_path_is_normalized_to_mp3(self, tmp_path, monkeypatch):
        import tools.youtube_download as mod

        monkeypatch.setattr(mod.shutil, "which", lambda _: "/usr/bin/ffmpeg")
        monkeypatch.setattr(
            mod.importlib,
            "import_module",
            lambda name: _build_fake_yt_dlp(),
        )

        target = tmp_path / "custom_name.wav"
        output = mod.download_youtube_mp3(
            "https://youtu.be/abc",
            output_path=str(target),
        )

        assert output == str((tmp_path / "custom_name.mp3").resolve())
        assert (tmp_path / "custom_name.mp3").is_file()
