"""Tests for publish."""

import sys
from subprocess import CompletedProcess

import pytest

import publish


@pytest.fixture
def run_calls(monkeypatch):
    """Capture commands instead of executing them."""
    calls: list[list[str]] = []

    def _fake_run_cmd(cmd, check=True):
        calls.append(cmd)
        return CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(publish, "run_cmd", _fake_run_cmd)
    return calls


def _make_zh_audio(tmp_path):
    zh_audio = tmp_path / "show_zh.mp3"
    zh_audio.write_bytes(b"")
    return zh_audio


class TestAddEpisode:
    """Test the site.py add invocation."""

    def test_passes_summaries_written_next_to_audio(self, tmp_path, run_calls):
        zh_audio = _make_zh_audio(tmp_path)
        # Same locations babel.py derives from the output path.
        summary = zh_audio.with_suffix(".summary.txt")
        summary.write_text("摘要", encoding="utf-8")
        detailed = zh_audio.with_suffix(".summary.detailed.md")
        detailed.write_text("# 目录", encoding="utf-8")

        assert publish.add_episode(tmp_path / "site", "Show", "show", zh_audio)

        cmd = run_calls[0]
        assert cmd[cmd.index("--summary") + 1] == str(summary)
        assert cmd[cmd.index("--detailed-summary") + 1] == str(detailed)

    def test_omits_summaries_when_missing(self, tmp_path, run_calls):
        zh_audio = _make_zh_audio(tmp_path)

        assert publish.add_episode(tmp_path / "site", "Show", "show", zh_audio)

        cmd = run_calls[0]
        assert "--summary" not in cmd
        assert "--detailed-summary" not in cmd

    def test_runs_site_cli_with_current_interpreter(self, tmp_path, run_calls):
        zh_audio = _make_zh_audio(tmp_path)

        publish.add_episode(tmp_path / "site", "Show", "show", zh_audio)

        assert run_calls[0][:3] == [sys.executable, "site.py", "add"]

    def test_returns_false_when_add_fails(self, tmp_path, monkeypatch):
        zh_audio = _make_zh_audio(tmp_path)
        monkeypatch.setattr(
            publish,
            "run_cmd",
            lambda cmd, check=True: CompletedProcess(cmd, 1, stdout="", stderr="boom"),
        )

        assert publish.add_episode(tmp_path / "site", "Show", "show", zh_audio) is False


class TestSlugify:
    """Test slugs generated from titles."""

    @pytest.mark.parametrize(
        "title, expected",
        [
            (
                "The OpenAI/Hugging Face attack, clearly explained",
                "the-openai-hugging-face-attack-clearly-explained",
            ),
            (
                "Arsenal 0-2 Man City Reaction | Arsecast Extra",
                "arsenal-0-2-man-city-reaction-arsecast-extra",
            ),
            (
                "Moonlake: Multimodal, Interactive — with Fan-yun Sun",
                "moonlake-multimodal-interactive-with-fan-yun-sun",
            ),
            ("Sam Altman's plan", "sam-altmans-plan"),
            ("GPT-5.5 vs. Claude", "gpt-5-5-vs-claude"),
            ("snake_case_title", "snake-case-title"),
            ("  --Hello,   World!--  ", "hello-world"),
        ],
    )
    def test_punctuation_separates_words(self, title, expected):
        assert publish.slugify(title) == expected

    def test_site_and_publish_agree(self):
        from site_tools.episodes import slugify as site_slugify

        title = "The OpenAI/Hugging Face attack, clearly explained"

        assert site_slugify(title) == publish.slugify(title)

    def test_too_short_returns_none(self):
        assert publish.slugify("a") is None
        assert publish.slugify("!!!") is None

    def test_long_title_is_cut_without_trailing_dash(self):
        slug = publish.slugify("word " * 40)

        assert len(slug) <= 80
        assert not slug.endswith("-")


class TestUpload:
    """Test the upload to R2."""

    def test_audio_is_uploaded_with_its_content_type(self, tmp_path, run_calls):
        zh_audio = _make_zh_audio(tmp_path)

        assert publish.upload_to_r2(zh_audio, "audio/show/zh.mp3")

        (cmd,) = run_calls
        assert cmd[:4] == ["wrangler", "r2", "object", "put"]
        assert "babel-podcast/audio/show/zh.mp3" in cmd
        assert "--remote" in cmd
        assert "--content-type=audio/mpeg" in cmd

    def test_unknown_file_type_has_no_content_type(self, tmp_path, run_calls):
        blob = tmp_path / "data.unknownext"
        blob.write_bytes(b"")

        assert publish.upload_to_r2(blob, "misc/data")

        (cmd,) = run_calls
        assert not any(arg.startswith("--content-type") for arg in cmd)
