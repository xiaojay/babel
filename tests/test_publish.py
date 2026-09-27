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
