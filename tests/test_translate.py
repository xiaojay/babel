"""Tests for tools.translate."""

from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _set_api_keys(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake_deepseek_key")
    monkeypatch.setenv("OPENAI_API_KEY", "fake_openai_key")


def _make_response(text: str):
    """Create a mock API response."""
    choice = MagicMock()
    choice.message.content = text
    response = MagicMock()
    response.choices = [choice]
    return response


class TestNumberParsing:
    """Test removal of numbering from translated lines."""

    def test_parses_dot_numbered_lines(self, monkeypatch):
        client = MagicMock()
        client.chat.completions.create.return_value = _make_response(
            "1. 你好世界\n2. 再见世界"
        )
        import tools.translate as mod
        monkeypatch.setattr(mod, "OpenAI", lambda **kw: client)

        segments = [
            {"start": 0.0, "end": 1.0, "text": "Hello world", "speaker": "S0"},
            {"start": 1.0, "end": 2.0, "text": "Goodbye world", "speaker": "S0"},
        ]

        result = mod.translate_segments(segments)

        assert result[0]["text_zh"] == "你好世界"
        assert result[1]["text_zh"] == "再见世界"

    def test_parses_chinese_numbered_lines(self, monkeypatch):
        client = MagicMock()
        client.chat.completions.create.return_value = _make_response(
            "1、你好\n2、再见"
        )
        import tools.translate as mod
        monkeypatch.setattr(mod, "OpenAI", lambda **kw: client)

        segments = [
            {"start": 0.0, "end": 1.0, "text": "Hi", "speaker": "S0"},
            {"start": 1.0, "end": 2.0, "text": "Bye", "speaker": "S0"},
        ]

        result = mod.translate_segments(segments)

        assert result[0]["text_zh"] == "你好"
        assert result[1]["text_zh"] == "再见"

    def test_handles_unnumbered_lines(self, monkeypatch):
        client = MagicMock()
        client.chat.completions.create.return_value = _make_response(
            "你好\n再见"
        )
        import tools.translate as mod
        monkeypatch.setattr(mod, "OpenAI", lambda **kw: client)

        segments = [
            {"start": 0.0, "end": 1.0, "text": "Hi", "speaker": "S0"},
            {"start": 1.0, "end": 2.0, "text": "Bye", "speaker": "S0"},
        ]

        result = mod.translate_segments(segments)

        assert result[0]["text_zh"] == "你好"
        assert result[1]["text_zh"] == "再见"


class TestBatching:
    """Test batch splitting logic."""

    def test_splits_into_batches(self, monkeypatch):
        import tools.translate as mod
        monkeypatch.setattr(mod, "BATCH_SIZE", 2)

        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _make_response("1. 翻译A\n2. 翻译B"),
            _make_response("1. 翻译C"),
        ]
        monkeypatch.setattr(mod, "OpenAI", lambda **kw: client)

        segments = [
            {"start": 0.0, "end": 1.0, "text": "A", "speaker": "S0"},
            {"start": 1.0, "end": 2.0, "text": "B", "speaker": "S0"},
            {"start": 2.0, "end": 3.0, "text": "C", "speaker": "S0"},
        ]

        result = mod.translate_segments(segments)

        assert client.chat.completions.create.call_count == 2
        assert result[0]["text_zh"] == "翻译A"
        assert result[1]["text_zh"] == "翻译B"
        assert result[2]["text_zh"] == "翻译C"


class TestFallback:
    """Test fallback when parsing fails."""

    def test_keeps_original_text_on_parse_failure(self, monkeypatch):
        client = MagicMock()
        client.chat.completions.create.return_value = _make_response("1. 只有一行")
        import tools.translate as mod
        monkeypatch.setattr(mod, "OpenAI", lambda **kw: client)

        segments = [
            {"start": 0.0, "end": 1.0, "text": "First", "speaker": "S0"},
            {"start": 1.0, "end": 2.0, "text": "Second", "speaker": "S0"},
            {"start": 2.0, "end": 3.0, "text": "Third", "speaker": "S0"},
        ]

        result = mod.translate_segments(segments)

        assert result[0]["text_zh"] == "只有一行"
        assert result[1]["text_zh"] == "Second"
        assert result[2]["text_zh"] == "Third"


class TestProviderSelection:
    """Test provider/model selection behavior."""

    def test_default_provider_uses_deepseek(self, monkeypatch):
        client = MagicMock()
        client.chat.completions.create.return_value = _make_response("1. 你好")
        captured_kwargs = {}

        def _openai_factory(**kwargs):
            captured_kwargs.update(kwargs)
            return client

        import tools.translate as mod
        monkeypatch.setattr(mod, "OpenAI", _openai_factory)

        result = mod.translate_segments(
            [{"start": 0.0, "end": 1.0, "text": "Hi", "speaker": "S0"}]
        )

        assert captured_kwargs == {
            "api_key": "fake_deepseek_key",
            "base_url": "https://api.deepseek.com",
        }
        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "deepseek-chat"
        assert result[0]["text_zh"] == "你好"

    def test_openai_provider_uses_gpt5_mini_by_default(self, monkeypatch):
        client = MagicMock()
        client.chat.completions.create.return_value = _make_response("1. 你好")
        captured_kwargs = {}

        def _openai_factory(**kwargs):
            captured_kwargs.update(kwargs)
            return client

        import tools.translate as mod
        monkeypatch.setattr(mod, "OpenAI", _openai_factory)

        result = mod.translate_segments(
            [{"start": 0.0, "end": 1.0, "text": "Hi", "speaker": "S0"}],
            provider="openai",
        )

        assert captured_kwargs == {"api_key": "fake_openai_key"}
        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-5-mini"
        assert result[0]["text_zh"] == "你好"

    def test_openai_provider_supports_custom_model(self, monkeypatch):
        client = MagicMock()
        client.chat.completions.create.return_value = _make_response("1. 你好")
        import tools.translate as mod
        monkeypatch.setattr(mod, "OpenAI", lambda **kw: client)

        mod.translate_segments(
            [{"start": 0.0, "end": 1.0, "text": "Hi", "speaker": "S0"}],
            provider="openai",
            model="gpt-5-mini-2026-01-01",
        )

        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-5-mini-2026-01-01"


class TestMissingApiKey:
    """Test behavior when required API keys are not set."""

    def test_exits_without_deepseek_api_key(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

        from tools.translate import translate_segments

        with pytest.raises(SystemExit):
            translate_segments(
                [{"start": 0.0, "end": 1.0, "text": "Hi", "speaker": "S0"}]
            )

    def test_exits_without_openai_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        from tools.translate import translate_segments

        with pytest.raises(SystemExit):
            translate_segments(
                [{"start": 0.0, "end": 1.0, "text": "Hi", "speaker": "S0"}],
                provider="openai",
            )

    def test_exits_on_invalid_provider(self):
        from tools.translate import translate_segments

        with pytest.raises(SystemExit):
            translate_segments(
                [{"start": 0.0, "end": 1.0, "text": "Hi", "speaker": "S0"}],
                provider="invalid-provider",
            )
