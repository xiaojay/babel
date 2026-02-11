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


class TestSummary:
    """Test translation-summary generation."""

    def test_summary_uses_selected_model(self, monkeypatch):
        client = MagicMock()
        client.chat.completions.create.return_value = _make_response("这是一段总结。")
        captured_kwargs = {}

        def _openai_factory(**kwargs):
            captured_kwargs.update(kwargs)
            return client

        import tools.translate as mod
        monkeypatch.setattr(mod, "OpenAI", _openai_factory)

        summary = mod.summarize_translated_segments(
            [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "hello",
                    "text_zh": "你好",
                    "speaker": "S0",
                }
            ],
            provider="openai",
            model="gpt-5-mini-2026-01-01",
        )

        assert captured_kwargs == {"api_key": "fake_openai_key"}
        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-5-mini-2026-01-01"
        assert "你好" in call_kwargs["messages"][1]["content"]
        assert summary == "这是一段总结。"

    def test_summary_returns_placeholder_on_empty_segments(self, monkeypatch):
        client = MagicMock()
        import tools.translate as mod
        monkeypatch.setattr(mod, "OpenAI", lambda **kw: client)

        summary = mod.summarize_translated_segments(
            [{"start": 0.0, "end": 1.0, "text": "  ", "text_zh": "", "speaker": "S0"}]
        )

        assert summary == "（无可总结内容）"
        client.chat.completions.create.assert_not_called()


class TestDetailedSummary:
    """Test detailed translation-summary generation."""

    def test_detailed_summary_uses_chunk_merge_and_final_calls(self, monkeypatch):
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _make_response("分块一摘要"),
            _make_response("分块二摘要"),
            _make_response("合并后的综合摘要"),
            _make_response(
                "# 目录\n- 主题1：技术趋势\n\n## 主题1：技术趋势\n"
                "- 核心观点：A\n- 关键论据：B\n- 结论：C\n\n## 总结结论\n总体结论"
            ),
        ]

        import tools.translate as mod
        monkeypatch.setattr(mod, "OpenAI", lambda **kw: client)
        monkeypatch.setattr(
            mod,
            "_split_lines_into_chunks",
            lambda lines, max_chars=0, overlap_lines=0: [["第一块"], ["第二块"]],
        )

        summary = mod.summarize_translated_segments_detailed(
            [
                {
                    "start": 0.0,
                    "end": 300.0,
                    "text": "hello",
                    "text_zh": "第一段",
                    "speaker": "S0",
                },
                {
                    "start": 300.0,
                    "end": 600.0,
                    "text": "world",
                    "text_zh": "第二段",
                    "speaker": "S0",
                },
            ],
            provider="openai",
            model="gpt-5-mini-2026-01-01",
        )

        assert summary.startswith("# 目录")
        assert client.chat.completions.create.call_count == 4
        call_kwargs_list = [call.kwargs for call in client.chat.completions.create.call_args_list]
        assert "第 1/2 个分块" in call_kwargs_list[0]["messages"][1]["content"]
        assert "已有综合摘要" in call_kwargs_list[2]["messages"][1]["content"]
        assert "目标篇幅：1200-1800 字；主题数量：3-5" in call_kwargs_list[3]["messages"][1]["content"]

    def test_detailed_summary_returns_placeholder_on_empty_segments(self, monkeypatch):
        client = MagicMock()
        import tools.translate as mod
        monkeypatch.setattr(mod, "OpenAI", lambda **kw: client)

        summary = mod.summarize_translated_segments_detailed(
            [{"start": 0.0, "end": 1.0, "text": " ", "text_zh": "", "speaker": "S0"}]
        )

        assert summary == "（无可总结内容）"
        client.chat.completions.create.assert_not_called()

    @pytest.mark.parametrize(
        ("duration_seconds", "expected"),
        [
            (10 * 60, ("1200-1800", "3-5")),
            (30 * 60, ("1800-3200", "4-7")),
            (90 * 60, ("3200-5000", "6-10")),
        ],
    )
    def test_detailed_summary_profile_scales_with_duration(
        self, duration_seconds, expected
    ):
        import tools.translate as mod

        assert mod._resolve_detailed_summary_profile(duration_seconds) == expected
