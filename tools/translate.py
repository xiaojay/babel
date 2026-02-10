"""Step 3: LLM 翻译（DeepSeek / OpenAI）."""

import os
import sys

from openai import OpenAI

TRANSLATE_SYSTEM_PROMPT = (
    "你是一位专业的播客翻译。请将以下英文播客内容翻译成中文。\n"
    "要求：\n"
    "1. 保持口语化、自然的播客风格\n"
    "2. 人名可保留英文或音译\n"
    "3. 只返回翻译后的文本，每行对应一个原文片段\n"
    "4. 不要添加任何解释或标注"
)

BATCH_SIZE = 20  # segments per API call
SUMMARY_MAX_SEGMENTS = 300
SUMMARY_MAX_CHARS = 24000
SUMMARY_SYSTEM_PROMPT = (
    "你是一位资深中文播客编辑。请基于播客稿内容输出简洁、准确的中文总结。\n"
    "要求：\n"
    "1. 使用简体中文，控制在300-500字\n"
    "2. 覆盖核心观点、关键论据和主要结论\n"
    "3. 不杜撰信息，不输出与材料无关内容\n"
    "4. 仅输出总结正文，不要标题、编号或Markdown"
)

TRANSLATE_PROVIDERS = {
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-chat",
    },
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url": None,
        "default_model": "gpt-5-mini",
    },
}


def _build_translate_client(provider: str) -> tuple[OpenAI, str]:
    config = TRANSLATE_PROVIDERS.get(provider)
    if config is None:
        supported = ", ".join(TRANSLATE_PROVIDERS.keys())
        print(
            f"错误: 不支持的翻译提供方 {provider!r}，可选: {supported}",
            file=sys.stderr,
        )
        sys.exit(1)

    api_key_env = config["api_key_env"]
    api_key = os.getenv(api_key_env)
    if not api_key:
        print(f"错误: 未设置 {api_key_env}", file=sys.stderr)
        sys.exit(1)

    client_kwargs = {"api_key": api_key}
    base_url = config["base_url"]
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)
    return client, config["default_model"]


def _resolve_model_name(model: str | None, default_model: str) -> str:
    if model and model.strip():
        return model.strip()
    return default_model


def _collect_summary_source_lines(
    segments: list[dict],
    max_segments: int = SUMMARY_MAX_SEGMENTS,
    max_chars: int = SUMMARY_MAX_CHARS,
) -> tuple[list[str], int]:
    lines: list[str] = []
    for seg in segments:
        text = (seg.get("text_zh") or seg.get("text") or "").strip()
        if text:
            lines.append(text)

    total_lines = len(lines)
    if total_lines == 0:
        return [], 0

    if total_lines > max_segments:
        sampled = [lines[int(i * total_lines / max_segments)] for i in range(max_segments)]
        sampled[-1] = lines[-1]
    else:
        sampled = lines

    picked: list[str] = []
    current_chars = 0
    for line in sampled:
        extra = len(line) + 1
        if picked and current_chars + extra > max_chars:
            break
        if not picked and len(line) > max_chars:
            picked.append(line[:max_chars])
            break
        picked.append(line)
        current_chars += extra

    return picked, total_lines


def translate_segments(
    segments: list[dict],
    provider: str = "deepseek",
    model: str | None = None,
) -> list[dict]:
    """Translate segment texts from English to Chinese via configurable LLM API.

    Returns segments with an added 'text_zh' field.
    """
    provider = provider.strip().lower()
    client, default_model = _build_translate_client(provider)
    model_name = _resolve_model_name(model, default_model)

    print(f"[Step 3] 使用 {provider}:{model_name} 翻译 {len(segments)} 个片段...")
    translated = list(segments)  # shallow copy

    for i in range(0, len(segments), BATCH_SIZE):
        batch = segments[i : i + BATCH_SIZE]
        numbered_lines = "\n".join(
            f"{j + 1}. {seg['text']}" for j, seg in enumerate(batch)
        )
        user_msg = (
            f"请翻译以下 {len(batch)} 个片段（保持编号对应）：\n\n{numbered_lines}"
        )

        request_kwargs = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": TRANSLATE_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        }
        # gpt-5-mini only supports the default temperature value; avoid sending it.
        if provider == "deepseek":
            request_kwargs["temperature"] = 0.3

        response = client.chat.completions.create(**request_kwargs)

        reply = response.choices[0].message.content.strip()
        lines = reply.splitlines()

        # Parse numbered lines: strip leading "1. ", "2. " etc.
        parsed: list[str] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Remove leading number and punctuation like "1. " or "1、"
            for sep in [". ", "、", "。", ") ", "） "]:
                idx = line.find(sep)
                if idx != -1 and line[:idx].isdigit():
                    line = line[idx + len(sep):]
                    break
            parsed.append(line)

        # Assign translations back
        for j, seg_idx in enumerate(range(i, i + len(batch))):
            if j < len(parsed):
                translated[seg_idx]["text_zh"] = parsed[j]
            else:
                # Fallback: keep original if parsing failed
                translated[seg_idx]["text_zh"] = batch[j]["text"]
                print(f"  警告: 片段 {seg_idx} 翻译解析失败，保留原文")

        print(f"  已翻译 {min(i + BATCH_SIZE, len(segments))}/{len(segments)}")

    return translated


def summarize_translated_segments(
    segments: list[dict],
    provider: str = "deepseek",
    model: str | None = None,
) -> str:
    """Summarize translated segments into a short Chinese text."""
    provider = provider.strip().lower()
    client, default_model = _build_translate_client(provider)
    model_name = _resolve_model_name(model, default_model)
    source_lines, total_lines = _collect_summary_source_lines(segments)

    if not source_lines:
        return "（无可总结内容）"

    print(f"[Step 3.5] 使用 {provider}:{model_name} 生成翻译总结...")

    source_text = "\n".join(f"{i + 1}. {line}" for i, line in enumerate(source_lines))
    sampled_hint = ""
    if total_lines > len(source_lines):
        sampled_hint = (
            f"说明：原始稿件共 {total_lines} 段，以下为抽样后的 {len(source_lines)} 段片段。\n"
        )
    user_msg = (
        "请根据以下中文播客稿内容写总结。\n"
        f"{sampled_hint}"
        "输出要求：300-500字，信息准确，不要标题和编号。\n\n"
        f"{source_text}"
    )

    request_kwargs = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    }
    if provider == "deepseek":
        request_kwargs["temperature"] = 0.3

    response = client.chat.completions.create(**request_kwargs)
    summary = (response.choices[0].message.content or "").strip()
    if not summary:
        return "（总结生成失败：模型未返回内容）"
    return summary
