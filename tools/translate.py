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
    model_name = model.strip() if model else default_model

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
