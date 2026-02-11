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
DETAILED_SUMMARY_CHUNK_MAX_CHARS = 12000
DETAILED_SUMMARY_CHUNK_OVERLAP_LINES = 6
DETAILED_SUMMARY_INTERMEDIATE_MAX_CHARS = 2200
DETAILED_SUMMARY_CHUNK_SYSTEM_PROMPT = (
    "你是一位资深中文播客编辑。请把给定播客片段块提炼成结构化主题摘要，"
    "用于后续跨分块合并。"
)
DETAILED_SUMMARY_MERGE_SYSTEM_PROMPT = (
    "你是一位资深中文播客编辑。请把已有摘要与新增摘要合并为去重且更完整的综合摘要。"
)
DETAILED_SUMMARY_FINAL_SYSTEM_PROMPT = (
    "你是一位资深中文播客编辑。请输出详细、准确、结构清晰的中文播客总结。"
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


def _collect_segment_text_lines(segments: list[dict]) -> list[str]:
    lines: list[str] = []
    for seg in segments:
        text = (seg.get("text_zh") or seg.get("text") or "").strip()
        if text:
            lines.append(text)
    return lines


def _create_chat_completion(
    client: OpenAI,
    provider: str,
    model_name: str,
    system_prompt: str,
    user_msg: str,
) -> str:
    request_kwargs = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
    }
    if provider == "deepseek":
        request_kwargs["temperature"] = 0.3

    response = client.chat.completions.create(**request_kwargs)
    return (response.choices[0].message.content or "").strip()


def _collect_summary_source_lines(
    segments: list[dict],
    max_segments: int = SUMMARY_MAX_SEGMENTS,
    max_chars: int = SUMMARY_MAX_CHARS,
) -> tuple[list[str], int]:
    lines = _collect_segment_text_lines(segments)
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


def _estimate_audio_duration_seconds(segments: list[dict]) -> float:
    starts: list[float] = []
    ends: list[float] = []
    for seg in segments:
        start = seg.get("start")
        end = seg.get("end")
        if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end >= start:
            starts.append(float(start))
            ends.append(float(end))
    if not starts:
        return 0.0
    return max(0.0, max(ends) - min(starts))


def _resolve_detailed_summary_profile(duration_seconds: float) -> tuple[str, str]:
    if duration_seconds < 20 * 60:
        return "1200-1800", "3-5"
    if duration_seconds <= 60 * 60:
        return "1800-3200", "4-7"
    return "3200-5000", "6-10"


def _split_lines_into_chunks(
    lines: list[str],
    max_chars: int = DETAILED_SUMMARY_CHUNK_MAX_CHARS,
    overlap_lines: int = DETAILED_SUMMARY_CHUNK_OVERLAP_LINES,
) -> list[list[str]]:
    if not lines:
        return []

    chunks: list[list[str]] = []
    start = 0
    total_lines = len(lines)

    while start < total_lines:
        chunk: list[str] = []
        chunk_chars = 0
        idx = start

        while idx < total_lines:
            line = lines[idx]
            line_in_chunk = line if len(line) <= max_chars else line[:max_chars]
            extra = len(line_in_chunk) + 1
            if chunk and chunk_chars + extra > max_chars:
                break
            chunk.append(line_in_chunk)
            chunk_chars += extra
            idx += 1
            if chunk_chars >= max_chars:
                break

        if not chunk:
            chunk = [lines[start][:max_chars]]
            idx = start + 1

        chunks.append(chunk)
        if idx >= total_lines:
            break
        start = max(start + 1, idx - overlap_lines)

    return chunks


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
    summary = _create_chat_completion(
        client=client,
        provider=provider,
        model_name=model_name,
        system_prompt=SUMMARY_SYSTEM_PROMPT,
        user_msg=user_msg,
    )
    if not summary:
        return "（总结生成失败：模型未返回内容）"
    return summary


def summarize_translated_segments_detailed(
    segments: list[dict],
    provider: str = "deepseek",
    model: str | None = None,
) -> str:
    """Summarize translated segments into a detailed Chinese report."""
    provider = provider.strip().lower()
    client, default_model = _build_translate_client(provider)
    model_name = _resolve_model_name(model, default_model)
    source_lines = _collect_segment_text_lines(segments)

    if not source_lines:
        return "（无可总结内容）"

    duration_seconds = _estimate_audio_duration_seconds(segments)
    target_chars, topic_range = _resolve_detailed_summary_profile(duration_seconds)
    chunks = _split_lines_into_chunks(source_lines)

    print(
        f"[Step 3.6] 使用 {provider}:{model_name} 生成详细总结..."
        f"（分块 {len(chunks)}）"
    )

    chunk_summaries: list[str] = []
    for idx, chunk_lines in enumerate(chunks):
        source_text = "\n".join(f"{i + 1}. {line}" for i, line in enumerate(chunk_lines))
        user_msg = (
            f"这是中文播客稿的第 {idx + 1}/{len(chunks)} 个分块。\n"
            "请提炼该分块的主题信息，供后续汇总。\n"
            "输出要求：\n"
            "1. 仅基于给定内容，不杜撰\n"
            "2. 给出 2-5 个主题候选；每个主题包含：主题名、核心观点、关键论据、阶段性结论\n"
            "3. 保留关键人名、术语与观点差异\n"
            f"4. 总长度控制在 {DETAILED_SUMMARY_INTERMEDIATE_MAX_CHARS} 字以内\n"
            "5. 输出纯文本，不要 Markdown\n\n"
            f"{source_text}"
        )
        chunk_summary = _create_chat_completion(
            client=client,
            provider=provider,
            model_name=model_name,
            system_prompt=DETAILED_SUMMARY_CHUNK_SYSTEM_PROMPT,
            user_msg=user_msg,
        )
        if not chunk_summary:
            chunk_summary = "（该分块未返回可用摘要）"
        chunk_summaries.append(chunk_summary)
        print(f"  已处理详细分块 {idx + 1}/{len(chunks)}")

    rolling_summary = chunk_summaries[0]
    for idx in range(1, len(chunk_summaries)):
        merge_user_msg = (
            "请将“已有综合摘要”和“新增分块摘要”合并成更完整的一版。\n"
            "输出要求：\n"
            "1. 去重并合并相近主题\n"
            "2. 保留核心观点、关键论据和结论\n"
            f"3. 总长度控制在 {DETAILED_SUMMARY_INTERMEDIATE_MAX_CHARS} 字以内\n"
            "4. 输出纯文本，不要 Markdown\n\n"
            f"已有综合摘要：\n{rolling_summary}\n\n"
            f"新增分块摘要：\n{chunk_summaries[idx]}"
        )
        merged_summary = _create_chat_completion(
            client=client,
            provider=provider,
            model_name=model_name,
            system_prompt=DETAILED_SUMMARY_MERGE_SYSTEM_PROMPT,
            user_msg=merge_user_msg,
        )
        if merged_summary:
            rolling_summary = merged_summary
        else:
            rolling_summary = (
                f"{rolling_summary}\n\n{chunk_summaries[idx]}"
            ).strip()

    duration_minutes = duration_seconds / 60 if duration_seconds > 0 else 0.0
    final_user_msg = (
        "请基于以下综合摘要，输出最终详细总结。\n"
        f"播客时长约 {duration_minutes:.1f} 分钟。\n"
        f"目标篇幅：{target_chars} 字；主题数量：{topic_range}。\n"
        "格式要求（Markdown）：\n"
        "1. 先输出 '# 目录'，列出主题清单\n"
        "2. 再按 '## 主题X：标题' 展开，每个主题需包含：核心观点、关键论据、结论\n"
        "3. 最后输出 '## 总结结论'\n"
        "4. 内容要准确，不杜撰，不要输出与材料无关内容\n\n"
        f"{rolling_summary}"
    )
    detailed_summary = _create_chat_completion(
        client=client,
        provider=provider,
        model_name=model_name,
        system_prompt=DETAILED_SUMMARY_FINAL_SYSTEM_PROMPT,
        user_msg=final_user_msg,
    )
    if not detailed_summary:
        return "（详细总结生成失败：模型未返回内容）"
    return detailed_summary
