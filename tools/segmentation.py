"""Step 1 后处理：按词级时间戳和说话人重新分段."""

import re

DEFAULT_SPEAKER = "SPEAKER_00"

# A speaker is never treated as spurious if it holds at least this share of speech.
MIN_SPEAKER_RATIO = 0.05
# Inside a sentence, a run of another speaker shorter than this is diarization noise.
MIN_RUN_WORDS = 8
# The aligner sometimes stretches a word or a pause over non-speech audio. Lengths
# are measured with both capped, so one bad timestamp cannot split a sentence.
MAX_WORD_SECONDS = 1.0
MAX_PAUSE_SECONDS = 1.0
# An unpunctuated WhisperX segment followed by a pause this long ends a sentence.
SEGMENT_BREAK_PAUSE = 2.0
# How strongly a speaker's share of all speech weighs on a sentence vote.
SPEAKER_SHARE_WEIGHT = 0.25
# In a passage without punctuation, a pause this long separates two clauses.
CLAUSE_PAUSE = 0.3

_SENTENCE_END_RE = re.compile(r"[.!?…]+[\"'”’)\]]*$")
_CLAUSE_END_RE = re.compile(r"[,;:，；：—–-][\"'”’)\]]*$")
_INITIAL_RE = re.compile(r"^[A-Z]\.$")
_ABBREVIATIONS = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "vs.",
    "e.g.", "i.e.", "inc.", "ltd.", "co.", "no.", "u.s.", "u.k.",
}


def _fill_missing_times(seg_words: list[dict], seg_start: float, seg_end: float) -> None:
    """WhisperX leaves words it cannot align (e.g. numerals) without timestamps."""
    for i, word in enumerate(seg_words):
        if word["start"] is None:
            word["start"] = seg_words[i - 1]["end"] if i > 0 else seg_start
        if word["end"] is None:
            next_start = next(
                (w["start"] for w in seg_words[i + 1:] if w["start"] is not None),
                None,
            )
            word["end"] = next_start if next_start is not None else seg_end
        word["start"] = float(word["start"])
        word["end"] = max(float(word["end"]), word["start"])


def _fill_missing_speakers(words: list[dict], fallback: str | None = None) -> None:
    """Give unlabeled words the speaker of the nearest labeled word."""
    last = None
    for word in words:
        if word["speaker"] is None:
            word["speaker"] = last
        else:
            last = word["speaker"]

    last = None
    for word in reversed(words):
        if word["speaker"] is None:
            word["speaker"] = last
        else:
            last = word["speaker"]

    for word in words:
        if word["speaker"] is None:
            word["speaker"] = fallback


def _flatten_words(raw_segments: list[dict]) -> list[dict]:
    """Turn WhisperX segments into one word list.

    Each word is {text, start, end, speaker, whole, last}: `whole` marks a
    segment that has no word alignment and cannot be split, `last` marks the
    final word of a WhisperX segment.
    """
    words: list[dict] = []
    for seg in raw_segments:
        seg_start = float(seg.get("start", 0.0))
        seg_end = max(float(seg.get("end", seg_start)), seg_start)
        seg_speaker = seg.get("speaker") or None

        seg_words = [
            {
                "text": w["word"].strip(),
                "start": w.get("start"),
                "end": w.get("end"),
                "speaker": w.get("speaker") or None,
                "whole": False,
                "last": False,
            }
            for w in (seg.get("words") or [])
            if (w.get("word") or "").strip()
        ]
        if not seg_words:
            text = (seg.get("text") or "").strip()
            if text:
                words.append({
                    "text": text,
                    "start": seg_start,
                    "end": seg_end,
                    "speaker": seg_speaker,
                    "whole": True,
                    "last": True,
                })
            continue

        _fill_missing_times(seg_words, seg_start, seg_end)
        _fill_missing_speakers(seg_words, fallback=seg_speaker)
        seg_words[-1]["last"] = True
        words.extend(seg_words)

    _fill_missing_speakers(words, fallback=DEFAULT_SPEAKER)
    return words


def _word_seconds(word: dict) -> float:
    duration = word["end"] - word["start"]
    return duration if word["whole"] else min(duration, MAX_WORD_SECONDS)


def _pause_seconds(before: dict, after: dict) -> float:
    return min(max(after["start"] - before["end"], 0.0), MAX_PAUSE_SECONDS)


def _speech_seconds(chunk: list[dict]) -> float:
    """Length of a chunk with stretched words and pauses capped."""
    total = sum(_word_seconds(word) for word in chunk)
    total += sum(_pause_seconds(chunk[i - 1], chunk[i]) for i in range(1, len(chunk)))
    return total


def _runs(words: list[dict]) -> list[tuple[int, int]]:
    """Index ranges [start, end) of consecutive words by the same speaker."""
    runs: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(words) + 1):
        if i == len(words) or words[i]["speaker"] != words[start]["speaker"]:
            runs.append((start, i))
            start = i
    return runs


def _has_terminal_punctuation(word: dict) -> bool:
    return bool(_SENTENCE_END_RE.search(word["text"]))


def _ends_sentence(words: list[dict], i: int) -> bool:
    word = words[i]
    if i + 1 == len(words) or word["whole"] or words[i + 1]["whole"]:
        return True

    following = words[i + 1]
    pause = following["start"] - word["end"]
    if not _has_terminal_punctuation(word):
        # WhisperX cuts its input at silences, which can fall inside a sentence.
        return word["last"] and pause >= SEGMENT_BREAK_PAUSE

    # The next word's case says nothing: Whisper writes whole passages in lowercase.
    if word["text"].lower() in _ABBREVIATIONS or _INITIAL_RE.match(word["text"]):
        return pause >= 0.5
    return True


def _split_sentences(words: list[dict]) -> list[list[dict]]:
    sentences: list[list[dict]] = []
    current: list[dict] = []
    for i, word in enumerate(words):
        current.append(word)
        if _ends_sentence(words, i):
            sentences.append(current)
            current = []
    return sentences


def _speaker_totals(words: list[dict]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for word in words:
        totals[word["speaker"]] = totals.get(word["speaker"], 0.0) + _word_seconds(word)
    return totals


def _majority_speaker(chunk: list[dict], shares: dict[str, float]) -> str:
    """Votes are weighted by each speaker's share of all speech, so a speaker who
    is rarely heard needs a clearer majority than one of the hosts.
    """
    counts: dict[str, int] = {}
    for word in chunk:
        counts[word["speaker"]] = counts.get(word["speaker"], 0) + 1
    return max(
        counts,
        key=lambda speaker: counts[speaker] * shares[speaker] ** SPEAKER_SHARE_WEIGHT,
    )


def _split_clauses(sentence: list[dict]) -> list[list[dict]]:
    clauses = [[sentence[0]]]
    for before, after in zip(sentence, sentence[1:]):
        if after["start"] - before["end"] >= CLAUSE_PAUSE:
            clauses.append([])
        clauses[-1].append(after)
    return clauses


def _vote_sentence_speaker(
    sentence: list[dict], shares: dict[str, float], max_seconds: float
) -> None:
    """Give a sentence to the speaker who says most of it.

    Diarization boundaries are often off by a few words, which would cut
    "Do you have meetings like that?" into "Do you have" / "meetings like that?".
    A long run by another speaker is kept: that is a real change of speaker.
    """
    if len(_runs(sentence)) < 2:
        return

    if _speech_seconds(sentence) > max_seconds:
        # Whisper wrote this passage without punctuation, so it may hold several
        # turns. Pauses are the only boundaries left: vote clause by clause.
        for clause in _split_clauses(sentence):
            speaker = _majority_speaker(clause, shares)
            for word in clause:
                word["speaker"] = speaker
        return

    majority = _majority_speaker(sentence, shares)
    for start, end in _runs(sentence):
        if sentence[start]["speaker"] != majority and end - start < MIN_RUN_WORDS:
            for word in sentence[start:end]:
                word["speaker"] = majority


def _merge_minor_speakers(words: list[dict], min_speaker_seconds: float) -> None:
    """Fold speakers with almost no speech into the speaker talking next to them.

    Diarization tends to invent extra speakers from laughter, overlap or noise.
    Each one would otherwise get its own (unusably short) voice-clone reference.
    """
    totals = _speaker_totals(words)
    if min_speaker_seconds <= 0 or len(totals) < 2:
        return

    threshold = min(min_speaker_seconds, MIN_SPEAKER_RATIO * sum(totals.values()))
    minor = {speaker for speaker, total in totals.items() if total < threshold}
    if len(minor) == len(totals):
        minor.discard(max(totals, key=totals.get))
    if not minor:
        return

    for start, end in _runs(words):
        if words[start]["speaker"] not in minor:
            continue

        prev_word = next(
            (words[i] for i in range(start - 1, -1, -1) if words[i]["speaker"] not in minor),
            None,
        )
        next_word = next(
            (words[i] for i in range(end, len(words)) if words[i]["speaker"] not in minor),
            None,
        )
        if prev_word is None:
            target = next_word["speaker"]
        elif next_word is None:
            target = prev_word["speaker"]
        else:
            gap_before = words[start]["start"] - prev_word["end"]
            gap_after = next_word["start"] - words[end - 1]["end"]
            target = prev_word["speaker"] if gap_before <= gap_after else next_word["speaker"]

        for i in range(start, end):
            words[i]["speaker"] = target


def _split_long(
    chunk: list[dict], max_seconds: float, min_seconds: float
) -> list[list[dict]]:
    """Cut an overlong sentence at its best pause or clause boundary."""
    total = _speech_seconds(chunk)
    if len(chunk) < 2 or total <= max_seconds:
        return [chunk]

    best_index = None
    best_score = 0.0
    left = 0.0
    for i in range(1, len(chunk)):
        left += _word_seconds(chunk[i - 1])
        pause = _pause_seconds(chunk[i - 1], chunk[i])
        right = total - left - pause
        if left >= min_seconds and right >= min_seconds:
            score = pause - abs(left / total - 0.5)
            if _CLAUSE_END_RE.search(chunk[i - 1]["text"]) or _has_terminal_punctuation(chunk[i - 1]):
                score += 0.5
            if best_index is None or score > best_score:
                best_index, best_score = i, score
        left += pause

    if best_index is None:
        return [chunk]
    return (
        _split_long(chunk[:best_index], max_seconds, min_seconds)
        + _split_long(chunk[best_index:], max_seconds, min_seconds)
    )


def _pack(
    sentences: list[list[dict]],
    max_seconds: float,
    target_seconds: float,
    min_seconds: float,
    max_merge_gap: float,
) -> list[list[dict]]:
    """Join neighbouring sentences of one turn into TTS-sized chunks."""
    chunks: list[list[dict]] = []
    current: list[dict] = []
    for sentence in sentences:
        if not current:
            current = list(sentence)
            continue

        gap = sentence[0]["start"] - current[-1]["end"]
        current_seconds = _speech_seconds(current)
        sentence_seconds = _speech_seconds(sentence)
        merged = current_seconds + sentence_seconds + _pause_seconds(current[-1], sentence[0])
        too_short = current_seconds < min_seconds or sentence_seconds < min_seconds
        if gap <= max_merge_gap and merged <= max_seconds and (
            too_short or merged <= target_seconds
        ):
            current.extend(sentence)
        else:
            chunks.append(current)
            current = list(sentence)
    if current:
        chunks.append(current)
    return chunks


def resegment(
    raw_segments: list[dict],
    min_speaker_seconds: float = 15.0,
    max_segment_seconds: float = 15.0,
    target_segment_seconds: float = 8.0,
    min_segment_seconds: float = 2.0,
    max_merge_gap: float = 1.0,
) -> list[dict]:
    """Rebuild segments from WhisperX word-level timestamps and speakers.

    Every returned segment has a single speaker. Sentences are kept whole and
    given to the speaker who says most of them, short ones are joined with
    their neighbours, and overlong ones are cut at pauses.

    Returns a list of segments: [{start, end, text, speaker}, ...]
    """
    words = _flatten_words(raw_segments)
    if not words:
        return []

    totals = _speaker_totals(words)
    all_speech = sum(totals.values()) or 1.0
    shares = {speaker: total / all_speech for speaker, total in totals.items()}

    sentences = _split_sentences(words)
    for sentence in sentences:
        _vote_sentence_speaker(sentence, shares, max_segment_seconds)
    _merge_minor_speakers(words, min_speaker_seconds)

    # A sentence still holding two speakers is a real change: cut it there.
    parts = [
        sentence[start:end]
        for sentence in sentences
        for start, end in _runs(sentence)
    ]

    segments: list[dict] = []
    turn: list[list[dict]] = []
    for i, part in enumerate(parts):
        turn.extend(_split_long(part, max_segment_seconds, min_segment_seconds))
        is_last = i + 1 == len(parts)
        if not is_last and parts[i + 1][0]["speaker"] == part[0]["speaker"]:
            continue

        for chunk in _pack(
            turn,
            max_seconds=max_segment_seconds,
            target_seconds=target_segment_seconds,
            min_seconds=min_segment_seconds,
            max_merge_gap=max_merge_gap,
        ):
            text = " ".join(w["text"] for w in chunk)
            # Whisper sometimes emits a line of dots; there is nothing to say in it.
            if not any(ch.isalnum() for ch in text):
                continue
            segments.append({
                "start": round(min(w["start"] for w in chunk), 3),
                "end": round(max(w["end"] for w in chunk), 3),
                "text": text,
                "speaker": chunk[0]["speaker"],
            })
        turn = []
    return segments
