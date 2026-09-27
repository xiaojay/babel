"""Tests for tools.segmentation."""

from tools.segmentation import DEFAULT_SPEAKER, resegment


def _words(text: str, start: float, speaker: str | None, step: float = 0.4) -> list[dict]:
    """Build evenly spaced WhisperX-style words: each lasts 0.3s, then a 0.1s gap."""
    words = []
    for i, token in enumerate(text.split()):
        word = {
            "word": token,
            "start": round(start + i * step, 3),
            "end": round(start + i * step + 0.3, 3),
        }
        if speaker is not None:
            word["speaker"] = speaker
        words.append(word)
    return words


def _segment(words: list[dict], speaker: str | None = None) -> dict:
    seg = {
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "text": " ".join(w["word"] for w in words),
        "words": words,
    }
    if speaker is not None:
        seg["speaker"] = speaker
    return seg


def _long_turn(speaker: str, start: float, sentences: int = 12) -> list[dict]:
    """About 4s per sentence, so a speaker easily clears the minor-speaker threshold."""
    words: list[dict] = []
    for i in range(sentences):
        words += _words(
            "This is a full sentence with ten words in it.",
            start + i * 4.5,
            speaker,
        )
    return words


class TestSpeakerSplit:
    """Every output segment has exactly one speaker."""

    def test_splits_segment_at_speaker_change(self):
        a = _long_turn("SPEAKER_00", 0.0)
        b = _long_turn("SPEAKER_01", 60.0)
        # One WhisperX segment that spans both speakers.
        mixed = _segment(a[-10:] + b[:10], speaker="SPEAKER_00")
        raw = [_segment(a[:-10], "SPEAKER_00"), mixed, _segment(b[10:], "SPEAKER_01")]

        result = resegment(raw)

        speakers = [s["speaker"] for s in result]
        assert set(speakers) == {"SPEAKER_00", "SPEAKER_01"}
        # Speakers never interleave: all of A's segments come before B's.
        assert speakers == sorted(speakers)
        assert all(s["end"] <= 60.0 for s in result if s["speaker"] == "SPEAKER_00")
        assert all(s["start"] >= 60.0 for s in result if s["speaker"] == "SPEAKER_01")

    def test_keeps_all_words_in_order(self):
        a = _long_turn("SPEAKER_00", 0.0, sentences=6)
        b = _long_turn("SPEAKER_01", 30.0, sentences=6)
        raw = [_segment(a, "SPEAKER_00"), _segment(b, "SPEAKER_01")]

        result = resegment(raw)

        original = [w["word"] for w in a + b]
        assert " ".join(s["text"] for s in result).split() == original


class TestMerging:
    """Fragments by one speaker are joined into sentence-sized segments."""

    def test_merges_short_fragments_of_same_speaker(self):
        raw = [
            _segment(_words("So I think", 0.0, "SPEAKER_00")),
            _segment(_words("that this", 1.2, "SPEAKER_00")),
            _segment(_words("really works.", 2.0, "SPEAKER_00")),
        ]

        result = resegment(raw)

        assert len(result) == 1
        assert result[0]["text"] == "So I think that this really works."
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 2.7

    def test_does_not_merge_across_long_pause(self):
        raw = [
            _segment(_words("First part here.", 0.0, "SPEAKER_00")),
            _segment(_words("Second part here.", 5.0, "SPEAKER_00")),
        ]

        result = resegment(raw)

        assert [s["text"] for s in result] == ["First part here.", "Second part here."]

    def test_stops_merging_at_target_length(self):
        raw = [_segment(_long_turn("SPEAKER_00", 0.0, sentences=6))]

        result = resegment(raw, target_segment_seconds=8.0)

        assert len(result) > 1
        assert all(s["end"] - s["start"] <= 15.0 for s in result)
        # Segments end on sentence boundaries.
        assert all(s["text"].endswith(".") for s in result)

    def test_splits_overlong_sentence_at_pause(self):
        first = _words("one two three four five six seven eight nine ten,", 0.0, "SPEAKER_00", step=1.0)
        second = _words("eleven twelve thirteen fourteen fifteen sixteen seventeen.", 11.0, "SPEAKER_00", step=1.0)

        result = resegment([_segment(first + second)], max_segment_seconds=15.0)

        assert len(result) == 2
        assert result[0]["text"].endswith("ten,")
        assert result[1]["text"].startswith("eleven")

    def test_does_not_split_on_abbreviation(self):
        raw = [_segment(_words("I met Dr. Smith and J. Doe today.", 0.0, "SPEAKER_00"))]

        result = resegment(raw, target_segment_seconds=0.0, min_segment_seconds=0.0)

        assert len(result) == 1


class TestMinorSpeakers:
    """Speakers with almost no speech are folded into their neighbour."""

    def test_merges_speaker_with_almost_no_speech(self):
        a = _long_turn("SPEAKER_00", 0.0)
        ghost = _words("uh huh", 54.5, "SPEAKER_07")
        b = _long_turn("SPEAKER_01", 60.0)
        raw = [_segment(a + ghost, "SPEAKER_00"), _segment(b, "SPEAKER_01")]

        result = resegment(raw, min_speaker_seconds=15.0)

        assert {s["speaker"] for s in result} == {"SPEAKER_00", "SPEAKER_01"}
        assert "uh huh" in " ".join(s["text"] for s in result if s["speaker"] == "SPEAKER_00")

    def test_minor_speaker_joins_closest_neighbour_in_time(self):
        a = _long_turn("SPEAKER_00", 0.0)
        ghost = _words("right okay.", 59.0, "SPEAKER_07")
        b = _long_turn("SPEAKER_01", 60.0)

        result = resegment([_segment(a + ghost + b)], min_speaker_seconds=15.0)

        owner = next(s["speaker"] for s in result if "right okay." in s["text"])
        assert owner == "SPEAKER_01"

    def test_zero_threshold_disables_merging(self):
        a = _long_turn("SPEAKER_00", 0.0)
        ghost = _words("Uh huh.", 54.5, "SPEAKER_07")

        result = resegment([_segment(a + ghost)], min_speaker_seconds=0)

        assert "SPEAKER_07" in {s["speaker"] for s in result}

    def test_short_clip_keeps_both_speakers(self):
        raw = [
            _segment(_words("Hello there.", 0.0, "SPEAKER_01")),
            _segment(_words("Hi, how are you?", 1.5, "SPEAKER_02")),
        ]

        result = resegment(raw, min_speaker_seconds=15.0)

        assert [s["speaker"] for s in result] == ["SPEAKER_01", "SPEAKER_02"]


class TestSentenceSpeaker:
    """A sentence goes to the speaker who says most of it."""

    def test_relabels_stray_word_inside_sentence(self):
        a = _long_turn("SPEAKER_00", 0.0)
        b = _long_turn("SPEAKER_01", 60.0)
        a[3]["speaker"] = "SPEAKER_01"  # one word mid-sentence

        result = resegment([_segment(a), _segment(b)])

        speakers = [s["speaker"] for s in result]
        assert speakers == sorted(speakers)

    def test_keeps_short_reply_as_its_own_turn(self):
        a = _long_turn("SPEAKER_00", 0.0, sentences=6)
        reply = _words("Yeah.", 27.2, "SPEAKER_01")
        a_more = _long_turn("SPEAKER_00", 28.0, sentences=6)
        b = _long_turn("SPEAKER_01", 60.0)

        result = resegment([_segment(a + reply + a_more), _segment(b)])

        assert any(s["text"] == "Yeah." and s["speaker"] == "SPEAKER_01" for s in result)

    def test_moves_sentence_tail_back_to_its_speaker(self):
        a = _long_turn("SPEAKER_00", 0.0)
        b = _long_turn("SPEAKER_01", 60.0)
        for word in a[-3:]:  # "words in it." attributed to the next speaker
            word["speaker"] = "SPEAKER_01"

        result = resegment([_segment(a + b)])

        last_a = [s for s in result if s["speaker"] == "SPEAKER_00"][-1]
        first_b = [s for s in result if s["speaker"] == "SPEAKER_01"][0]
        assert last_a["text"].endswith("ten words in it.")
        assert first_b["text"].startswith("This is")

    def test_moves_sentence_head_forward_to_its_speaker(self):
        a = _long_turn("SPEAKER_00", 0.0)
        b = _long_turn("SPEAKER_01", 60.0)
        for word in b[:3]:  # "This is a" attributed to the previous speaker
            word["speaker"] = "SPEAKER_00"

        result = resegment([_segment(a + b)])

        last_a = [s for s in result if s["speaker"] == "SPEAKER_00"][-1]
        first_b = [s for s in result if s["speaker"] == "SPEAKER_01"][0]
        assert last_a["text"].endswith("in it.")
        assert first_b["text"].startswith("This is a full")

    def test_rarely_heard_speaker_needs_clear_majority(self):
        host = _long_turn("SPEAKER_00", 0.0, sentences=40)
        caller = _long_turn("SPEAKER_01", 200.0, sentences=5)
        sentence = _words("The Manhattan Project was a government project.", 190.0, "SPEAKER_00")
        for word in sentence[1:5]:  # four of seven words carry the caller's label
            word["speaker"] = "SPEAKER_01"

        result = resegment([_segment(host), _segment(sentence), _segment(caller)])

        owner = next(s["speaker"] for s in result if "Manhattan" in s["text"])
        assert owner == "SPEAKER_00"

    def test_keeps_long_run_by_other_speaker(self):
        a = _words("and so what I wanted to say about all of this is that", 0.0, "SPEAKER_00")
        b = _words("well hold on because I think we should look at the numbers first.", 5.4, "SPEAKER_01")

        result = resegment([_segment(a + b)], min_speaker_seconds=0)

        assert [s["speaker"] for s in result] == ["SPEAKER_00", "SPEAKER_01"]
        assert result[0]["text"].endswith("is that")
        assert result[1]["text"].startswith("well hold on")


class TestUnpunctuatedPassage:
    """Whisper sometimes writes long passages without any punctuation."""

    def _passage(self):
        a = _words(
            "the coolest meeting i had this week was a guy who used it to make"
            " a custom vaccine for his dog and it took him only a few weeks"
            " which is just incredible when you think about how long that used to take",
            0.0, "SPEAKER_00",
        )
        reply = _words("whoa", a[-1]["end"] + 0.5, "SPEAKER_01")
        question = _words(
            "do you have meetings like that every week", reply[-1]["end"] + 0.6, "SPEAKER_01"
        )
        for word in question[:3]:  # diarization is late by three words
            word["speaker"] = "SPEAKER_00"
        answer = _words(
            "for whatever reason this one hit me just hearing him tell the story",
            question[-1]["end"] + 0.7, "SPEAKER_00",
        )
        # Both speakers talk a lot in the rest of the episode.
        rest = _long_turn("SPEAKER_01", answer[-1]["end"] + 5.0, sentences=8)
        return [_segment(a + reply + question + answer), _segment(rest)]

    def test_reply_between_pauses_keeps_its_speaker(self):
        result = resegment(self._passage())

        assert any(
            s["speaker"] == "SPEAKER_01" and s["text"].startswith("whoa") for s in result
        )

    def test_clause_goes_to_the_speaker_who_says_most_of_it(self):
        result = resegment(self._passage())

        owner = next(s["speaker"] for s in result if "meetings like that" in s["text"])
        assert owner == "SPEAKER_01"
        assert not any(
            s["speaker"] == "SPEAKER_00" and "do you have" in s["text"] for s in result
        )


class TestSentenceIntegrity:
    """Sentences are not cut by alignment artifacts."""

    def test_joins_sentence_cut_by_whisperx_segment(self):
        raw = [
            _segment(_words("The dream that I have is", 0.0, "SPEAKER_00")),
            _segment(_words("that it just works.", 3.6, "SPEAKER_00")),
        ]

        result = resegment(raw)

        assert [s["text"] for s in result] == ["The dream that I have is that it just works."]

    def test_long_silence_after_unpunctuated_segment_ends_sentence(self):
        raw = [
            _segment(_words("Thank you very much", 0.0, "SPEAKER_00")),
            _segment(_words("So let us begin.", 6.0, "SPEAKER_00")),
        ]

        result = resegment(raw)

        assert [s["text"] for s in result] == ["Thank you very much", "So let us begin."]

    def test_stretched_word_does_not_split_sentence(self):
        words = _words("I will separate what we did from how we did it.", 0.0, "SPEAKER_00")
        words[2]["end"] = words[2]["start"] + 9.0  # aligner stretched one word
        for word in words[3:]:
            word["start"] += 15.0
            word["end"] += 15.0

        result = resegment([_segment(words)])

        assert len(result) == 1
        assert result[0]["text"] == "I will separate what we did from how we did it."


class TestIncompleteInput:
    """WhisperX output is not always fully aligned or labeled."""

    def test_empty_input(self):
        assert resegment([]) == []

    def test_no_diarization_uses_default_speaker(self):
        raw = [_segment(_words("Hello world.", 0.0, None))]

        result = resegment(raw)

        assert [s["speaker"] for s in result] == [DEFAULT_SPEAKER]

    def test_segment_without_words_is_kept_whole(self):
        raw = [{"start": 0.0, "end": 1.5, "text": " Hello world ", "speaker": "SPEAKER_00"}]

        result = resegment(raw)

        assert result == [
            {"start": 0.0, "end": 1.5, "text": "Hello world", "speaker": "SPEAKER_00"}
        ]

    def test_word_without_timestamps_is_kept(self):
        words = _words("It cost 2014 dollars.", 0.0, "SPEAKER_00")
        for key in ("start", "end", "speaker"):
            del words[2][key]

        result = resegment([_segment(words, "SPEAKER_00")])

        assert len(result) == 1
        assert result[0]["text"] == "It cost 2014 dollars."
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == words[-1]["end"]

    def test_unlabeled_word_takes_neighbouring_speaker(self):
        a = _long_turn("SPEAKER_00", 0.0)
        b = _long_turn("SPEAKER_01", 60.0)
        del a[5]["speaker"]

        result = resegment([_segment(a, "SPEAKER_00"), _segment(b, "SPEAKER_01")])

        assert {s["speaker"] for s in result} == {"SPEAKER_00", "SPEAKER_01"}
