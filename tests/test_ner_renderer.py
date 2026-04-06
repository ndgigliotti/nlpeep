"""Tests for NER tagged-sequence detection and rendering."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.widgets import Static

from nlpeep.data import RecordStore
from nlpeep.renderers import (
    ValueType,
    _render_tagged_sequence,
    classify_value,
)
from nlpeep.schema import FieldArchetype, SchemaMapping

_REF = Path(__file__).parent / "reference_data"


# -- Auto-detection of conll2003 NER data ------------------------------------


class TestNerAutoDetection:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        store = RecordStore.load(_REF / "conll2003_ner.jsonl")
        self.store = store
        self.mapping = SchemaMapping.auto_detect(store.sample())

    def test_tagged_sequence_pair_detected(self) -> None:
        pairs = self.mapping.tagged_sequence_pairs()
        assert len(pairs) == 1
        token_m, tag_m = pairs[0]
        assert token_m.json_path == "tokens"
        assert tag_m.json_path == "tags"

    def test_archetype_is_tagged_sequence(self) -> None:
        pairs = self.mapping.tagged_sequence_pairs()
        token_m, tag_m = pairs[0]
        assert token_m.archetype == FieldArchetype.TAGGED_SEQUENCE
        assert tag_m.archetype == FieldArchetype.TAGGED_SEQUENCE

    def test_paired_with_set_correctly(self) -> None:
        pairs = self.mapping.tagged_sequence_pairs()
        token_m, tag_m = pairs[0]
        assert token_m.paired_with == "tags"
        assert tag_m.paired_with == "tokens"

    def test_lengths_match_in_data(self) -> None:
        """Verify the reference data actually has aligned lengths."""
        for rec in self.store.records:
            tokens = rec.data["tokens"]
            tags = rec.data["tags"]
            assert len(tokens) == len(tags)


# -- classify_value ----------------------------------------------------------


class TestTaggedSequenceClassification:
    def test_classify_combined_dict_int_tags(self) -> None:
        value = {"tokens": ["Hello", "world"], "tags": [0, 0]}
        assert classify_value(value) == ValueType.TAGGED_SEQUENCE

    def test_classify_combined_dict_string_tags(self) -> None:
        value = {"tokens": ["John", "runs"], "tags": ["B-PER", "O"]}
        assert classify_value(value) == ValueType.TAGGED_SEQUENCE

    def test_classify_with_archetype_hint(self) -> None:
        value = {"tokens": ["Hello", "world"], "tags": [0, 0]}
        vtype = classify_value(value, archetype=FieldArchetype.TAGGED_SEQUENCE)
        assert vtype == ValueType.TAGGED_SEQUENCE

    def test_mismatched_lengths_not_tagged(self) -> None:
        value = {"tokens": ["Hello", "world"], "tags": [0]}
        assert classify_value(value) != ValueType.TAGGED_SEQUENCE

    def test_empty_lists_not_tagged(self) -> None:
        value = {"tokens": [], "tags": []}
        assert classify_value(value) != ValueType.TAGGED_SEQUENCE

    def test_extra_keys_not_tagged(self) -> None:
        """A dict with tokens/tags plus other keys falls through to normal dict handling."""
        value = {"tokens": ["a"], "tags": [0], "extra": "stuff"}
        # Should still detect as tagged sequence (tokens + tags present and aligned)
        assert classify_value(value) == ValueType.TAGGED_SEQUENCE


# -- Renderer output ---------------------------------------------------------


class TestTaggedSequenceRenderer:
    def test_integer_tags_produces_static(self) -> None:
        value = {"tokens": ["SOCCER", "-", "JAPAN"], "tags": [0, 0, 5]}
        widget = _render_tagged_sequence(value, "test", {}, None)
        assert isinstance(widget, Static)

    def test_integer_tags_two_line_format(self) -> None:
        value = {"tokens": ["SOCCER", "-", "JAPAN"], "tags": [0, 0, 5]}
        widget = _render_tagged_sequence(value, "test", {}, None)
        text = str(widget.content)
        # Should contain both token and tag lines
        assert "SOCCER" in text
        assert "JAPAN" in text
        # Non-zero tags get highlighted with bright_cyan markup
        assert "bright_cyan" in text

    def test_integer_tags_zero_dimmed(self) -> None:
        value = {"tokens": ["the", "cat"], "tags": [0, 0]}
        widget = _render_tagged_sequence(value, "test", {}, None)
        text = str(widget.content)
        assert "[dim]" in text

    def test_bio_tags_produces_colored_output(self) -> None:
        value = {
            "tokens": ["John", "loves", "NYC"],
            "tags": ["B-PER", "O", "B-LOC"],
        }
        widget = _render_tagged_sequence(value, "test", {}, None)
        text = str(widget.content)
        assert "John" in text
        assert "(PER)" in text
        assert "(LOC)" in text
        # PER -> cyan, LOC -> yellow
        assert "cyan" in text
        assert "yellow" in text

    def test_bio_tags_span_grouping(self) -> None:
        """Consecutive B-X + I-X tokens should be grouped together."""
        value = {
            "tokens": ["New", "York", "City", "is", "great"],
            "tags": ["B-LOC", "I-LOC", "I-LOC", "O", "O"],
        }
        widget = _render_tagged_sequence(value, "test", {}, None)
        text = str(widget.content)
        # "New York City" should appear as one grouped entity
        assert "New York City" in text
        assert "(LOC)" in text

    def test_empty_sequence(self) -> None:
        value = {"tokens": [], "tags": []}
        widget = _render_tagged_sequence(value, "test", {}, None)
        assert isinstance(widget, Static)
        assert "empty" in str(widget.content).lower()

    def test_mismatched_lengths_fallback(self) -> None:
        value = {"tokens": ["Hello"], "tags": [0, 1]}
        widget = _render_tagged_sequence(value, "test", {}, None)
        # Falls back to JSON rendering, still a widget
        assert isinstance(widget, Static)

    def test_invalid_value_fallback(self) -> None:
        widget = _render_tagged_sequence("not a dict", "test", {}, None)
        assert isinstance(widget, Static)

    def test_non_list_fallback(self) -> None:
        value = {"tokens": "hello", "tags": "world"}
        widget = _render_tagged_sequence(value, "test", {}, None)
        assert isinstance(widget, Static)


# -- Tag-list detection in schema --------------------------------------------


class TestIsTagList:
    def test_int_list_is_tags(self) -> None:
        from nlpeep.schema import _is_tag_list

        assert _is_tag_list([0, 1, 2, 0, 0, 3]) is True

    def test_bio_strings_are_tags(self) -> None:
        from nlpeep.schema import _is_tag_list

        assert _is_tag_list(["O", "B-PER", "I-PER", "O", "B-LOC"]) is True

    def test_regular_strings_are_not_tags(self) -> None:
        from nlpeep.schema import _is_tag_list

        assert _is_tag_list(["hello", "world", "foo", "bar"]) is False

    def test_mixed_types_are_not_tags(self) -> None:
        from nlpeep.schema import _is_tag_list

        assert _is_tag_list([0, "O", 1, "B-PER"]) is False
