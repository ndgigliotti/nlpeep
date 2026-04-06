"""Tests for aligned-pair detection and rendering."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.widgets import Static

from nlpeep.data import RecordStore
from nlpeep.renderers import (
    ValueType,
    _render_aligned_pair,
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

    def test_aligned_pair_detected(self) -> None:
        pairs = self.mapping.aligned_pairs()
        assert len(pairs) == 1
        token_m, tag_m = pairs[0]
        assert token_m.json_path == "tokens"
        assert tag_m.json_path == "tags"

    def test_archetype_is_aligned_pair(self) -> None:
        pairs = self.mapping.aligned_pairs()
        token_m, tag_m = pairs[0]
        assert token_m.archetype == FieldArchetype.ALIGNED_PAIR
        assert tag_m.archetype == FieldArchetype.ALIGNED_PAIR

    def test_paired_with_set_correctly(self) -> None:
        pairs = self.mapping.aligned_pairs()
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


class TestAlignedPairClassification:
    def test_classify_two_aligned_lists(self) -> None:
        value = {"tokens": ["Hello", "world"], "tags": [0, 0]}
        assert classify_value(value) == ValueType.ALIGNED_PAIR

    def test_classify_string_pair(self) -> None:
        value = {"words": ["John", "runs"], "labels": ["B-PER", "O"]}
        assert classify_value(value) == ValueType.ALIGNED_PAIR

    def test_classify_with_archetype_hint(self) -> None:
        value = {"tokens": ["Hello", "world"], "tags": [0, 0]}
        vtype = classify_value(value, archetype=FieldArchetype.ALIGNED_PAIR)
        assert vtype == ValueType.ALIGNED_PAIR

    def test_mismatched_lengths_not_aligned(self) -> None:
        value = {"tokens": ["Hello", "world"], "tags": [0]}
        assert classify_value(value) != ValueType.ALIGNED_PAIR

    def test_empty_lists_not_aligned(self) -> None:
        value = {"tokens": [], "tags": []}
        assert classify_value(value) != ValueType.ALIGNED_PAIR

    def test_three_lists_not_aligned(self) -> None:
        value = {"a": [1, 2], "b": [3, 4], "c": [5, 6]}
        assert classify_value(value) != ValueType.ALIGNED_PAIR

    def test_one_list_not_aligned(self) -> None:
        value = {"tokens": ["Hello", "world"], "extra": "stuff"}
        assert classify_value(value) != ValueType.ALIGNED_PAIR


# -- Renderer output ---------------------------------------------------------


class TestAlignedPairRenderer:
    def test_classify_int_pair(self) -> None:
        value = {"tokens": ["SOCCER", "-", "JAPAN"], "tags": [0, 0, 5]}
        assert classify_value(value) == ValueType.ALIGNED_PAIR

    def test_classify_string_label_pair(self) -> None:
        value = {"words": ["John", "loves", "NYC"], "labels": ["B-PER", "O", "B-LOC"]}
        assert classify_value(value) == ValueType.ALIGNED_PAIR

    def test_empty_sequence(self) -> None:
        value = {"tokens": [], "tags": []}
        widget = _render_aligned_pair(value, "test", {}, None)
        assert isinstance(widget, Static)
        assert "empty" in str(widget.content).lower()

    def test_mismatched_lengths_fallback(self) -> None:
        value = {"tokens": ["Hello"], "tags": [0, 1]}
        widget = _render_aligned_pair(value, "test", {}, None)
        # Falls back to JSON rendering
        assert isinstance(widget, Static)

    def test_invalid_value_fallback(self) -> None:
        widget = _render_aligned_pair("not a dict", "test", {}, None)
        assert isinstance(widget, Static)

    def test_non_list_values_fallback(self) -> None:
        value = {"tokens": "hello", "tags": "world"}
        widget = _render_aligned_pair(value, "test", {}, None)
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
