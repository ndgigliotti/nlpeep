"""Tests for file format loaders (CSV, TSV, JSON, JSONL)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from nlpeep.data import RecordStore

_REF = Path(__file__).parent / "reference_data"


# -- JSONL -------------------------------------------------------------------

class TestJsonlLoader:
    def test_loads_records(self) -> None:
        store = RecordStore.load(_REF / "simple_rag_output.jsonl")
        assert len(store) == 3
        assert store.skipped == 0

    def test_records_have_data(self) -> None:
        store = RecordStore.load(_REF / "simple_rag_output.jsonl")
        assert "query" in store[0].data

    def test_skips_invalid_lines(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"a": 1}\n')
            f.write("not json\n")
            f.write('{"b": 2}\n')
            path = Path(f.name)
        try:
            store = RecordStore.load(path)
            assert len(store) == 2
            assert store.skipped == 1
        finally:
            path.unlink()


# -- JSON --------------------------------------------------------------------

class TestJsonLoader:
    def test_loads_array_of_objects(self) -> None:
        data = [{"q": "What?", "a": "That."}, {"q": "Who?", "a": "Me."}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = Path(f.name)
        try:
            store = RecordStore.load(path)
            assert len(store) == 2
            assert store[0].data["q"] == "What?"
            assert store[1].data["a"] == "Me."
        finally:
            path.unlink()

    def test_loads_single_object(self) -> None:
        data = {"key": "value", "nested": {"a": 1}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = Path(f.name)
        try:
            store = RecordStore.load(path)
            assert len(store) == 1
            assert store[0].data["key"] == "value"
        finally:
            path.unlink()

    def test_loads_reference_json(self) -> None:
        store = RecordStore.load(_REF / "ragas_comparison.json")
        assert len(store) >= 1

    def test_jsonl_falls_back_to_json_array(self) -> None:
        """A .jsonl file whose first line is a JSON array should load as JSON."""
        data = [{"x": 1}, {"x": 2}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            json.dump(data, f)
            path = Path(f.name)
        try:
            store = RecordStore.load(path)
            assert len(store) == 2
        finally:
            path.unlink()


# -- CSV / TSV ---------------------------------------------------------------

class TestCsvLoader:
    def test_loads_csv(self) -> None:
        content = "text,label,score\nHello,pos,0.95\nBad,neg,0.12\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(content)
            path = Path(f.name)
        try:
            store = RecordStore.load(path)
            assert len(store) == 2
            assert store[0].data["text"] == "Hello"
            assert store[0].data["label"] == "pos"
        finally:
            path.unlink()

    def test_coerces_numeric_types(self) -> None:
        content = "name,count,score\nfoo,10,0.85\nbar,20,0.42\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(content)
            path = Path(f.name)
        try:
            store = RecordStore.load(path)
            assert store[0].data["count"] == 10
            assert isinstance(store[0].data["count"], int)
            assert store[0].data["score"] == 0.85
            assert isinstance(store[0].data["score"], float)
        finally:
            path.unlink()

    def test_embedded_json_stays_as_string(self) -> None:
        """Polars leaves embedded JSON as strings; the renderer handles them."""
        content = 'name,tags\nfoo,"[""a"",""b""]"\nbar,"[""c""]"\n'
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(content)
            path = Path(f.name)
        try:
            store = RecordStore.load(path)
            assert isinstance(store[0].data["tags"], str)
        finally:
            path.unlink()

    def test_loads_tsv(self) -> None:
        content = "query\tresponse\nWhat?\tThat.\nWho?\tMe.\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write(content)
            path = Path(f.name)
        try:
            store = RecordStore.load(path)
            assert len(store) == 2
            assert store[0].data["query"] == "What?"
            assert store[1].data["response"] == "Me."
        finally:
            path.unlink()


# -- CSV detection parity with JSONL -----------------------------------------

class TestCsvDetectionParity:
    """Flat datasets should produce identical schema detection from CSV and JSONL."""

    def test_imdb_sentiment(self) -> None:
        from nlpeep.schema import SchemaMapping
        jsonl = SchemaMapping.auto_detect(RecordStore.load(_REF / "imdb_sentiment.jsonl").sample())
        csv = SchemaMapping.auto_detect(RecordStore.load(_REF / "imdb_sentiment.csv").sample())
        jsonl_roles = {m.role.value: m.json_path for m in jsonl.mappings if m.role.value != "unmapped"}
        csv_roles = {m.role.value: m.json_path for m in csv.mappings if m.role.value != "unmapped"}
        assert csv_roles == jsonl_roles

    def test_cnn_dailymail(self) -> None:
        from nlpeep.schema import SchemaMapping
        jsonl = SchemaMapping.auto_detect(RecordStore.load(_REF / "cnn_dailymail_summarization.jsonl").sample())
        csv = SchemaMapping.auto_detect(RecordStore.load(_REF / "cnn_dailymail_summarization.csv").sample())
        jsonl_roles = {m.role.value: m.json_path for m in jsonl.mappings if m.role.value != "unmapped"}
        csv_roles = {m.role.value: m.json_path for m in csv.mappings if m.role.value != "unmapped"}
        assert csv_roles == jsonl_roles
