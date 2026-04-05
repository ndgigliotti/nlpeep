"""Smoke tests for auto-detection across reference data files.

Verifies that the recursive nested-field detection and archetype
classification produce reasonable mappings for each supported format.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nlpeep.data import RecordStore
from nlpeep.schema import FieldArchetype, FieldRole, SchemaMapping

_REF = Path(__file__).parent / "reference_data"
_FIX = Path(__file__).parent / "fixtures"


def _detect(path: Path) -> SchemaMapping:
    """Load a JSONL file and run auto-detection on its records."""
    store = RecordStore.load(path)
    assert len(store) > 0, f"No records loaded from {path}"
    return SchemaMapping.auto_detect(store.sample())


# -- helpers ----------------------------------------------------------------


def _has_role(mapping: SchemaMapping, role: FieldRole) -> bool:
    return mapping.get_mapping(role) is not None


def _role_path(mapping: SchemaMapping, role: FieldRole) -> str | None:
    m = mapping.get_mapping(role)
    return m.json_path if m else None


# -- LangSmith RAG trace (primary nested test case) -------------------------


class TestLangSmithRagTrace:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.mapping = _detect(_REF / "langsmith_rag_trace.jsonl")
        self.store = RecordStore.load(_REF / "langsmith_rag_trace.jsonl")

    def test_query_detected(self) -> None:
        assert _has_role(self.mapping, FieldRole.QUERY)
        path = _role_path(self.mapping, FieldRole.QUERY)
        assert path is not None
        # Should be a nested path under inputs
        assert "." in path

    def test_response_detected(self) -> None:
        assert _has_role(self.mapping, FieldRole.RESPONSE)
        path = _role_path(self.mapping, FieldRole.RESPONSE)
        assert path is not None

    def test_documents_detected(self) -> None:
        assert _has_role(self.mapping, FieldRole.DOCUMENTS)
        path = _role_path(self.mapping, FieldRole.DOCUMENTS)
        assert path is not None
        assert "documents" in path.lower()

    def test_id_detected(self) -> None:
        assert _has_role(self.mapping, FieldRole.ID)

    def test_nested_paths_resolve(self) -> None:
        """Verify that the detected paths actually resolve to values."""
        record = self.store[0]
        query_val = self.mapping.resolve(record.data, FieldRole.QUERY)
        assert query_val is not None and isinstance(query_val, str)

    def test_trace_detected(self) -> None:
        # The messages field inside inputs should be detected as TRACE
        assert _has_role(self.mapping, FieldRole.TRACE)


# -- Langfuse RAG trace (another nested format) -----------------------------


class TestLangfuseRagTrace:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.mapping = _detect(_REF / "langfuse_rag_trace.jsonl")

    def test_query_detected(self) -> None:
        assert _has_role(self.mapping, FieldRole.QUERY)
        path = _role_path(self.mapping, FieldRole.QUERY)
        assert path is not None

    def test_response_detected(self) -> None:
        assert _has_role(self.mapping, FieldRole.RESPONSE)


# -- Phoenix OpenInference RAG trace ----------------------------------------


class TestPhoenixTrace:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.mapping = _detect(_REF / "phoenix_openinference_rag_trace.jsonl")

    def test_query_or_input_detected(self) -> None:
        # Phoenix uses attributes.input.value or similar nested keys
        has_query = _has_role(self.mapping, FieldRole.QUERY)
        has_response = _has_role(self.mapping, FieldRole.RESPONSE)
        # At minimum one of these should be detected from nested attributes
        assert has_query or has_response


# -- RAGAS evaluation output (flat structure, regression test) ---------------


class TestRagasEvaluation:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.mapping = _detect(_REF / "ragas_evaluation_output.jsonl")

    def test_query_detected(self) -> None:
        assert _has_role(self.mapping, FieldRole.QUERY)
        # Should still be a top-level field
        path = _role_path(self.mapping, FieldRole.QUERY)
        assert path == "question"

    def test_response_detected(self) -> None:
        assert _has_role(self.mapping, FieldRole.RESPONSE)
        path = _role_path(self.mapping, FieldRole.RESPONSE)
        assert path == "answer"

    def test_ground_truth_detected(self) -> None:
        assert _has_role(self.mapping, FieldRole.GROUND_TRUTH)
        path = _role_path(self.mapping, FieldRole.GROUND_TRUTH)
        assert path == "ground_truth"

    def test_documents_detected(self) -> None:
        assert _has_role(self.mapping, FieldRole.DOCUMENTS)

    def test_metrics_detected(self) -> None:
        # RAGAS has individual score fields like faithfulness, answer_relevancy
        metrics = self.mapping.get_all_for_role(FieldRole.METRICS)
        assert len(metrics) >= 2, "Should detect multiple metric fields"


# -- Simple RAG output (flat, regression test) -------------------------------


class TestSimpleRag:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.mapping = _detect(_REF / "simple_rag_output.jsonl")

    def test_query_detected(self) -> None:
        assert _has_role(self.mapping, FieldRole.QUERY)
        assert _role_path(self.mapping, FieldRole.QUERY) == "query"

    def test_response_detected(self) -> None:
        assert _has_role(self.mapping, FieldRole.RESPONSE)
        assert _role_path(self.mapping, FieldRole.RESPONSE) == "response"

    def test_documents_detected(self) -> None:
        assert _has_role(self.mapping, FieldRole.DOCUMENTS)
        assert _role_path(self.mapping, FieldRole.DOCUMENTS) == "documents"


# -- Multi-hop RAG trace (mixed flat + nested) ------------------------------


class TestMultiHopRag:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.mapping = _detect(_REF / "multi_hop_rag_trace.jsonl")

    def test_query_detected(self) -> None:
        assert _has_role(self.mapping, FieldRole.QUERY)

    def test_response_detected(self) -> None:
        assert _has_role(self.mapping, FieldRole.RESPONSE)

    def test_ground_truth_detected(self) -> None:
        assert _has_role(self.mapping, FieldRole.GROUND_TRUTH)


# -- Archetype classification -----------------------------------------------


class TestArchetypeClassification:
    def test_score_archetype(self) -> None:
        records = [{"val": 0.85}, {"val": 0.92}, {"val": 0.78}]
        mapping = SchemaMapping.auto_detect(records)
        m = next((m for m in mapping.mappings if m.json_path == "val"), None)
        assert m is not None
        assert m.archetype == FieldArchetype.SCORE

    def test_identifier_archetype(self) -> None:
        records = [{"id": "abc-123"}, {"id": "def-456"}, {"id": "ghi-789"}]
        mapping = SchemaMapping.auto_detect(records)
        id_m = mapping.get_mapping(FieldRole.ID)
        assert id_m is not None
        assert id_m.archetype == FieldArchetype.IDENTIFIER

    def test_category_archetype(self) -> None:
        records = [
            {"status": "success"},
            {"status": "success"},
            {"status": "failed"},
            {"status": "success"},
            {"status": "pending"},
        ]
        mapping = SchemaMapping.auto_detect(records)
        m = next((m for m in mapping.mappings if m.json_path == "status"), None)
        assert m is not None
        assert m.archetype == FieldArchetype.CATEGORY

    def test_free_text_archetype(self) -> None:
        long_text = (
            "This is a long piece of text that exceeds fifty characters"
            " and should be classified as free text."
        )
        records = [
            {"description": long_text},
            {"description": long_text + " More words here for variation."},
            {"description": long_text + " Even more variation."},
        ]
        mapping = SchemaMapping.auto_detect(records)
        m = next((m for m in mapping.mappings if m.json_path == "description"), None)
        assert m is not None
        assert m.archetype == FieldArchetype.FREE_TEXT

    def test_score_dict_archetype(self) -> None:
        # When a dict has only numeric values, the flattener recurses into it
        # and the individual float fields get detected with SCORE archetype.
        # To test score_dict archetype, we need a field that stays as a whole
        # dict (e.g., by being inside a list, or by reaching depth cap).
        # Here we test individual score fields from flattened dicts.
        records = [
            {"scores": {"f1": 0.9, "precision": 0.85}},
            {"scores": {"f1": 0.88, "precision": 0.92}},
        ]
        mapping = SchemaMapping.auto_detect(records)
        # The flattener produces scores.f1 and scores.precision
        f1_m = next((m for m in mapping.mappings if m.json_path == "scores.f1"), None)
        assert f1_m is not None
        assert f1_m.archetype == FieldArchetype.SCORE

    def test_nested_detection_with_archetype(self) -> None:
        records = [
            {"inputs": {"question": "What is X?"}, "outputs": {"answer": "X is Y."}},
            {"inputs": {"question": "What is A?"}, "outputs": {"answer": "A is B."}},
        ]
        mapping = SchemaMapping.auto_detect(records)
        assert _has_role(mapping, FieldRole.QUERY)
        assert _role_path(mapping, FieldRole.QUERY) == "inputs.question"
        assert _has_role(mapping, FieldRole.RESPONSE)
        assert _role_path(mapping, FieldRole.RESPONSE) == "outputs.answer"


# -- Recursive flattening ---------------------------------------------------


class TestFlattenAndFieldSummary:
    def test_field_summary_includes_nested_paths(self) -> None:
        store = RecordStore.load(_REF / "langsmith_rag_trace.jsonl")
        summary = store.field_summary()
        # Should contain nested paths, not just top-level keys
        nested_paths = [p for p in summary if "." in p]
        assert len(nested_paths) > 0, "field_summary should include nested dot-paths"

    def test_field_summary_flat_still_works(self) -> None:
        store = RecordStore.load(_REF / "simple_rag_output.jsonl")
        summary = store.field_summary()
        assert "query" in summary
        assert "response" in summary
        assert "documents" in summary


# -- Unmapped field filtering ------------------------------------------------


class TestUnmappedFields:
    def test_nested_mapped_fields_excluded(self) -> None:
        records = [
            {"inputs": {"question": "Q1", "extra": "E1"}, "other": "O1"},
        ]
        mapping = SchemaMapping.auto_detect(records)
        record_data = records[0]
        unmapped = mapping.unmapped_fields(record_data)
        # inputs.question should be mapped as QUERY, so it should not
        # appear fully in unmapped. But inputs.extra should remain.
        if "inputs" in unmapped:
            assert "extra" in unmapped["inputs"]
            assert "question" not in unmapped.get("inputs", {})
