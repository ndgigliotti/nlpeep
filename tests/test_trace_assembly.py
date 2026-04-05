"""Tests for trace assembly of multi-span pipeline data.

Verifies that RecordStore.maybe_assemble_traces() correctly detects
trace data (structurally heterogeneous records sharing a group ID)
and assembles spans into composite records, while leaving non-trace
data unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nlpeep.data import Record, RecordStore

_REF = Path(__file__).parent / "reference_data"


# -- LangSmith trace file: should assemble into 1 composite record -----------


class TestLangSmithTraceAssembly:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.store = RecordStore.load(_REF / "langsmith_rag_trace.jsonl")
        self.assembled = self.store.maybe_assemble_traces()

    def test_assembles_into_one_record(self) -> None:
        assert len(self.assembled) == 1

    def test_composite_has_query(self) -> None:
        data = self.assembled[0].data
        # Root span has inputs.question
        question = data.get("inputs", {}).get("question")
        assert question is not None
        assert "Basel III" in question

    def test_composite_has_response(self) -> None:
        data = self.assembled[0].data
        answer = data.get("outputs", {}).get("answer")
        assert answer is not None
        assert len(answer) > 20

    def test_composite_has_documents(self) -> None:
        data = self.assembled[0].data
        # Documents come from the retriever child span via outputs.documents
        docs = data.get("outputs", {}).get("documents")
        assert docs is not None
        assert isinstance(docs, list)
        assert len(docs) == 4

    def test_composite_has_messages(self) -> None:
        data = self.assembled[0].data
        # Messages come from the ChatOpenAI child span via inputs.messages
        messages = data.get("inputs", {}).get("messages")
        assert messages is not None
        assert isinstance(messages, list)
        assert len(messages) == 2

    def test_composite_has_trace_spans(self) -> None:
        data = self.assembled[0].data
        spans = data.get("_trace_spans")
        assert spans is not None
        assert isinstance(spans, list)
        assert len(spans) == 4

    def test_trace_spans_have_span_names(self) -> None:
        spans = self.assembled[0].data["_trace_spans"]
        names = [s.get("_span_name") for s in spans]
        assert "rag_pipeline" in names
        assert "embedding_query" in names
        assert "vector_store_retriever" in names
        assert "ChatOpenAI" in names

    def test_trace_spans_ordered_by_start_time(self) -> None:
        spans = self.assembled[0].data["_trace_spans"]
        times = [s.get("start_time", "") for s in spans]
        assert times == sorted(times)

    def test_trace_spans_contain_original_data(self) -> None:
        spans = self.assembled[0].data["_trace_spans"]
        # The retriever span should have its original documents
        retriever_span = next(s for s in spans if s.get("name") == "vector_store_retriever")
        docs = retriever_span.get("outputs", {}).get("documents")
        assert docs is not None
        assert len(docs) == 4


# -- RAGAS evaluation: should NOT fire trace assembly -------------------------


class TestRagasNoAssembly:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.store = RecordStore.load(_REF / "ragas_evaluation_output.jsonl")
        self.assembled = self.store.maybe_assemble_traces()

    def test_no_assembly(self) -> None:
        # RAGAS data has homogeneous structure, so assembly should not fire
        assert len(self.assembled) == len(self.store)

    def test_returns_same_records(self) -> None:
        # Records should be unchanged
        for i in range(len(self.store)):
            assert self.assembled[i].data == self.store[i].data


# -- Simple RAG output: should NOT fire (no group ID field) -------------------


class TestSimpleRagNoAssembly:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.store = RecordStore.load(_REF / "simple_rag_output.jsonl")
        self.assembled = self.store.maybe_assemble_traces()

    def test_no_assembly(self) -> None:
        assert len(self.assembled) == len(self.store)

    def test_records_unchanged(self) -> None:
        for i in range(len(self.store)):
            assert self.assembled[i].data == self.store[i].data


# -- Synthetic test: records with shared group_id, different structures -------


class TestSyntheticTraceAssembly:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        records = [
            Record(
                index=0,
                data={
                    "trace_id": "t1",
                    "name": "root",
                    "parent_id": None,
                    "inputs": {"query": "What is X?"},
                    "outputs": {"answer": "X is Y."},
                    "start_time": "2026-01-01T00:00:01",
                },
            ),
            Record(
                index=1,
                data={
                    "trace_id": "t1",
                    "name": "retriever",
                    "parent_id": "root-id",
                    "outputs": {
                        "documents": [
                            {"text": "Doc about X", "score": 0.9},
                            {"text": "More about X", "score": 0.8},
                        ]
                    },
                    "start_time": "2026-01-01T00:00:02",
                },
            ),
            Record(
                index=2,
                data={
                    "trace_id": "t1",
                    "name": "llm",
                    "parent_id": "root-id",
                    "inputs": {
                        "messages": [
                            {"role": "system", "content": "Be helpful."},
                            {"role": "user", "content": "What is X?"},
                        ]
                    },
                    "outputs": {"text": "X is Y."},
                    "start_time": "2026-01-01T00:00:03",
                },
            ),
            # Second trace group
            Record(
                index=3,
                data={
                    "trace_id": "t2",
                    "name": "root2",
                    "parent_id": None,
                    "inputs": {"query": "What is Z?"},
                    "outputs": {"answer": "Z is W."},
                    "start_time": "2026-01-01T00:01:01",
                },
            ),
            Record(
                index=4,
                data={
                    "trace_id": "t2",
                    "name": "retriever2",
                    "parent_id": "root2-id",
                    "outputs": {
                        "documents": [
                            {"text": "Doc about Z", "score": 0.95},
                        ]
                    },
                    "start_time": "2026-01-01T00:01:02",
                },
            ),
        ]
        self.store = RecordStore(records=records)
        self.assembled = self.store.maybe_assemble_traces()

    def test_assembled_count(self) -> None:
        # Two trace groups should produce two composite records
        assert len(self.assembled) == 2

    def test_first_group_query(self) -> None:
        data = self.assembled[0].data
        assert data["inputs"]["query"] == "What is X?"

    def test_first_group_documents_promoted(self) -> None:
        data = self.assembled[0].data
        docs = data["outputs"].get("documents")
        assert docs is not None
        assert len(docs) == 2

    def test_first_group_messages_promoted(self) -> None:
        data = self.assembled[0].data
        messages = data["inputs"].get("messages")
        assert messages is not None
        assert len(messages) == 2

    def test_first_group_has_trace_spans(self) -> None:
        spans = self.assembled[0].data["_trace_spans"]
        assert len(spans) == 3
        names = [s["_span_name"] for s in spans]
        assert "root" in names
        assert "retriever" in names
        assert "llm" in names

    def test_second_group_documents_promoted(self) -> None:
        data = self.assembled[1].data
        docs = data["outputs"].get("documents")
        assert docs is not None
        assert len(docs) == 1

    def test_second_group_has_trace_spans(self) -> None:
        spans = self.assembled[1].data["_trace_spans"]
        assert len(spans) == 2


# -- Edge case: single record store should not assemble -----------------------


class TestSingleRecordNoAssembly:
    def test_single_record(self) -> None:
        store = RecordStore(
            records=[
                Record(index=0, data={"trace_id": "t1", "value": "hello"}),
            ]
        )
        assembled = store.maybe_assemble_traces()
        assert len(assembled) == 1
        assert assembled[0].data == store[0].data


# -- Edge case: homogeneous records with shared ID should not assemble --------


class TestHomogeneousSharedIdNoAssembly:
    def test_same_structure(self) -> None:
        """Records share a group ID but all have the same key set."""
        records = [
            Record(
                index=0,
                data={
                    "group": "g1",
                    "name": "Alice",
                    "score": 0.9,
                },
            ),
            Record(
                index=1,
                data={
                    "group": "g1",
                    "name": "Bob",
                    "score": 0.8,
                },
            ),
            Record(
                index=2,
                data={
                    "group": "g2",
                    "name": "Charlie",
                    "score": 0.7,
                },
            ),
        ]
        store = RecordStore(records=records)
        assembled = store.maybe_assemble_traces()
        # Same structure means no assembly
        assert len(assembled) == len(records)
