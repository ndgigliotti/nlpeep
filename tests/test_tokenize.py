"""Tests for field-name tokenizer and keyword matching."""
from __future__ import annotations

import pytest

from nlpeep.schema import (
    _depluralize,
    _tokenize_name,
    _ROLE_KEYWORDS,
    _EXACT_KEYWORDS,
    FieldRole,
)


# -- _depluralize -------------------------------------------------------------

class TestDepluralize:
    @pytest.mark.parametrize("token, expected", [
        # regular -s
        ("metrics", "metric"),
        ("scores", "score"),
        ("documents", "document"),
        ("passages", "passage"),
        ("steps", "step"),
        # -ies -> -y
        ("entities", "entity"),
        ("categories", "category"),
        # -sses/-xes/-zes -> strip -es
        ("classes", "class"),
        ("indexes", "index"),
        # -se + s should strip -s, not -es
        ("responses", "response"),
        ("phrases", "phrase"),
        ("databases", "database"),
        # should not strip
        ("class", "class"),
        ("miss", "miss"),       # ends in "ss" -- leave alone
        ("status", "status"),   # Latin singular ending in -us
        ("extras", "extra"),    # regular -as plural, not Latin
        ("id", "id"),           # too short
        ("is", "is"),           # too short
    ])
    def test_depluralize(self, token: str, expected: str) -> None:
        assert _depluralize(token) == expected


# -- _tokenize_name ----------------------------------------------------------

class TestTokenizeName:
    """Verify splitting and depluralization across naming conventions."""

    @pytest.mark.parametrize("name, expected", [
        ("snake_case", ["snake", "case"]),
        ("multi__underscore", ["multi", "underscore"]),
        ("leading_", ["leading"]),
        ("_leading", ["leading"]),
    ])
    def test_snake_case(self, name: str, expected: list[str]) -> None:
        assert _tokenize_name(name) == expected

    @pytest.mark.parametrize("name, expected", [
        ("camelCase", ["camel", "case"]),
        ("userQuery", ["user", "query"]),
        ("getHTTPResponse", ["get", "http", "response"]),
    ])
    def test_camel_case(self, name: str, expected: list[str]) -> None:
        assert _tokenize_name(name) == expected

    @pytest.mark.parametrize("name, expected", [
        ("PascalCase", ["pascal", "case"]),
        ("SourceText", ["source", "text"]),
        ("UserQuestion", ["user", "question"]),
    ])
    def test_pascal_case(self, name: str, expected: list[str]) -> None:
        assert _tokenize_name(name) == expected

    @pytest.mark.parametrize("name, expected", [
        ("HTMLParser", ["html", "parser"]),
        ("NERTags", ["ner", "tag"]),
        ("URLSource", ["url", "source"]),
        ("ID", ["id"]),
    ])
    def test_acronym_runs(self, name: str, expected: list[str]) -> None:
        assert _tokenize_name(name) == expected

    @pytest.mark.parametrize("name, expected", [
        ("kebab-case", ["kebab", "case"]),
        ("dot.separated", ["dot", "separated"]),
        ("has spaces", ["ha", "space"]),
        ("Ground Truth", ["ground", "truth"]),
        ("NER Tags", ["ner", "tag"]),
    ])
    def test_other_separators(self, name: str, expected: list[str]) -> None:
        assert _tokenize_name(name) == expected

    @pytest.mark.parametrize("name, expected", [
        ("mixed_camelCase", ["mixed", "camel", "case"]),
        ("my-input_text", ["my", "input", "text"]),
        ("a.b_c-d E", ["a", "b", "c", "d", "e"]),
    ])
    def test_mixed_conventions(self, name: str, expected: list[str]) -> None:
        assert _tokenize_name(name) == expected

    def test_single_word(self) -> None:
        assert _tokenize_name("query") == ["query"]

    def test_empty_string(self) -> None:
        assert _tokenize_name("") == []

    def test_plurals_normalized(self) -> None:
        """Plural field names should reduce to singular tokens."""
        assert _tokenize_name("retrieved_documents") == ["retrieved", "document"]
        assert _tokenize_name("ner_tags") == ["ner", "tag"]
        assert _tokenize_name("entities") == ["entity"]
        assert _tokenize_name("eval_scores") == ["eval", "score"]


# -- keyword matching ---------------------------------------------------------

def _match_role(name: str) -> FieldRole | None:
    """Simulate the keyword matching logic from auto_detect."""
    tokens = set(_tokenize_name(name))
    exact_role = _EXACT_KEYWORDS.get(name.lower())
    for role, keywords in _ROLE_KEYWORDS:
        if tokens & keywords or exact_role == role:
            return role
    return None


class TestKeywordMatching:
    """Verify that compound field names resolve to the correct role."""

    @pytest.mark.parametrize("name, expected", [
        ("query", FieldRole.QUERY),
        ("user_question", FieldRole.QUERY),
        ("userQuery", FieldRole.QUERY),
        ("search_query", FieldRole.QUERY),
        ("q", FieldRole.QUERY),
    ])
    def test_query(self, name: str, expected: FieldRole) -> None:
        assert _match_role(name) == expected

    @pytest.mark.parametrize("name, expected", [
        ("text", FieldRole.INPUT),
        ("input", FieldRole.INPUT),
        ("source_text", FieldRole.INPUT),
        ("sourceArticle", FieldRole.INPUT),
        ("input_tokens", FieldRole.INPUT),
    ])
    def test_input(self, name: str, expected: FieldRole) -> None:
        assert _match_role(name) == expected

    @pytest.mark.parametrize("name, expected", [
        ("response", FieldRole.RESPONSE),
        ("model_output", FieldRole.RESPONSE),
        ("modelOutput", FieldRole.RESPONSE),
        ("llm_response", FieldRole.RESPONSE),
        ("ref_summary", FieldRole.RESPONSE),
        ("predicted_label", FieldRole.RESPONSE),
    ])
    def test_response(self, name: str, expected: FieldRole) -> None:
        assert _match_role(name) == expected

    @pytest.mark.parametrize("name, expected", [
        ("label", FieldRole.GROUND_TRUTH),
        ("ground_truth", FieldRole.GROUND_TRUTH),
        ("bio_tags", FieldRole.GROUND_TRUTH),
        ("target_text", FieldRole.GROUND_TRUTH),
        ("ner_tags", FieldRole.GROUND_TRUTH),
        ("highlights", FieldRole.GROUND_TRUTH),
    ])
    def test_ground_truth(self, name: str, expected: FieldRole) -> None:
        assert _match_role(name) == expected

    @pytest.mark.parametrize("name, expected", [
        ("documents", FieldRole.DOCUMENTS),
        ("retrievedDocuments", FieldRole.DOCUMENTS),
        ("context", FieldRole.DOCUMENTS),
        ("contextPassage", FieldRole.DOCUMENTS),
    ])
    def test_documents(self, name: str, expected: FieldRole) -> None:
        assert _match_role(name) == expected

    @pytest.mark.parametrize("name, expected", [
        ("evalResults", FieldRole.METRICS),
        ("eval", FieldRole.METRICS),
        ("scores", FieldRole.METRICS),
        # singular and plural both work via depluralization
        ("metric", FieldRole.METRICS),
        ("metrics", FieldRole.METRICS),
    ])
    def test_metrics(self, name: str, expected: FieldRole) -> None:
        assert _match_role(name) == expected

    @pytest.mark.parametrize("name", [
        "created_at", "version", "model_name", "timestamp", "url",
    ])
    def test_no_match(self, name: str) -> None:
        assert _match_role(name) is None

    def test_exact_only_no_token_leak(self) -> None:
        """'text' should match INPUT only as the full field name,
        not as a token inside a compound name like 'target_text'."""
        assert _match_role("text") == FieldRole.INPUT
        assert _match_role("target_text") == FieldRole.GROUND_TRUTH

    def test_plural_field_names_match(self) -> None:
        """Plural field names should match singular keywords."""
        assert _match_role("labels") == FieldRole.GROUND_TRUTH
        assert _match_role("passages") == FieldRole.DOCUMENTS
        assert _match_role("steps") == FieldRole.TRACE
        assert _match_role("settings") == FieldRole.METADATA
