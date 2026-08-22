from __future__ import annotations

from rag_core.skill_selector import MAX_SKILLS_PER_ANSWER, OpenRouterSkillSelector, parse_selection


def test_parse_selection_filters_unknown_and_dedupes() -> None:
    content = '{"skills": ["b", "zzz", "a", "b"]}'
    assert parse_selection(content, {"a", "b"}) == ["b", "a"]


def test_parse_selection_caps_at_max() -> None:
    content = '{"skills": ["a", "b", "c"]}'
    assert parse_selection(content, {"a", "b", "c"}) == ["a", "b"]
    assert MAX_SKILLS_PER_ANSWER == 2


def test_parse_selection_garbage_returns_empty() -> None:
    assert parse_selection("not json at all", {"a"}) == []


def test_parse_selection_code_fenced_json() -> None:
    content = '```json\n{"skills": ["a"]}\n```'
    assert parse_selection(content, {"a"}) == ["a"]


def test_select_empty_catalog_returns_empty_without_api_call() -> None:
    selector = OpenRouterSkillSelector(api_key="k")
    assert selector.select("q", []) == []
