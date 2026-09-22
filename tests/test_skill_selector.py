from __future__ import annotations

from rag_core.skill_selector import MAX_SKILLS_PER_ANSWER, RuleSkillSelector
from rag_core.skills import Skill

SKILLS = [
    Skill(
        name="code-examples",
        description="d",
        instructions="i",
        keywords=("thuật toán", "cấu trúc dữ liệu", "algorithm", "hash"),
    ),
    Skill(
        name="eli5",
        description="d",
        instructions="i",
        keywords=("giải thích", "là gì", "explain"),
    ),
]


def test_keyword_match_selects_skill() -> None:
    selector = RuleSkillSelector()
    assert selector.select("giải thích bảng băm là gì?", SKILLS) == ["eli5"]
    assert selector.select("thuật toán sắp xếp hoạt động ra sao", SKILLS) == ["code-examples"]


def test_no_keyword_match_selects_nothing() -> None:
    selector = RuleSkillSelector()
    assert selector.select("hôm nay thời tiết thế nào", SKILLS) == []


def test_case_insensitive_match() -> None:
    selector = RuleSkillSelector()
    # "HASH algorithm" viết hoa vẫn match keyword thường
    assert "code-examples" in selector.select("HASH table algorithm", SKILLS)
    # keyword tiếng Việt có dấu match câu thường
    assert "eli5" in selector.select("GIẢI THÍCH bảng băm", SKILLS)
    assert MAX_SKILLS_PER_ANSWER == 2


def test_catalog_order_preserved() -> None:
    selector = RuleSkillSelector()
    # cả 2 match: giữ thứ tự catalog (code-examples đứng trước eli5)
    assert selector.select("explain hash algorithm", SKILLS) == ["code-examples", "eli5"]


def test_empty_catalog_returns_empty() -> None:
    selector = RuleSkillSelector()
    assert selector.select("giải thích bảng băm", []) == []


def test_skill_without_keywords_never_selected() -> None:
    no_kw = [Skill(name="plain", description="d", instructions="i")]
    selector = RuleSkillSelector()
    assert selector.select("bất kỳ câu gì", no_kw) == []


