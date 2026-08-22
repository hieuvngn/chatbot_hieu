from __future__ import annotations

from pathlib import Path

import pytest

from rag_core.skills import Skill, load_skills, parse_skill_md

VALID_MD = """---
name: eli5
description: Giải thích đơn giản
---
Dùng ví dụ đời thường.
"""


def test_parse_valid_skill_md() -> None:
    skill = parse_skill_md(VALID_MD)
    assert skill == Skill(
        name="eli5", description="Giải thích đơn giản", instructions="Dùng ví dụ đời thường."
    )


@pytest.mark.parametrize(
    "text",
    [
        "",
        "chỉ markdown thường",
        "---\nfrontmatter không đóng\n",
        "---\ndescription: thiếu name\n---\nbody",
        "---\nname: chỉ name\n---\n",
    ],
)
def test_parse_invalid_returns_none(text: str) -> None:
    assert parse_skill_md(text) is None


def test_parse_rejects_comma_in_name() -> None:
    text = "---\nname: a,b\ndescription: d\n---\nbody"
    assert parse_skill_md(text) is None


def test_load_skills_handles_utf8_bom(tmp_path: Path) -> None:
    directory = tmp_path / "bommed"
    directory.mkdir()
    (directory / "SKILL.md").write_bytes(
        b"\xef\xbb\xbf---\nname: bom\ndescription: d\n---\nbody\n"
    )

    skills = load_skills(tmp_path)

    assert [s.name for s in skills] == ["bom"]


def write_skill(root: Path, folder: str, text: str) -> None:
    directory = root / folder
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(text, encoding="utf-8")


def test_load_skills_sorted_and_skips_invalid(tmp_path: Path) -> None:
    write_skill(tmp_path, "zebra", VALID_MD.replace("eli5", "zebra"))
    write_skill(tmp_path, "alpha", VALID_MD.replace("eli5", "alpha"))
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "SKILL.md").write_text("không có frontmatter", encoding="utf-8")
    (tmp_path / "loose.md").write_text("không nằm trong thư mục con", encoding="utf-8")

    skills = load_skills(tmp_path)

    assert [s.name for s in skills] == ["alpha", "zebra"]


def test_load_skills_missing_directory_returns_empty(tmp_path: Path) -> None:
    assert load_skills(tmp_path / "nope") == []
