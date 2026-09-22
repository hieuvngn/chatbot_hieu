"""Chọn skill phù hợp cho một câu hỏi theo rule (keyword matching).

Trước đây skill được chọn bằng một LLM call riêng (~14s trên model free).
Giờ chọn bằng keyword khai báo trong frontmatter ``keywords:`` của mỗi
SKILL.md — 0 network call, kết quả deterministic.

Progressive disclosure: chỉ frontmatter (name + keywords) được dùng khi
chọn; nội dung đầy đủ của skill được nạp sau bởi RagCore.
"""

from __future__ import annotations

from typing import Protocol

from rag_core.skills import Skill

MAX_SKILLS_PER_ANSWER = 2


class SkillSelector(Protocol):
    def select(self, query: str, skills: list[Skill]) -> list[str]: ...


def _normalize(text: str) -> str:
    return text.lower()


class RuleSkillSelector:
    """Chọn skill theo keyword match: skill nào có keyword xuất hiện trong
    câu hỏi sẽ được chọn, giữ thứ tự catalog, tối đa ``max_skills``."""

    def __init__(self, max_skills: int = MAX_SKILLS_PER_ANSWER) -> None:
        self.max_skills = max_skills

    def select(self, query: str, skills: list[Skill]) -> list[str]:
        if not skills:
            return []
        low = _normalize(query)
        picked: list[str] = []
        for skill in skills:
            if any(kw in low for kw in skill.keywords):
                picked.append(skill.name)
                if len(picked) == self.max_skills:
                    break
        return picked
