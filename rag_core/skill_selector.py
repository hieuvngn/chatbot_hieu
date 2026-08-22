"""Chọn skill phù hợp cho một câu hỏi dựa trên catalog name/description.

Progressive disclosure: chỉ frontmatter mô tả đi vào prompt selector;
nội dung đầy đủ của skill được chọn sẽ do RagCore nạp sau.
"""

from __future__ import annotations

import json
import re
from typing import Protocol

from rag_core.config import DEFAULT_BASE_URL, DEFAULT_LLM_MODEL
from rag_core.skills import Skill

MAX_SKILLS_PER_ANSWER = 2

_SYSTEM_PROMPT = (
    "You are a skill router for a course chatbot. Given the student's "
    "question and a catalog of available skills (name + description), pick "
    'which skills would improve the answer. Reply ONLY JSON {"skills": '
    '["<name>", ...]} with at most 2 names taken verbatim from the catalog, '
    'or {"skills": []} if none helps.'
)


class SkillSelector(Protocol):
    def select(self, query: str, skills: list[Skill]) -> list[str]: ...


def _strip_code_fence(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```[a-zA-Z]*\n?", "", content)
        content = re.sub(r"\n?```$", "", content)
    return content


def parse_selection(
    content: str,
    valid_names: set[str],
    max_skills: int = MAX_SKILLS_PER_ANSWER,
) -> list[str]:
    """Parse JSON reply; giữ tên hợp lệ theo thứ tự, dedupe, cap max_skills."""
    try:
        data = json.loads(_strip_code_fence(content))
    except json.JSONDecodeError:
        return []
    raw = data.get("skills") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return []
    picked: list[str] = []
    for item in raw:
        name = item if isinstance(item, str) else ""
        if name in valid_names and name not in picked:
            picked.append(name)
        if len(picked) == max_skills:
            break
    return picked


class OpenRouterSkillSelector:
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_LLM_MODEL,
        base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def select(self, query: str, skills: list[Skill]) -> list[str]:
        if not skills:
            return []
        try:
            catalog = "\n".join(f"- {s.name}: {s.description}" for s in skills)
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Available skills:\n{catalog}\n\nQuestion: {query}",
                    },
                ],
                temperature=0,
            )
            reply = response.choices[0].message.content or ""
        except Exception:
            return []
        return parse_selection(reply, {s.name for s in skills})
