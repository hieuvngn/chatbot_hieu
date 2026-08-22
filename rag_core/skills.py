"""Skills: gói hướng dẫn markdown nạp từ thư mục ``skills/`` của repo.

Mỗi skill là ``skills/<tên>/SKILL.md`` với frontmatter tối thiểu
``name`` + ``description``; phần thân là instructions đưa vào prompt
của Generator khi skill được chọn.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


@dataclass(frozen=True)
class Skill:
    """Một skill: định danh, mô tả (cho selector) và instructions (cho generator)."""

    name: str
    description: str
    instructions: str


def parse_skill_md(text: str) -> Skill | None:
    """Parse SKILL.md: frontmatter ``---`` + thân markdown.

    Trả None khi frontmatter thiếu/sai hoặc thân rỗng. Không dùng pyyaml.
    """
    lines = text.strip().splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        return None
    meta: dict[str, str] = {}
    for line in lines[1:end]:
        key, sep, value = line.partition(":")
        if sep:
            meta[key.strip()] = value.strip()
    name = meta.get("name", "")
    description = meta.get("description", "")
    instructions = "\n".join(lines[end + 1 :]).strip()
    if not name or not description or not instructions:
        return None
    return Skill(name=name, description=description, instructions=instructions)


def load_skills(directory: Path) -> list[Skill]:
    """Nạp mọi ``<dir>/*/SKILL.md``, sort theo name.

    File lỗi bị bỏ qua kèm warning; thư mục vắng/không tồn tại trả [].
    """
    logger = logging.getLogger(__name__)
    if not directory.is_dir():
        return []
    skills: list[Skill] = []
    for path in sorted(directory.glob("*/SKILL.md")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            logger.warning("cannot read skill file: %s", path)
            continue
        skill = parse_skill_md(text)
        if skill is None:
            logger.warning("skipping malformed skill: %s", path)
            continue
        skills.append(skill)
    return sorted(skills, key=lambda s: s.name)
