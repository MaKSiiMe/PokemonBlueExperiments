"""Énigmes de la progression (puzzles.yaml), données issues de pokered."""

from __future__ import annotations

from functools import cache
from pathlib import Path

import yaml

PUZZLES_FILE = Path(__file__).parent / "puzzles.yaml"


@cache
def puzzle(name: str) -> dict:
    return yaml.safe_load(PUZZLES_FILE.read_text(encoding="utf-8"))[name]
