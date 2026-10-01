"""Données d'énigmes (puzzles.yaml) recoupées avec les sources pokered."""

import re
from pathlib import Path

import pytest
import yaml

from pokeblue.knowledge.pokered.source import POKERED_COMMIT

PUZZLES = Path(__file__).parents[1] / "src/pokeblue/knowledge/puzzles.yaml"
POKERED = Path(__file__).parents[1] / f".cache/pokered/pokered-{POKERED_COMMIT}"

pytestmark = pytest.mark.skipif(not POKERED.exists(), reason="sources pokered absentes (.cache)")


def test_vermilion_trash_cans_match_pokered():
    data = yaml.safe_load(PUZZLES.read_text(encoding="utf-8"))["vermilion_trash_cans"]
    hidden = (POKERED / "data/events/hidden_events.asm").read_text()
    cans = [(int(x), int(y), int(i)) for x, y, i in
            re.findall(r"hidden_event\s+(\d+),\s+(\d+),\s+GymTrashScript,\s+(\d+)", hidden)]
    assert [(x, y) for x, y, _ in cans] == [tuple(c) for c in data["cans"]]
    assert [i for _, _, i in cans] == list(range(len(cans)))

    trash = (POKERED / "engine/events/hidden_events/vermilion_gym_trash.asm").read_text()
    table = trash.split("GymTrashCans:")[1]
    rows = re.findall(r"^\s*db\s+(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*;\s*(\d+)", table, re.M)
    assert len(rows) == len(cans)
    for row, expected in zip(rows, data["second_lock_candidates"], strict=True):
        count, *candidates, _ = map(int, row)
        assert candidates[:count] == expected
    assert "and $e" in trash.split(".trySecondLock")[-1]   # nouveau 1er interrupteur : indice pair
