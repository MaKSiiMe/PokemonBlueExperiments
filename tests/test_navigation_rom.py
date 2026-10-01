"""Modèle de déplacement comparé à l'émulateur : depuis chaque savestate, un pas dans
chaque direction doit aboutir là où `Navigator.neighbors` l'avait prédit."""

from pathlib import Path

import pytest

from pokeblue.emulator import Emulator
from pokeblue.knowledge.gen1_data import MAPS
from pokeblue.knowledge.navigation import DIRECTIONS, Navigator, Square, gate_squares, gates, path
from pokeblue.state.game_state import GameState
from pokeblue.state.mode_detector import Mode, detect_mode

STATES = sorted(Path("states").glob("*.state")) if Path("states").is_dir() else []


@pytest.fixture(scope="module")
def emulator(rom_path):
    emu = Emulator(rom_path)
    yield emu
    emu.close()


def _gate_squares(map_name: str) -> set[tuple[int, int]]:
    # Le joueur peut poser le pied sur la case d'un verrou avant d'être repoussé :
    # pour la planification ces cases sont infranchissables, on ne les compare pas.
    return {sq for g in gates() if g["map"] == map_name for sq in gate_squares(g)}


@pytest.mark.parametrize("state_file", STATES, ids=[p.stem for p in STATES])
def test_single_steps_match_emulator(emulator, state_file):
    for direction, (dx, dy) in DIRECTIONS.items():
        emulator.load_state(state_file)
        emulator.tick(2)
        before = GameState.from_memory(emulator.snapshot())
        if detect_mode(before) is not Mode.OVERWORLD:
            pytest.skip("écran occupé par une interface")
        here = Square(MAPS[before.map_id].name, before.x, before.y)
        if (here.x + dx, here.y + dy) in _gate_squares(here.map):
            continue
        predicted = [
            sq for sq in Navigator.from_state(before).neighbors(here)
            if sq.map == here.map and (sq.x - here.x, sq.y - here.y) in ((dx, dy), (2 * dx, 2 * dy))
        ]
        emulator.act(direction)
        emulator.tick(40)
        after = GameState.from_memory(emulator.snapshot())
        moved_to = Square(MAPS[after.map_id].name, after.x, after.y)
        if moved_to.map != here.map:
            continue   # warp ou connexion : couverts par test_maps_rom
        if predicted:
            assert moved_to in predicted, (direction, here, moved_to)
        else:
            assert moved_to == here, (direction, here, moved_to)


def test_route_from_savestate_progress(emulator, state_path):
    """Pokédex obtenu, Pierre pas encore battu : Argenta oui, la Route 3 non."""
    emulator.load_state(state_path("22_route2_down.state"))
    emulator.tick(2)
    state = GameState.from_memory(emulator.snapshot())
    nav = Navigator.from_state(state)
    start = Square(MAPS[state.map_id].name, state.x, state.y)
    to_pewter = nav.search([start], lambda sq: sq.map == "PEWTER_CITY")
    assert to_pewter is not None and "VIRIDIAN_FOREST" in to_pewter.maps
    assert nav.search([start], lambda sq: sq.map == "ROUTE_3") is None
    assert path("PALLET_TOWN", "ROUTE_3") is not None   # ouvert une fois tout débloqué
