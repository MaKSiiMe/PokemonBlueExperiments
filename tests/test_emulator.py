"""Wrapper d'émulateur : recettes d'inputs (sans ROM) et durées d'appui (ROM)."""

import pytest

from pokeblue.emulator import Emulator, parse_recipe
from pokeblue.state import ram_symbols as sym
from pokeblue.state.game_state import GameState
from pokeblue.state.mode_detector import Mode, detect_mode


def test_parse_recipe():
    assert parse_recipe("wait:30 up*2 hold:left:6 a") == [
        ("wait", "", 30), ("act", "up", 1), ("act", "up", 1), ("hold", "left", 6), ("act", "a", 1),
    ]
    assert parse_recipe(["start", "b"]) == [("act", "start", 1), ("act", "b", 1)]
    for bad in ("jump", "up*x", "hold:left", "wait:soon"):
        with pytest.raises(ValueError):
            parse_recipe(bad)


@pytest.fixture(scope="module")
def emulator(rom_path):
    emu = Emulator(rom_path)
    yield emu
    emu.close()


def test_direction_action_moves_exactly_one_tile(emulator, state_path):
    """Régression : une direction maintenue 23 frames faisait deux pas."""
    emulator.load_state(state_path("37_pewter_city.state"))
    emulator.tick(30)
    x0 = emulator.snapshot()[sym.W_X_COORD]
    for step in range(1, 4):
        emulator.act("left")
        assert emulator.snapshot()[sym.W_X_COORD] == x0 - step


@pytest.mark.parametrize("phase", range(8))
def test_button_action_is_never_dropped(emulator, state_path, phase):
    """Régression : le jeu lit la manette une frame sur deux, un appui d'une frame
    était perdu une fois sur deux. L'action doit ouvrir le menu à toute phase."""
    emulator.load_state(state_path("37_pewter_city.state"))
    emulator.tick(30 + phase)
    emulator.act("start")
    emulator.tick(10)
    assert detect_mode(GameState.from_memory(emulator.snapshot())) is Mode.MENU


def test_savestate_round_trip(emulator, state_path):
    emulator.load_state(state_path("37_pewter_city.state"))
    saved = emulator.save_state()
    before = emulator.snapshot()
    emulator.run("left left up")
    assert emulator.snapshot() != before
    emulator.load_state(saved)
    assert emulator.snapshot().wram == before.wram
