"""Détection de mode : règles sur écrans synthétiques, puis validation sur le jeu de
savestates étiquetés à la main (configs/states/modes.yaml, ROM requise)."""

import dataclasses
from pathlib import Path

import pytest

from pokeblue.emulator.recipes import load_manifest
from pokeblue.knowledge.gen1_data import CHARMAP, FADE_PALETTES
from pokeblue.state import ram_symbols as sym
from pokeblue.state.game_state import GameState
from pokeblue.state.memory import MemorySnapshot
from pokeblue.state.mode_detector import Mode, detect_mode, steady_bgp
from pokeblue.state.screen import TEXT_BOX_TOP, Screen

MANIFEST = Path(__file__).resolve().parents[1] / "configs" / "states" / "modes.yaml"
W, H = sym.SCREEN_WIDTH, sym.SCREEN_HEIGHT
NORMAL_BGP = FADE_PALETTES[3][0]   # FadePal4


# ── Écrans synthétiques ───────────────────────────────────────────────────────

def _blank_state(**changes) -> GameState:
    mem = MemorySnapshot.blank({sym.R_LCDC: 1 << sym.B_LCDC_ENABLE, sym.R_BGP: NORMAL_BGP})
    return dataclasses.replace(GameState.from_memory(mem), **changes)


def _draw_box(tiles: bytearray, left: int, top: int, right: int, bottom: int) -> None:
    for x in range(left + 1, right):
        tiles[top * W + x] = tiles[bottom * W + x] = CHARMAP["─"]
    for y in range(top + 1, bottom):
        tiles[y * W + left] = tiles[y * W + right] = CHARMAP["│"]
    tiles[top * W + left], tiles[top * W + right] = CHARMAP["┌"], CHARMAP["┐"]
    tiles[bottom * W + left], tiles[bottom * W + right] = CHARMAP["└"], CHARMAP["┘"]


def _write(tiles: bytearray, x: int, y: int, text: str) -> None:
    for i, char in enumerate(text):
        tiles[y * W + x + i] = CHARMAP[char]


def _text_box(text: str = "Hello") -> bytearray:
    tiles = bytearray(range(0x30)) * 8          # décor : tuiles de jeu de tuiles
    tiles = tiles[:W * H]
    _draw_box(tiles, 0, TEXT_BOX_TOP, W - 1, H - 1)
    _write(tiles, 1, TEXT_BOX_TOP + 2, text)
    return tiles


def test_overworld_screen():
    tiles = bytes(bytearray(range(0x30)) * 8)[:W * H]
    assert detect_mode(_blank_state(tilemap=tiles)) is Mode.OVERWORLD


def test_dialog_and_menu_screens():
    tiles = _text_box()
    assert Screen(bytes(tiles)).has_text_box
    assert detect_mode(_blank_state(tilemap=bytes(tiles))) is Mode.DIALOG
    _write(tiles, 1, TEXT_BOX_TOP + 4, "▶YES")
    assert detect_mode(_blank_state(tilemap=bytes(tiles))) is Mode.MENU


def test_battle_screens():
    battle = GameState.from_memory(MemorySnapshot.blank({sym.W_IS_IN_BATTLE: sym.WILD_BATTLE})).battle
    tiles = _text_box("Wild PIDGEY")
    assert detect_mode(_blank_state(tilemap=bytes(tiles), battle=battle)) is Mode.BATTLE_ANIM
    _write(tiles, 9, TEXT_BOX_TOP + 2, "▶FIGHT")
    assert detect_mode(_blank_state(tilemap=bytes(tiles), battle=battle)) is Mode.BATTLE_MENU
    _write(tiles, 1, TEXT_BOX_TOP - 3, "TYPE/")
    assert detect_mode(_blank_state(tilemap=bytes(tiles), battle=battle)) is Mode.BATTLE_MOVE_MENU
    # En combat, sans boîte de texte : spirale d'entrée en combat.
    no_box = bytes(bytearray(W * H))
    assert detect_mode(_blank_state(tilemap=no_box, battle=battle)) is Mode.TRANSITION


def test_transition_from_palette_or_lcd():
    tiles = bytes(_text_box())
    assert detect_mode(_blank_state(tilemap=tiles, lcd_on=False)) is Mode.TRANSITION
    assert detect_mode(_blank_state(tilemap=tiles, bgp=FADE_PALETTES[1][0])) is Mode.TRANSITION


def test_dark_map_palette_is_steady():
    # Grotte sombre sans Flash : wMapPalOffset = 6, la palette FadePal2 est la norme.
    dark = FADE_PALETTES[1][0]
    assert steady_bgp(6) == dark and steady_bgp(0) == NORMAL_BGP
    state = _blank_state(map_pal_offset=6, bgp=dark)
    assert detect_mode(state) is Mode.OVERWORLD
    assert detect_mode(dataclasses.replace(state, bgp=NORMAL_BGP)) is Mode.TRANSITION


# ── Savestates étiquetés ──────────────────────────────────────────────────────

ENTRIES = load_manifest(MANIFEST)


def test_manifest_covers_every_mode():
    assert {e.mode for e in ENTRIES} == {m.value for m in Mode}


@pytest.fixture(scope="module")
def emulator(rom_path):
    from pokeblue.emulator import Emulator
    emu = Emulator(rom_path)
    yield emu
    emu.close()


@pytest.mark.parametrize("entry", ENTRIES, ids=[e.name for e in ENTRIES])
def test_labelled_savestates(emulator, state_path, entry):
    from pokeblue.emulator.recipes import play
    states_dir = Path(state_path(entry.base)).parent
    play(emulator, entry, states_dir)
    state = GameState.from_memory(emulator.snapshot())
    assert detect_mode(state).value == entry.mode, entry.note
