"""Données de cartes générées (pokered) comparées à la RAM du jeu, sur tous les
savestates de states/ : tileset, dimensions, warps, connexions et tuiles affichées."""

from pathlib import Path

import pytest

from pokeblue.emulator import Emulator
from pokeblue.knowledge.gen1_data import MAP_IDS
from pokeblue.knowledge.maps import LAST_MAP, map_by_id, world_rules
from pokeblue.state import ram_symbols as sym
from pokeblue.state.game_state import GameState
from pokeblue.state.mode_detector import Mode, detect_mode

STATES = sorted(Path("states").glob("*.state")) if Path("states").is_dir() else []
LAST_MAP_ID = 0xFF   # warp_event : destination -1 = wLastMap
DIRECTIONS = {"north": "NORTH", "south": "SOUTH", "west": "WEST", "east": "EAST"}
PLAYER_SCREEN_SQUARE = (4, 4)   # le joueur est affiché aux tuiles (8, 8)–(9, 9)


@pytest.fixture(scope="module")
def emulator(rom_path):
    emu = Emulator(rom_path)
    yield emu
    emu.close()


@pytest.fixture(params=STATES, ids=[p.stem for p in STATES])
def loaded(request, emulator):
    emulator.load_state(request.param)
    emulator.tick(2)
    mem = emulator.snapshot()
    return mem, map_by_id(mem[sym.W_CUR_MAP])


def test_header_matches_ram(loaded):
    mem, data = loaded
    assert world_rules()["tilesets"][data.tileset]["id"] == mem[sym.W_CUR_MAP_TILESET]
    assert (data.width, data.height) == (mem[sym.W_CUR_MAP_WIDTH], mem[sym.W_CUR_MAP_HEIGHT])


def test_warps_match_ram(loaded):
    mem, data = loaded
    assert mem[sym.W_NUMBER_OF_WARPS] == len(data.warps)
    for i, warp in enumerate(data.warps):
        y, x, dest_warp, dest_map = mem.read(sym.W_WARP_ENTRIES + 4 * i, 4)
        expected_map = LAST_MAP_ID if warp.map == LAST_MAP else MAP_IDS[warp.map]
        assert (x, y, dest_warp + 1, dest_map) == (warp.x, warp.y, warp.warp, expected_map)


def test_connections_match_ram(loaded):
    mem, data = loaded
    flags = mem[sym.W_CUR_MAP_CONNECTIONS]
    for direction, const in DIRECTIONS.items():
        connection = data.connection(direction)
        assert bool(flags & getattr(sym, const)) == (connection is not None), direction
        if connection is None:
            continue
        assert mem[getattr(sym, f"W_{const}_CONNECTED_MAP")] == MAP_IDS[connection.map]
        # Alignement (macro connection) : -2 × décalage sur l'axe parallèle au bord.
        axis = "X" if direction in ("north", "south") else "Y"
        alignment = mem[getattr(sym, f"W_{const}_CONNECTED_MAP_{axis}_ALIGNMENT")]
        assert alignment == (-2 * connection.offset) & 0xFF


def test_visible_tiles_match_map_data(loaded, emulator):
    mem, data = loaded
    state = GameState.from_memory(mem)
    if detect_mode(state) is not Mode.OVERWORLD:
        pytest.skip("écran occupé par une interface")
    px, py = PLAYER_SCREEN_SQUARE
    compared = 0
    for j in range(sym.SCREEN_HEIGHT // 2):
        for i in range(sym.SCREEN_WIDTH // 2):
            x, y = state.x - px + i, state.y - py + j
            if not data.in_bounds(x, y):
                continue
            screen_tile = state.tilemap[(2 * j + 1) * sym.SCREEN_WIDTH + 2 * i]
            assert screen_tile == data.tile(x, y), f"case ({x}, {y})"
            compared += 1
    assert compared > 0
