"""Navigation entre cartes (sans ROM) : trajets, verrous, données YAML."""

import dataclasses
import re
from pathlib import Path

import pytest

from pokeblue.knowledge.gen1_data import EVENT_IDS, ITEM_IDS
from pokeblue.knowledge.maps import load_map, map_names, world_rules
from pokeblue.knowledge.navigation import (
    Ability,
    Navigator,
    Progress,
    Square,
    _last_map_candidates,
    block_events,
    gate_squares,
    gates,
    path,
)
from pokeblue.state import ram_symbols as sym

EVERYTHING = Progress.everything()


def test_pallet_town_to_indigo_plateau():
    route = path("PALLET_TOWN", "INDIGO_PLATEAU")
    assert route is not None
    assert route.maps[0] == "PALLET_TOWN" and route.maps[-1] == "INDIGO_PLATEAU"
    for step in ("VIRIDIAN_CITY", "ROUTE_22_GATE", "ROUTE_23", "VICTORY_ROAD_1F"):
        assert step in route.maps


def test_league_needs_surf():
    progress = dataclasses.replace(EVERYTHING, abilities=EVERYTHING.abilities & ~Ability.SURF)
    assert path("PALLET_TOWN", "INDIGO_PLATEAU", progress) is None


def test_victory_road_switches_need_strength():
    switches = [e["event"] for e in block_events() if e.get("solvable_with") == "STRENGTH"]
    events = (1 << sym.NUM_EVENTS) - 1
    for event in switches:
        events &= ~(1 << EVENT_IDS[event])
    unsolved = dataclasses.replace(EVERYTHING, events=events)
    assert path("PALLET_TOWN", "INDIGO_PLATEAU", unsolved) is not None
    no_strength = dataclasses.replace(unsolved, abilities=Ability.CUT | Ability.SURF)
    assert path("PALLET_TOWN", "INDIGO_PLATEAU", no_strength) is None


def test_league_needs_earth_badge():
    progress = dataclasses.replace(EVERYTHING, badges=0xFF & ~(1 << sym.BIT_EARTHBADGE))
    assert path("PALLET_TOWN", "INDIGO_PLATEAU", progress) is None


def test_new_game_is_stuck_south_of_viridian():
    # Sans Pokédex, le vieil homme endormi bloque la route au nord de Jadielle.
    assert path("PALLET_TOWN", "VIRIDIAN_CITY", Progress()) is not None
    assert path("PALLET_TOWN", "PEWTER_CITY", Progress()) is None


def test_last_map_exits_follow_the_side_you_came_from():
    gate = load_map("ROUTE_22_GATE")
    north = next(w for w in gate.warps if w.y == 0)
    south = next(w for w in gate.warps if w.y == gate.square_height - 1)
    assert _last_map_candidates("ROUTE_22_GATE", (north.x, north.y)) == ("ROUTE_23",)
    assert _last_map_candidates("ROUTE_22_GATE", (south.x, south.y)) == ("ROUTE_22",)


def _known_condition(cond: dict) -> None:
    for event in cond.get("events", []) + cond.get("events_unset", []):
        assert event in EVENT_IDS, event
    for badge in cond.get("badges", []):
        assert hasattr(sym, f"BIT_{badge}"), badge
    for item in cond.get("items", []):
        assert item in ITEM_IDS, item
    for bit in cond.get("status_flags1", []):
        assert hasattr(sym, bit), bit


def test_gates_reference_real_data():
    ids = [g["id"] for g in gates()]
    assert len(ids) == len(set(ids))
    for gate in gates():
        assert gate["map"] in map_names(), gate["id"]
        data = load_map(gate["map"])
        assert all(data.in_bounds(x, y) for x, y in gate_squares(gate)), gate["id"]
        for cond in gate.get("open_when_any", [gate.get("open_when", {})]):
            _known_condition(cond)


def test_block_events_reference_real_data():
    for ev in block_events():
        data = load_map(ev["map"])
        bx, by = ev["block"]
        assert 0 <= bx < data.width and 0 <= by < data.height, ev
        assert ev["event"] in EVENT_IDS, ev
        n_blocks = len(world_rules()["tilesets"][data.tileset]["block_squares"]) // 8
        assert ev["block_id"] < n_blocks, ev


POKERED = Path(__file__).resolve().parents[1] / ".cache" / "pokered"


def _extract_simple_block_events(scripts: Path) -> dict[str, set]:
    """Extraction automatique des remplacements de forme simple :
    CheckEvent E ; ret|jr z/nz ; ld a, $B ; ld [wNewTileBlockID], a ; lb bc, Y, X ;
    predef ReplaceTileBlock. Les formes à sous-programmes sont relues à la main."""
    from pokeblue.knowledge.pokered.asm import logical_lines
    found: dict[str, set] = {}
    for script in scripts.glob("*.asm"):
        event = when = block = coords = pending = skip_label = None
        expect = False
        for line in logical_lines(script):
            if re.fullmatch(r"[A-Za-z]\w*::?", line):   # nouvelle routine : contexte remis à zéro
                event = when = block = coords = skip_label = None
                continue
            if m := re.match(r"^CheckEvent\w*\s+(EVENT_\w+)", line):
                event, when, expect = m.group(1), None, True
                continue
            if expect:
                # `ret cc` ou `jr cc, .label` qui saute PAR-DESSUS le remplacement :
                # z = on saute si l'événement est inactif → remplacement s'il est actif.
                expect = False
                if m := re.match(r"^(ret|jr|jp) (n?z)\b(?:, (\.?\w+))?", line):
                    when = m.group(2) == "z"
                    skip_label = m.group(3)
                continue
            if skip_label and line.rstrip(":") == skip_label:
                when = None   # la cible du saut précède le remplacement : forme non simple
                skip_label = None
            if line == "ld [wNewTileBlockID], a":
                block = pending
            pending = None
            if m := re.fullmatch(r"ld a, \$([0-9a-fA-F]{2})", line):
                pending = int(m.group(1), 16)
            if m := re.fullmatch(r"lb bc, (\d+), (\d+)", line):
                coords = (int(m.group(2)), int(m.group(1)))
            if line.startswith("predef") and "ReplaceTileBlock" in line:
                if when is not None and block is not None:
                    found.setdefault(script.stem, set()).add((event, when, coords, block))
                block = None
    return found


def test_block_events_agree_with_simple_script_extraction():
    roots = list(POKERED.glob("pokered-*/scripts"))
    if not roots:
        pytest.skip("sources pokered absentes du cache")
    labels = {name: load_map(name).label for name in map_names()}
    curated = {
        (labels[ev["map"]], ev["event"], ev["when_set"], tuple(ev["block"]), ev["block_id"])
        for ev in block_events()
    }
    extracted = _extract_simple_block_events(roots[0])
    assert extracted, "aucun remplacement extrait"
    for label, entries in extracted.items():
        for event, when, coords, block in entries:
            assert (label, event, when, coords, block) in curated, (label, event, coords)


def test_underground_path_to_route_7_uses_the_real_header():
    """UndergroundPathRoute7Copy déclare aussi UNDERGROUND_PATH_ROUTE_7 : les données
    doivent venir du header que MapHeaderPointers associe à la carte (warp 5 de Route 7)."""
    assert {w.warp for w in load_map("UNDERGROUND_PATH_ROUTE_7").warps if w.map == "LAST_MAP"} == {5}
    nav = Navigator(Progress())          # sans boisson pour les gardes de Safrania
    found = nav.search([Square("UNDERGROUND_PATH_WEST_EAST", 47, 2)], lambda sq: sq.map == "ROUTE_7")
    assert found is not None and "SAFFRON_CITY" not in found.maps
