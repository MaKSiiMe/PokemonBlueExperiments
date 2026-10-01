"""Graphe de progression (progression.yaml) et équipes des dresseurs."""

import dataclasses
from pathlib import Path

import pytest

from pokeblue.knowledge.gen1_data import (
    EVENT_IDS,
    ITEM_IDS,
    MOVE_IDS,
    MOVES,
    SPECIES_IDS,
    TOGGLE_IDS,
    TRAINER_PARTIES,
)
from pokeblue.knowledge.maps import map_names
from pokeblue.knowledge.navigation import Ability, Navigator, Progress
from pokeblue.knowledge.progression import (
    milestone,
    milestones,
    missing_requirements,
    next_milestone,
    required_level,
)
from pokeblue.knowledge.trainers import default_moves, gym_leader_numbers, rival_party, trainer_team
from pokeblue.state import ram_symbols as sym

INITIAL = Progress()   # aucun drapeau, objets activables dans leur état initial


def _check_condition(cond: dict) -> None:
    for e in cond.get("events", []) + cond.get("events_unset", []):
        assert e in EVENT_IDS, e
    for b in cond.get("badges", []):
        assert hasattr(sym, f"BIT_{b}"), b
    for i in cond.get("items", []):
        assert i in ITEM_IDS, i
    for f in cond.get("status_flags1", []):
        assert hasattr(sym, f), f
    for t in cond.get("objects_hidden", []):
        assert t in TOGGLE_IDS, t


def test_milestones_reference_real_data():
    ids = [m.id for m in milestones()]
    assert len(ids) == len(set(ids))
    seen = set()
    for m in milestones():
        assert m.target_map in map_names(), m.id
        assert m.completes_when, m.id
        for cond in m.completes_when:
            _check_condition(cond)
        _check_condition(m.requires)
        assert all(req in seen for req in m.requires_milestones), m.id   # ordre du fichier
        if m.trainer_class:
            parties = TRAINER_PARTIES[m.trainer_class]
            last = (m.rival_base + 2) if m.rival_base else m.trainer_party
            assert 1 <= last <= len(parties), m.id
        seen.add(m.id)


def test_every_milestone_is_checkable_in_ram():
    """Rien n'est accompli en début de partie, tout l'est une fois le jeu terminé."""
    assert not [m.id for m in milestones() if m.done(INITIAL)]
    assert all(m.done(Progress.everything()) for m in milestones())
    assert next_milestone(INITIAL).id == "get_starter"
    assert next_milestone(Progress.everything()) is None


def test_plan_locks_are_covered():
    ids = {m.id for m in milestones()}
    for required in ("get_hm01_cut", "get_hm05_flash", "get_hm03_surf", "get_hm04_strength",
                     "beat_hideout_giovanni", "get_poke_flute", "wake_snorlax", "get_gold_teeth",
                     "get_card_key", "beat_lorelei", "beat_champion"):
        assert required in ids
    assert milestone("get_hm05_flash").optional
    assert milestone("beat_ghost_marowak").requires == {"items": ["SILPH_SCOPE"]}
    lorelei = milestone("beat_lorelei").requires_abilities
    assert Ability.SURF in lorelei and Ability.STRENGTH in lorelei


def test_story_targets_are_reachable_in_order():
    nav = Navigator(Progress.everything())
    previous = "REDS_HOUSE_2F"
    for m in milestones():
        route = nav.search(nav.entry_squares(previous), lambda sq, goal=m.target_map: sq.map == goal)
        assert route is not None, f"{previous} → {m.target_map} ({m.id})"
        previous = m.target_map


def test_required_levels():
    assert required_level("beat_brock") == 14          # Onix N14
    assert required_level("beat_lance") == 62
    assert required_level("beat_champion", "SQUIRTLE") == 65
    levels = [required_level(m) for m in milestones() if not m.optional]
    assert levels[-1] == max(levels)


def test_missing_requirements_name_the_blocker():
    # Badge Cascade et CS01 sans Coupe apprise : le Major Bob est bloqué par l'arbre.
    done = {"get_starter", "rival_oaks_lab", "get_oaks_parcel", "get_pokedex", "beat_brock",
            "cross_mt_moon", "beat_cerulean_rival", "beat_misty", "get_ss_ticket", "get_hm01_cut"}
    events = 0
    for m in milestones():
        if m.id in done:
            for e in m.completes_when[0].get("events", []):
                events |= 1 << EVENT_IDS[e]
    badges = (1 << sym.BIT_BOULDERBADGE) | (1 << sym.BIT_CASCADEBADGE)
    progress = Progress(events=events, badges=badges)
    assert next_milestone(progress).id == "beat_lt_surge"
    assert missing_requirements(milestone("beat_lt_surge"), progress) == ["capacité CUT"]
    with_cut = dataclasses.replace(progress, abilities=Ability.CUT)
    assert missing_requirements(milestone("beat_lt_surge"), with_cut) == []


# ── Dresseurs ─────────────────────────────────────────────────────────────────

def _team(trainer_class, party, starter=None):
    return [(m.name, m.level, [MOVES[x].name for x in m.moves if x])
            for m in trainer_team(trainer_class, party, starter)]


def test_gym_leaders_and_their_special_moves():
    numbers = gym_leader_numbers()
    assert numbers == {("BROCK", 1): 1, ("MISTY", 1): 2, ("LT_SURGE", 1): 3, ("ERIKA", 1): 4,
                       ("KOGA", 1): 5, ("SABRINA", 1): 6, ("BLAINE", 1): 7, ("GIOVANNI", 3): 8}
    brock = _team("BROCK", 1)
    assert [(n, lv) for n, lv, _ in brock] == [("GEODUDE", 12), ("ONIX", 14)]
    assert "BIDE" in brock[1][2]                           # Patience de l'Onix de Pierre
    assert "BUBBLEBEAM" in _team("MISTY", 1)[1][2]


def test_elite_four_and_champion_moves():
    assert "BLIZZARD" in _team("LORELEI", 1)[4][2]          # 5e Pokémon d'Olga
    champion = _team("RIVAL3", rival_party(1, "SQUIRTLE"), "SQUIRTLE")
    assert champion[0][0] == "PIDGEOT" and "SKY_ATTACK" in champion[0][2]
    assert champion[5][0] == "BLASTOISE" and "BLIZZARD" in champion[5][2]
    assert _team("RIVAL3", rival_party(1, "CHARMANDER"), "CHARMANDER")[5][0] == "CHARIZARD"


def test_rival_party_follows_rival_starter():
    assert [rival_party(7, s) for s in ("SQUIRTLE", "BULBASAUR", "CHARMANDER")] == [7, 8, 9]


def test_default_moves_drop_the_oldest_beyond_four():
    charmander = SPECIES_IDS["CHARMANDER"]
    assert default_moves(charmander, 11)[:3] == tuple(MOVE_IDS[m] for m in ("SCRATCH", "GROWL", "EMBER"))
    moves = default_moves(charmander, 30)
    assert MOVE_IDS["SCRATCH"] not in moves and MOVE_IDS["SLASH"] in moves


# ── Sur savestates ────────────────────────────────────────────────────────────

MANIFEST = Path(__file__).resolve().parents[1] / "configs" / "states" / "modes.yaml"


@pytest.fixture(scope="module")
def emulator(rom_path):
    from pokeblue.emulator import Emulator
    emu = Emulator(rom_path)
    yield emu
    emu.close()


@pytest.mark.parametrize(("state", "expected"), [
    ("PokemonBlue.gb.state", "get_starter"),
    ("07_route1_grass.state", "get_oaks_parcel"),
    ("22_route2_down.state", "beat_brock"),
    ("48_pewter_gym_badge.state", "beat_brock"),
])
def test_next_milestone_on_savestates(emulator, state_path, state, expected):
    from pokeblue.state.game_state import GameState
    emulator.load_state(state_path(state))
    emulator.tick(2)
    assert next_milestone(GameState.from_memory(emulator.snapshot())).id == expected


def test_default_moves_match_ram(emulator, state_path):
    """Un Pokémon sauvage et le Salamèche du joueur ont exactement les attaques par défaut."""
    from pokeblue.emulator.recipes import load_manifest, play
    from pokeblue.state.game_state import GameState
    entry = next(e for e in load_manifest(MANIFEST) if e.name == "battle_main_menu")
    play(emulator, entry, Path(state_path(entry.base)).parent)
    state = GameState.from_memory(emulator.snapshot())
    enemy = state.battle.enemy
    assert enemy.moves == default_moves(enemy.species, enemy.level)
    lead = state.party[0]
    assert lead.moves == default_moves(lead.species, lead.level)
