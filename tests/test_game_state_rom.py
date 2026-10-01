"""GameState sur savestates réels, comparé aux valeurs que le jeu affiche lui-même
(écrans relevés sur states/modes/sheet.png : boutique, équipe, sac, Pokédex, combat).
ROM et savestates requis."""

from pathlib import Path

import pytest

from pokeblue.emulator import TICKS_PER_ACTION, Emulator
from pokeblue.emulator.recipes import load_manifest, play
from pokeblue.knowledge.gen1_data import EVENT_IDS, ITEM_IDS, MOVE_IDS, MOVES, SPECIES_IDS
from pokeblue.state.game_state import GameState
from pokeblue.state.mode_detector import Mode, detect_mode

MANIFEST = Path(__file__).resolve().parents[1] / "configs" / "states" / "modes.yaml"
RECIPES = {e.name: e for e in load_manifest(MANIFEST)}


@pytest.fixture(scope="module")
def emulator(rom_path):
    emu = Emulator(rom_path)
    yield emu
    emu.close()


def _state_at(emulator, state_path, base: str, recipe: str = "wait:2") -> GameState:
    emulator.load_state(state_path(base))
    emulator.run(recipe)
    return GameState.from_memory(emulator.snapshot())


def _state_of(emulator, state_path, name: str) -> GameState:
    entry = RECIPES[name]
    play(emulator, entry, Path(state_path(entry.base)).parent)
    return GameState.from_memory(emulator.snapshot())


def test_pewter_values_match_game_screens(emulator, state_path):
    state = _state_at(emulator, state_path, "37_pewter_city.state")
    assert state.money == 3395                                   # « ¥3395 » à la boutique
    (charmander,) = state.party                                  # écran d'équipe
    assert charmander.species == SPECIES_IDS["CHARMANDER"]
    assert (charmander.level, charmander.hp, charmander.max_hp) == (12, 35, 35)
    assert state.bag == (                                        # sac : 2 / 1 / 1
        (ITEM_IDS["POTION"], 2), (ITEM_IDS["POKE_BALL"], 1), (ITEM_IDS["ANTIDOTE"], 1))
    assert state.pokedex_owned.bit_count() == 1                  # Pokédex : OWN 1
    assert state.pokedex_seen.bit_count() == 8                   #           SEEN 8
    assert state.owns(4)                                         # Salamèche


def test_event_flags_tell_the_story_so_far(emulator, state_path):
    state = _state_at(emulator, state_path, "48_pewter_gym_badge.state")
    for event in ("EVENT_GOT_STARTER", "EVENT_GOT_OAKS_PARCEL", "EVENT_GOT_POKEDEX",
                  "EVENT_BEAT_VIRIDIAN_FOREST_TRAINER_2", "EVENT_BEAT_PEWTER_GYM_TRAINER_0"):
        assert state.flag(EVENT_IDS[event]), event
    # Malgré son nom, ce savestate précède le combat contre Pierre.
    assert not state.flag(EVENT_IDS["EVENT_BEAT_BROCK"]) and state.n_badges == 0


def test_battle_values_match_game_screens(emulator, state_path):
    state = _state_of(emulator, state_path, "battle_move_menu")
    assert detect_mode(state) is Mode.BATTLE_MOVE_MENU
    battle = state.battle
    assert battle.is_wild
    assert (battle.enemy.species, battle.enemy.level) == (SPECIES_IDS["METAPOD"], 6)   # « METAPOD :L6 »
    player = battle.player
    assert (player.species, player.level, player.hp, player.max_hp) == (
        SPECIES_IDS["CHARMANDER"], 11, 32, 32)                                         # « :L11 32/32 »
    assert player.moves[:3] == (MOVE_IDS["SCRATCH"], MOVE_IDS["GROWL"], MOVE_IDS["EMBER"])
    assert player.pp[0] == 32 and MOVES[MOVE_IDS["SCRATCH"]].pp == 35                 # « 32/35 »
    assert state.party[battle.player_party_index].species == player.species


def test_scripted_battle_diff(emulator, state_path):
    """Mini-combat guidé par le détecteur de mode : Griffe jusqu'à la fin du combat."""
    prev = _state_of(emulator, state_path, "battle_main_menu")
    exp_before = prev.party[0].exp
    seen = {"enemy_fainted": False, "battle_ended": False}
    for _ in range(400):
        mode = detect_mode(prev)
        if mode is Mode.OVERWORLD:
            break
        if mode is Mode.TRANSITION:
            emulator.tick(TICKS_PER_ACTION)
        else:
            emulator.act("a")
        state = GameState.from_memory(emulator.snapshot())
        diff = state.diff(prev)
        seen["enemy_fainted"] |= diff.enemy_fainted
        seen["battle_ended"] |= diff.battle_ended
        prev = state
    assert seen == {"enemy_fainted": True, "battle_ended": True}
    assert detect_mode(prev) is Mode.OVERWORLD
    assert prev.party[0].exp > exp_before
